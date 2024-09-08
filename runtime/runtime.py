# Copyright (c) 2024 Ryb7532.
# Licensed under the MIT license.


import collections
import itertools
import os
import warnings
import torch
import torch.distributed as dist
from optim.optimizer import OptimizerWithWeightStashing

IMAGE_CLASSIFICATION = "image_classification"
TRANSLATION = "translation"
SPEECH_TO_TEXT = "speech_to_text"

NCCL = "nccl"
GLOO = "gloo"

class ModulesWithDependencies:
    def __init__(self, modules_with_dependencies):
        self._modules = []
        self._all_input_names = []
        self._all_output_names = []
        for (module, input_names, output_names) in modules_with_dependencies:
            self._modules.append(module)
            self._all_input_names.append(input_names)
            self._all_output_names.append(output_names)

    def modules(self):
        return self._modules

    def all_input_names(self):
        return self._all_input_names

    def all_output_names(self):
        return self._all_output_names

    def is_input_tensor(self, tensor_name):
        for module_input_names in self._all_input_names:
            if tensor_name in module_input_names:
                return True
        return False



class StageRuntime:
    def __init__(self, model, distributed_backend, fp16, loss_scale,
                 global_batch_size, tensor_shapes, training_tensor_dtypes,
                 inputs_module_destinations, target_tensor_names,
                 configuration_maps, master_addr, rank, model_type):
        self.tensors = []
        self.gradients = {}
        self.fp16 = fp16
        self.loss_scale = loss_scale
        self.global_batch_size = global_batch_size
        self.tensor_shapes = tensor_shapes
        self.training_tensor_dtypes = training_tensor_dtypes
        self.model_type = model_type
        self.target_tensor_names = target_tensor_names
        self.enable_recompute = False

        self.initialize(model, distributed_backend, inputs_module_destinations,
                        configuration_maps, master_addr, rank)

        self.forward_only = False

    def initialize(self, model, distributed_backend, inputs_module_destinations,
                   configuration_maps, master_addr, rank):
        self.send_ranks = []
        self.receive_ranks = []
        self.rank = rank
        self.stage = None
        self.criterion_input_name = str(model[-1][1][0])
        self.group = None
        self.recv_for_works = []
        self.send_for_works = []
        self.recv_back_works = []
        self.send_back_works = []
        self.optimizer = None

        module_to_stage_map = configuration_maps['module_to_stage_map']
        stage_to_rank_map = configuration_maps['stage_to_rank_map']
        stage_to_depth_map = configuration_maps['stage_to_depth_map']
        recompute_stage = configuration_maps['recompute_stage']

        assert len(module_to_stage_map) == len(model)
        assert self.rank is not None

        stage_to_module_map = collections.defaultdict(list)
        for module in range(len(module_to_stage_map)):
            stage_to_module_map[module_to_stage_map[module]].append(module)

        rank_to_stage_map = {}
        for stage in stage_to_rank_map:
            for rank in stage_to_rank_map[stage]:
                rank_to_stage_map[rank] = stage

        assert 0 <= self.rank < len(rank_to_stage_map)
        num_ranks = len(rank_to_stage_map)
        self.world_size=num_ranks
        self.num_stages = len(stage_to_module_map)
        self.stage = rank_to_stage_map[self.rank]
        self.rank_in_stage = stage_to_rank_map[self.stage].index(self.rank)
        num_ranks_in_stage = len(stage_to_rank_map[self.stage])
        self.num_ranks_in_previous_stage = 0
        ranks_in_previous_stage = []
        if self.stage > 0:
            self.num_ranks_in_previous_stage = len(
                stage_to_rank_map[self.stage - 1])
            ranks_in_previous_stage = stage_to_rank_map[self.stage - 1]
        self.num_ranks_in_next_stage = 0
        ranks_in_next_stage = []
        if self.stage < self.num_stages - 1:
            self.num_ranks_in_next_stage = len(
                stage_to_rank_map[self.stage + 1])
            ranks_in_next_stage = stage_to_rank_map[self.stage + 1]
        self.module_ids = stage_to_module_map[self.stage]
        self.modules_with_dependencies = ModulesWithDependencies(
            [model[module] for module in self.module_ids])
        self.is_criterion = self.stage == (self.num_stages - 1)
        if stage_to_depth_map is not None:
            self.num_warmup_minibatches = stage_to_depth_map[str(self.stage)]
        else:
            self.num_warmup_minibatches = self.num_stages - 1 - self.stage
        if recompute_stage is not None and self.stage in recompute_stage:
            self.enable_recompute = True
        if self.stage is None or (self.stage == (self.num_stages-1)):
            self.enable_recompute = False

        for i in range(len(model)):
            for j in range(i+1, len(model)):
                for tensor_name in model[i][2]:
                    if tensor_name in model[j][1]:
                        if module_to_stage_map[i] == \
                            module_to_stage_map[j]:
                            continue
                        if module_to_stage_map[j] == self.stage:
                            self.receive_ranks.append(tensor_name)
                        if module_to_stage_map[i] == self.stage:
                            self.send_ranks.append(tensor_name)
                        if module_to_stage_map[i] < self.stage and self.stage < module_to_stage_map[j]:
                            self.receive_ranks.append(tensor_name)
                            self.send_ranks.append(tensor_name)

        for model_inputs in inputs_module_destinations.keys():
            destination_stage = module_to_stage_map[
                inputs_module_destinations[model_inputs]]
            if self.stage < destination_stage:
                self.send_ranks.append(model_inputs)

            if self.stage <= destination_stage:
                self.receive_ranks.append(model_inputs)

        for target_tensor_name in self.target_tensor_names:
            self.receive_ranks.append(target_tensor_name)
            if self.num_ranks_in_next_stage > 0:
                self.send_ranks.append(target_tensor_name)

        self.previous_targets = []
        self.previous_splits = []
        self.next_targets = []
        self.next_splits = []

        num_data_this_rank = (self.global_batch_size+self.rank_in_stage)//num_ranks_in_stage
        if self.num_ranks_in_previous_stage != 0:
            n = self.num_ranks_in_previous_stage
            num_data_in_rank = {i: (self.global_batch_size+i)//n for i in range(n)}
            first_iter = 0
            idx = 0
            for i in range(self.rank_in_stage):
                first_iter += (self.global_batch_size+i)//num_ranks_in_stage
            while True:
                first_iter -= num_data_in_rank[idx]
                if first_iter < 0:
                    num_data_in_rank[idx] = -first_iter
                    break
                idx += 1
            num_data_this = num_data_this_rank
            while num_data_this != 0:
                self.previous_targets.append(ranks_in_previous_stage[idx])
                num_data = min(num_data_this, num_data_in_rank[idx])
                self.previous_splits.append(num_data)
                num_data_this -= num_data
                idx += 1
        if self.num_ranks_in_next_stage != 0:
            n = self.num_ranks_in_next_stage
            num_data_in_rank = {i: (self.global_batch_size+i)//n for i in range(n)}
            first_iter = 0
            idx = 0
            for i in range(self.rank_in_stage):
                first_iter += (self.global_batch_size+i)//num_ranks_in_stage
            while True:
                first_iter -= num_data_in_rank[idx]
                if first_iter < 0:
                    num_data_in_rank[idx] = -first_iter
                    break
                idx += 1
            num_data_this = num_data_this_rank
            while num_data_this != 0:
                self.next_targets.append(ranks_in_next_stage[idx])
                num_data = min(num_data_this, num_data_in_rank[idx])
                self.next_splits.append(num_data)
                num_data_this -= num_data
                idx += 1

        for shape in self.tensor_shapes.values():
            shape[0] = num_data_this_rank

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i] = modules[i].cuda()
            if self.fp16:
                import apex.fp16_utils as fp16_utils
                module[i] = fp16_utils.BN_convert_float(module[i].half())

        master_port = '12345'
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        dist.init_process_group(distributed_backend,
                                init_method="tcp://{}:{}".format(master_addr, master_port),
                                rank=self.rank, world_size=num_ranks)
        self.default_group = dist.GroupMember.WORLD

        self.process_groups = None
        if distributed_backend is not NCCL:
            self.process_groups = {}
            for i in range(self.num_stages-1):
                num_ranks_in_stage = len(stage_to_rank_map[i])
                num_ranks_in_next_stage = len(stage_to_rank_map[i+1])
                data_id_in_stage = []
                _sum = 0
                for j in range(num_ranks_in_stage):
                    num_data = (self.global_batch_size+j)//num_ranks_in_stage
                    data_id_in_stage.append((_sum, _sum+num_data))
                data_id_in_next_stage = []
                _sum = 0
                for j in range(num_ranks_in_next_stage):
                    num_data = (self.global_batch_size+j)//num_ranks_in_next_stage
                    data_id_in_next_stage.append((_sum, _sum+num_data))
                for j, (sj, ej) in enumerate(data_id_in_stage):
                    rank_j = stage_to_rank_map[i][j]
                    self.process_groups[rank_j] = {}
                    for k, (sk, ek) in enumerate(data_id_in_next_stage):
                        if not (sj >= ek or sk >= ej):
                            rank_k = stage_to_rank_map[i+1][k]
                            self.process_groups[rank_j][rank_k] = dist.new_group(ranks=[rank_j, rank_k])

        if stage_to_rank_map is not None:
            groups = []
            for stage in range(self.num_stages):
                ranks = stage_to_rank_map[stage]
                if len(ranks) > 1:
                    groups.append(dist.new_group(ranks=ranks))
                else:
                    groups.append(None)
            self.group = groups[self.stage]

        with torch.no_grad():
            num_parameters = 0
            works = []
            for i in range(len(modules)):
                if self.group is not None:
                    if ((i < (len(modules)-1) and self.is_criterion)
                        or not self.is_criterion):
                        src_rank = stage_to_rank_map[self.stage][0]
                        for param in modules[i].parameters():
                            num_parameters += param.numel()
                            work = dist.broadcast(param.detach(), src_rank, self.group, async_op=True)
                            works.append(work)
            for w in works:
                w.wait()
            if num_ranks_in_stage > 1:
                module_size = 4. * num_parameters
                print("Replicating stage: ranks=%d, module_size=%.3f\n" % (
                    num_ranks_in_stage, module_size))

        if self.fp16:
            self.master_parameters = []
            self.model_parameters = []
            for i in range(len(modules)):
                import apex.fp16_utils as fp16_utils
                module_parameters, module_master_parameters = \
                    fp16_utils.prep_param_lists(modules[i])
                self.master_parameters.extend(module_master_parameters)
                self.model_parameters.extend(module_parameters)
        else:
            self.master_parameters = list(self.parameters())
            self.model_parameters = None

    @property
    def target(self):
        return self.tensors[-1]["target"]

    def modules(self):
        return self.modules_with_dependencies.modules()

    def parameters(self):
        parameter_iterators = []
        for module in self.modules_with_dependencies.modules():
            parameter_iterators.append(module.parameters())
        return itertools.chain(*parameter_iterators)

    def state_dict(self):
        state_dict = collections.OrderedDict()
        for i, module in zip(self.module_ids, self.modules_with_dependencies.modules()):
            state_dict["module%d" % i] = module.state_dict()
        if self.fp16:
            state_dict["master_parameters"] = self.master_parameters
        return state_dict

    def load_state_dict(self, state_dict):
        for i, module in zip(self.module_ids, self.modules_with_dependencies.modules()):
            module.load_state_dict(state_dict["module%d" % i])
        if self.fp16:
            saved_master_parameters = state_dict["master_parameters"]
            for master_parameter, saved_master_parameter in zip(
                self.master_parameters, saved_master_parameters):
                master_parameter.data.copy_(saved_master_parameter.data)

    def cuda(self):
        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i] = modules[i].cuda()

    def zero_grad(self):
        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].zero_grad()

    def train(self):
        self.tensors = []
        self.gradients = {}
        self.output = None
        self.forward_only = False

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].train()

    def eval(self):
        self.tensors = []
        self.gradients = {}
        self.output = None
        self.forward_only = True

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].eval()

    def set_loader(self, loader):
        if loader is not None:
            self.loader_iter = iter(loader)
        else:
            self.loader_iter = None

    def receive_tensors_forward_async(self, preprocess=None):
        if self.forward_only and len(self.tensors) > 0:
            self.tensors.pop(0)
        self.tensors.append({})
        if self.loader_iter is not None:
            input = next(self.loader_iter)
            if self.model_type == TRANSLATION:
                (input, target) = input
                src, src_length = input
                tgt, tgt_length = target

                self.tensors[-1]["input0"] = src.cuda(non_blocking=True)
                self.tensors[-1]["input1"] = torch.LongTensor(src_length).cuda(
                    non_blocking=True)
                self.tensors[-1]["input2"] = tgt[:-1].cuda(non_blocking=True)
                self.tensors[-1]["target"] = tgt[1:].cuda().contiguous().view(-1)
                self.tensors[-1]["target_length"] = \
                    torch.tensor([int(sum(torch.LongTensor(tgt_length) - 1))],
                                 dtype=torch.int).cuda()
            elif self.model_type == IMAGE_CLASSIFICATION:
                (input, target) = input
                if self.fp16:
                    input = input.half()
                self.tensors[-1]["input0"] = input.cuda(non_blocking=True)
                self.tensors[-1]["target"] = target.cuda(non_blocking=True)
            elif self.model_type == SPEECH_TO_TEXT:
                input, target, input_percentages, target_sizes = input
                input_sizes = input_percentages.mul_(int(input.size(3))).int()
                self.tensors[-1]["input0"] = input.cuda(non_blocking=True)
                self.tensors[-1]["input1"] = input_sizes.cuda(non_blocking=True)
                self.tensors[-1]["target"] = target.cuda(non_blocking=True)
                self.tensors[-1]["target_length"] = target_sizes.cuda(
                    non_blocking=True)
            if preprocess is not None:
                preprocess(self.tensors[-1])
        elif self.process_groups is None:
            for input_name in self.receive_ranks:
                q = []
                tensor_shape = self.tensor_shapes[input_name]
                dtype = self.training_tensor_dtypes[input_name]
                recv_tensor = torch.empty(tensor_shape, dtype=dtype, device='cuda')
                recv_split = recv_tensor.split(self.previous_splits)
                for src_rank, tensor in zip(self.previous_targets, recv_split):
                    work = dist.irecv(tensor=tensor,
                        src=src_rank)
                    q.append(work)
                self.recv_for_works.append((input_name, recv_tensor, q))
        else:
            for input_name in self.receive_ranks:
                q = []
                tensor_shape = self.tensor_shapes[input_name]
                dtype = self.training_tensor_dtypes[input_name]
                recv_tensor = torch.empty(tensor_shape, dtype=dtype, device='cuda')
                recv_split = recv_tensor.split(self.previous_splits)
                for src_rank, tensor in zip(self.previous_targets, recv_split):
                    group = self.process_groups[src_rank][self.rank]
                    work = dist.broadcast(tensor=tensor,
                        group=group,
                        src=src_rank,
                        async_op=True)
                    q.append(work)
                self.recv_for_works.append((input_name, recv_tensor, q))

    def receive_tensors_forward_wait(self):
        for input_name, tensor, queue in self.recv_for_works:
            for work in queue:
                work.wait()
            if tensor.dtype == torch.float32:
                tensor = tensor.requires_grad_()
            self.tensors[-1][input_name] = tensor
        self.recv_for_works = []

    def receive_tensors_forward(self, preprocess=None):
        self.receive_tensors_forward_async(preprocess)
        self.receive_tensors_forward_wait()

    def send_tensors_forward_async(self):
        if self.num_ranks_in_next_stage == 0:
            return
        if self.process_groups is None:
            for output_name in self.send_ranks:
                copy_tensor = self.tensors[-1][output_name].detach().clone()
                list_send_tensor = list(copy_tensor.split(self.next_splits))
                for dst_rank, tensor in zip(self.next_targets, list_send_tensor):
                    work = dist.isend(tensor=tensor.contiguous(),
                        dst=dst_rank)
                    self.send_for_works.append(work)
        else:
            for output_name in self.send_ranks:
                copy_tensor = self.tensors[-1][output_name].detach().clone()
                list_send_tensor = list(copy_tensor.split(self.next_splits))
                for dst_rank, tensor in zip(self.next_targets, list_send_tensor):
                    group = self.process_groups[self.rank][dst_rank]
                    work = dist.broadcast(tensor=tensor.contiguous(),
                        group=group,
                        src=self.rank,
                        async_op=True)
                    self.send_for_works.append(work)

    def send_tensors_forward_wait(self):
        for work in self.send_for_works:
            work.wait()
        self.send_for_works = []
        if self.enable_recompute:
            stored_tensors = {}
            for name, tensor in self.tensors[-1].items():
                if name in self.receive_ranks:
                    stored_tensors[name] = tensor
            self.tensors[-1] = stored_tensors

    def send_tensors_forward(self):
        self.send_tensors_forward_async()
        self.send_tensors_forward_wait()

    def receive_tensors_backward_async(self):
        if self.num_ranks_in_next_stage == 0:
            return
        if self.process_groups is None:
            for output_name in self.send_ranks:
                if output_name in self.target_tensor_names:
                    continue
                q = []
                tensor_shape = self.tensor_shapes[output_name]
                dtype = self.training_tensor_dtypes[output_name]
                recv_tensor = torch.empty(tensor_shape, dtype=dtype, device='cuda')
                recv_split = recv_tensor.split(self.next_splits)
                for src_rank, tensor in zip(self.next_targets, recv_split):
                    work = dist.irecv(tensor=tensor,
                        src=src_rank)
                    q.append(work)
                self.recv_back_works.append((output_name, recv_tensor, q))
        else:
            for output_name in self.send_ranks:
                if output_name in self.target_tensor_names:
                    continue
                q = []
                tensor_shape = self.tensor_shapes[output_name]
                dtype = self.training_tensor_dtypes[output_name]
                recv_tensor = torch.empty(tensor_shape, dtype=dtype, device='cuda')
                recv_split = recv_tensor.split(self.next_splits)
                for src_rank, tensor in zip(self.next_targets, recv_split):
                    group = self.process_groups[self.rank][src_rank]
                    work = dist.broadcast(tensor=tensor,
                        group=group,
                        src=src_rank,
                        async_op=True)
                    q.append(work)
                self.recv_back_works.append((output_name, recv_tensor, q))

    def receive_tensors_backward_wait(self):
        for output_name, tensor, queue in self.recv_back_works:
            for work in queue:
                work.wait()
            self.gradients[output_name] = tensor
        self.recv_back_works = []

    def receive_tensors_backward(self):
        self.receive_tensors_backward_async()
        self.receive_tensors_backward_wait()

    def send_tensors_backward_async(self):
        if self.num_ranks_in_previous_stage == 0:
            return
        if self.process_groups is None:
            for input_name in self.receive_ranks:
                if input_name in self.target_tensor_names:
                    continue
                tensor = self.gradients[input_name].detach().clone()
                list_send_tensor = list(tensor.split(self.previous_splits))
                for dst_rank, gradient in zip(self.previous_targets, list_send_tensor):
                    work = dist.isend(tensor=gradient.contiguous(),
                        dst=dst_rank)
                    self.send_back_works.append(work)
        else:
            for input_name in self.receive_ranks:
                if input_name in self.target_tensor_names:
                    continue
                tensor = self.gradients[input_name].detach().clone()
                list_send_tensor = list(tensor.split(self.previous_splits))
                for dst_rank, gradient in zip(self.previous_targets, list_send_tensor):
                    group = self.process_groups[dst_rank][self.rank]
                    work = dist.broadcast(tensor=gradient.contiguous(),
                        group=group,
                        src=self.rank,
                        async_op=True)
                    self.send_back_works.append(work)

    def send_tensors_backward_wait(self):
        for work in self.send_back_works:
            work.wait()
        self.send_back_works = []
        self.gradients = {}

    def send_tensors_backward(self):
        self.send_tensors_backward_async()
        self.send_tensors_backward_wait()

    def synchronize_replicas(self):
        if self.group is None:
            return
        for module in self.modules_with_dependencies.modules():
            for param in module.parameters():
                dist.all_reduce(param.grad.data, group=self.group, op=dist.ReduceOp.SUM)

    def run_forward(self, preprocess=None):
        self.receive_tensors_forward(preprocess)
        tensors = self.tensors[-1]
        self._run_forward(tensors, False)
        self.send_tensors_forward()

    def _run_forward(self, tensors, recomp=False):
        no_store = self.enable_recompute and not recomp
        modules = self.modules_with_dependencies.modules()
        all_input_names = self.modules_with_dependencies.all_input_names()
        all_output_names = self.modules_with_dependencies.all_output_names()
        if no_store:
            with torch.no_grad():
                for i, (module, input_names, output_names) in \
                        enumerate(zip(modules, all_input_names, all_output_names)):
                    if i == (len(modules) - 1) and self.is_criterion:
                        if self.model_type == SPEECH_TO_TEXT:
                            output = tensors["output"].transpose(0, 1).float()
                            output_sizes = tensors["output_sizes"].cpu()
                            target = tensors["target"].cpu()
                            target_sizes = tensors["target_length"].cpu()
                            input0_size = tensors["input0_size"].cpu()
                            module_outputs = [module(output, target, output_sizes, target_sizes) / input0_size[0]]
                        else:
                            module_outputs = [module(tensors[input_name],
                                                        tensors["target"])
                                                for input_name in input_names]
                            module_outputs = [sum(module_outputs)]
                    else:
                        module_outputs = module(*[tensors[input_name]
                                                for input_name in input_names])
                        if not isinstance(module_outputs, tuple):
                            module_outputs = (module_outputs,)
                        module_outputs = list(module_outputs)

                    for (output_name, module_output) in zip(output_names, module_outputs):
                        tensors[output_name] = module_output
        else:
            for i, (module, input_names, output_names) in \
                    enumerate(zip(modules, all_input_names, all_output_names)):
                if i == (len(modules) - 1) and self.is_criterion:
                    if self.model_type == SPEECH_TO_TEXT:
                        output = tensors["output"].transpose(0, 1).float()
                        output_sizes = tensors["output_sizes"].cpu()
                        target = tensors["target"].cpu()
                        target_sizes = tensors["target_length"].cpu()
                        input0_size = tensors["input0_size"].cpu()
                        module_outputs = [module(output, target, output_sizes, target_sizes) / input0_size[0]]
                    else:
                        module_outputs = [module(tensors[input_name],
                                                    tensors["target"])
                                            for input_name in input_names]
                        module_outputs = [sum(module_outputs)]
                else:
                    module_outputs = module(*[tensors[input_name]
                                            for input_name in input_names])
                    if not isinstance(module_outputs, tuple):
                        module_outputs = (module_outputs,)
                    module_outputs = list(module_outputs)

                for (output_name, module_output) in zip(output_names, module_outputs):
                    tensors[output_name] = module_output

        if self.is_criterion:
            self.output = tensors[input_names[0]]
        if self.is_criterion and self.model_type == TRANSLATION:
            loss_per_batch = tensors[output_names[0]] * tensors[self.criterion_input_name].size(1)
            loss_per_token = loss_per_batch / tensors["target_length"][0].item()
            self.loss = loss_per_token
        elif self.is_criterion:
            tensors[output_names[0]] /= self.global_batch_size
            self.loss = tensors[output_names[0]]
        else:
            self.loss = 1

    def run_backward(self):
        self.receive_tensors_backward_async()
        if self.enable_recompute:
            self._run_forward(self.tensors[0], True)
        self.receive_tensors_backward_wait()
        self._run_backward()
        self.synchronize_replicas()
        self.send_tensors_backward()

    def _run_backward(self):
        inputs = {}
        outputs = {}
        gradients = {}
        output_gradients = {}

        all_input_names_set = set()
        all_output_names_set = set()

        modules = self.modules_with_dependencies.modules()
        all_input_names = self.modules_with_dependencies.all_input_names()
        all_output_names = self.modules_with_dependencies.all_output_names()

        for (input_names, output_names) in zip(all_input_names, all_output_names):
            for input_name in input_names:
                all_input_names_set.add(input_name)
            for output_name in output_names:
                all_output_names_set.add(output_name)

        tensors = self.tensors.pop(0)
        for (_, input_names, output_names) in \
            zip(reversed(modules), reversed(all_input_names), reversed(all_output_names)):
            for output_name in output_names:
                if output_name not in all_input_names_set:
                    if output_name not in self.gradients:
                        output_gradients[output_name] = None
                    else:
                        output_gradients[output_name] = self.gradients[output_name]
                    if tensors[output_name].requires_grad:
                        outputs[output_name] = tensors[output_name]
            for input_name in input_names:
                if input_name not in all_output_names_set:
                    inputs[input_name] = tensors[input_name]

        def hook_wrapper(input_name):
            def hook(input_gradient):
                gradients[input_name] = input_gradient
            return hook

        for input_name in inputs:
            if input_name != "input0" and input_name != "input1" and input_name != "input2" \
                    and inputs[input_name].requires_grad:
                inputs[input_name].register_hook(hook_wrapper(input_name))

        if "loss" in outputs:
            outputs["loss"] *= self.loss_scale

        torch.autograd.backward(tuple([outputs[output_name] for output_name in outputs]),
                                grad_tensors=tuple([output_gradients[output_name]
                                                    for output_name in outputs]))

        for input_name in inputs:
            if not inputs[input_name].requires_grad:
                self.gradients[input_name] = inputs[input_name]
                continue

            if input_name != "input0" and input_name != "input1" and input_name != "input2" and input_name != "input":
                self.gradients[input_name] = gradients[input_name]

    def set_optimizer(self, optimizer):
        assert isinstance(optimizer, OptimizerWithWeightStashing), "Use an optimizer with weight stashing instead of a naive optimizer."
        self.optimizer = optimizer


    def run_warmups(self, num_step, preprocess=None):
        for _ in range(num_step):
            self.run_forward(preprocess)
        self.receive_tensors_forward(preprocess)

    def run_one_step(self, preprocess=None, is_not_last=False):
        if self.optimizer is None:
            warnings.warn("No optimizer is set.")
        self._run_forward(self.tensors[-1], False)

        self.optimizer.zero_grad()
        self.optimizer.load_old_params()

        self.send_tensors_forward_async()
        self.receive_tensors_backward_async()
        if self.enable_recompute:
            self._run_forward(self.tensors[0], True)
        self.send_tensors_forward_wait()
        self.receive_tensors_backward_wait()

        self._run_backward()
        self.synchronize_replicas()

        self.optimizer.load_new_params()
        self.optimizer.step()

        if is_not_last:
            self.receive_tensors_forward_async(preprocess)
        self.send_tensors_backward_async()
        if is_not_last:
            self.receive_tensors_forward_wait()
        self.send_tensors_backward_wait()

    def run_cooldown(self):
        self.optimizer.zero_grad()
        self.optimizer.load_old_params()
        self.run_backward()
        self.optimizer.load_new_params()
        self.optimizer.step()

    def run_cooldowns(self, num_step):
        for _ in range(num_step):
            self.run_cooldown()

    def num_tokens(self):
        return self.tensors[-1]["target_length"][0].item()
