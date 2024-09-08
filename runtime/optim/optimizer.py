import torch
from . import optim_oop as optim
import time

from collections import deque  # Efficient ring buffer implementation.

class Version:
    def __init__(self, version=0):
        self.version = version

    def __repr__(self):
        return "v%d" % self.version

    def incr(self):
        return Version(version=self.version+1)

class OptimizerWithWeightStashing(object):
    """Optimizer class with weight stashing.

    Arguments:
        - optim_name: the name of optimizer, required to create the corresponding
                      base_optimizer (optim.{optim_name}).
        - optimizer_args: the keyword arguments passed to base_optimizer.
    """

    def __init__(self, optim_name, modules, master_parameters, model_parameters,
                 loss_scale, num_versions, verbose_freq=0,
                 **optimizer_args):
        self.modules = modules
        self.master_parameters = master_parameters
        self.model_parameters = model_parameters  # model_parameters is None if not fp16.
        self.loss_scale = loss_scale

        self.num_versions = num_versions
        if len(list(master_parameters)) == 0:
            self.base_optimizer = NoParamsOptim()
        else:
            self.base_optimizer = getattr(optim, optim_name)(
                [{'params': module.parameters()} for module in modules], **optimizer_args)
        self.latest_version = Version()
        self.current_version = Version()
        self.initialize_queue()
        self.verbose_freq = verbose_freq
        self.batch_counter = 0

    def __getstate__(self):
        return self.base_optimizer.__getstate__()

    def __setstate__(self, state):
        self.base_optimizer.__setstate__(state)

    def __repr__(self):
        return self.base_optimizer.__repr__()

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    def __getattr__(self, key):
        """Relay the unknown key to base_optimizer."""
        return getattr(self.base_optimizer, key)

    def initialize_queue(self):
        self.queue = deque(maxlen=self.num_versions)
        for _ in range(self.num_versions):
            self.queue.append(self.get_params(clone=True))
        self.buffered_state_dicts = self.queue[0][0]

    @torch.no_grad()
    def get_params(self, clone):
        if clone:
            state_dicts = []
            for module in self.modules:
                state_dict = {}
                for key, v in module.state_dict().items():
                    state_dict[key] = v.detach().clone()
                state_dicts.append(state_dict)
        else:
            for i, module in enumerate(self.modules):
                state_dict = module.state_dict()
                for key in state_dict:
                    # Running_mean and running_var for batchnorm layers should
                    # accumulate normally.
                    if "running_" in key:
                        continue
                    if "mask" in key:
                        self.buffered_state_dicts[i][key] = state_dict[key].detach().clone()
                    else:
                        self.buffered_state_dicts[i][key].copy_(state_dict[key])
            state_dicts = self.buffered_state_dicts
        return state_dicts, self.latest_version

    @torch.no_grad()
    def set_params(self, state_dicts, version):
        for (state_dict, module) in zip(state_dicts, self.modules):
            for key, v in state_dict.items():
                if "running_" in key:
                    continue
                obj = module
                names = key.split('.')
                for name in names:
                    obj = getattr(obj, name)
                obj.data = v

        self.current_version = version

    def load_old_params(self):
        if self.num_versions > 1:
            self.set_params(*self.queue[0])

    def load_new_params(self):
        if self.num_versions > 1:
            self.set_params(*self.queue[-1])

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                                          and returns the loss.
        """
        log_timing = self.verbose_freq > 0 and self.batch_counter % self.verbose_freq == 0
        if log_timing:
            start_time = time.time()
        if self.model_parameters is not None:
            import apex.fp16_utils as fp16_utils
            fp16_utils.model_grads_to_master_grads(self.model_parameters,
                                                   self.master_parameters)
            if self.loss_scale != 1.0:
                for parameter in self.master_parameters:
                    parameter.grad.data = parameter.grad.data / self.loss_scale

        for group in self.base_optimizer.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data += p.data / (lr * self.num_versions)

        self.latest_version = self.latest_version.incr()
        if self.num_versions > 1:
            self.buffered_state_dicts = self.queue[0][0]
            self.queue.append(self.get_params(clone=False))
            self.set_params(*self.queue[-1])

        for group in self.base_optimizer.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data -= p.data / (lr * self.num_versions)

        loss = self.base_optimizer.step()
        if self.model_parameters is not None:
            import apex.fp16_utils as fp16_utils
            fp16_utils.master_params_to_model_params(self.model_parameters,
                                                     self.master_parameters)

        if log_timing:
            print("Optimizer step took: %.3f" % (time.time() - start_time))
        self.batch_counter += 1
        return loss

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups


class NoParamsOptim(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = []
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        pass

    def __repr__(self):
        return ''

    def state_dict(self):
        return None

    def load_state_dict(self, state_dict):
        pass

    def zero_grad(self):
        pass

    def step(self):
        pass

    def add_param_group(self, param_group):
        pass

