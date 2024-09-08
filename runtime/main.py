# Copyright (c) 2024 Ryb7532.
# Licensed under the MIT license.


import argparse
import importlib
import json
import os
import sys
import time
import math
import codecs

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from runtime import StageRuntime, IMAGE_CLASSIFICATION
from labelsmoothing import LabelSmoothingCrossEntropy
from preprocess import cutout

sys.stdout.flush()

backend = 'nccl'

parser = argparse.ArgumentParser(description='PyTorch Image Classification Training')
parser.add_argument('--data_dir', type=str,
                    help='path to dataset')
parser.add_argument('--module', '-m', required=True,
                    help='name of module that contains model and tensor_shapes definition')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--end_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on switch parallels)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--optim_config', '-o', default=None, type=str,
                    help='path of optimizer\'s configuration file')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--fp16', action='store_true',
                    help='train model in fp16 precision')
parser.add_argument('--loss_scale', type=float, default=1,
                    help='static loss scale, positive power of 2 to improve fp16 convergence')
parser.add_argument('--master_addr', default=None, type=str,
                    help="IP address of master (machine with rank 0)")
parser.add_argument('--config_path', default=None, type=str,
                    help="Path of configuration file")
parser.add_argument('--no_input_pipelining', action='store_true',
                    help="No pipelining of inputs")
parser.add_argument('--rank', default=None, type=int,
                    help="Rank of worker")
parser.add_argument('--local_rank', default=0, type=int,
                    help="Local rank of worker")
parser.add_argument('--forward_only', action='store_true',
                    help="Run forward pass only")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint_dir', default='', type=str, metavar='PATH',
                    help='path to directory to save checkpoints')
parser.add_argument('-s', '--synthetic_data', action='store_true',
                    help="Use synthetic data")
parser.add_argument('-v', '--verbose_frequency', default=0, type=int, metavar='N',
                    help="Log verbose information")

best_prec1 = 0


class SyntheticDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, input_size, length, num_classes=1000):
        self.tensor = Variable(torch.rand(*input_size)).type(torch.FloatTensor)
        self.target = torch.Tensor(1).random_(0, num_classes)[0].type(torch.LongTensor)
        self.length = length

    def __getitem__(self, index):
        return self.tensor, self.target

    def __len__(self):
        return self.length


def is_first_stage():
    return args.stage is None or (args.stage == 0)

def is_last_stage():
    return args.stage is None or (args.stage == (args.num_stages-1))



def main():
    global args, best_prec1
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)

    torch.cuda.synchronize()
    start = time.time()
    print("\t\n%.6lf : Rank %d : start\n" % (start, args.rank))

    # Use 'sum' for criterion, not 'mean'
    # criterion = nn.CrossEntropyLoss(reduction='sum')
    criterion = LabelSmoothingCrossEntropy(reduction='sum')

    module = importlib.import_module(args.module)
    args.arch = module.arch()
    model = module.model(criterion)

    if args.arch == 'inception_v3':
        input_size = [1, 3, 299, 299]
    else:
        input_size = [1, 3, 224, 224]
    tensor_shapes = {"input0": input_size, "target": [1]}
    dtypes = {"input0": torch.int64, "target": torch.int64}
    inputs_module_destinations = {"input0": 0}
    target_tensor_names = {"target"}

    for (stage, inputs, outputs) in model[:-1]:  # Skip last module (loss).
        input_tensors = []
        for input in inputs:
            input_tensor = torch.zeros(tuple(tensor_shapes[input]),
                                       dtype=torch.float32)
            input_tensors.append(input_tensor)
        with torch.no_grad():
            output_tensors = stage(*tuple(input_tensors))
        if not type(output_tensors) is tuple:
            output_tensors = [output_tensors]
        for output, output_tensor in zip(outputs,
                                         list(output_tensors)):
            tensor_shapes[output] = list(output_tensor.size())
            dtypes[output] = output_tensor.dtype

    configuration_maps = {
        'module_to_stage_map': None,
        'stage_to_rank_map': None,
        'stage_to_depth_map': None,
        'recompute_stage': None
    }
    if args.config_path is not None:
        json_config_file = json.load(open(args.config_path, 'r'))
        configuration_maps['module_to_stage_map'] = json_config_file.get("module_to_stage_map", None)
        configuration_maps['stage_to_rank_map'] = json_config_file.get("stage_to_rank_map", None)
        configuration_maps['stage_to_rank_map'] = {
            int(k): v for (k,v) in configuration_maps['stage_to_rank_map'].items()}
        configuration_maps['stage_to_depth_map'] = json_config_file.get("stage_to_depth_map", None)
        configuration_maps['recompute_stage'] = json_config_file.get('recompute_stage', None)

    r = StageRuntime(
        model=model, distributed_backend=backend,
        fp16=args.fp16, loss_scale=args.loss_scale,
        global_batch_size=args.batch_size,
        tensor_shapes=tensor_shapes,
        training_tensor_dtypes=dtypes,
        inputs_module_destinations=inputs_module_destinations,
        target_tensor_names=target_tensor_names,
        configuration_maps=configuration_maps,
        master_addr=args.master_addr, rank=args.rank,
        model_type=IMAGE_CLASSIFICATION)

    del model

    args.stage = r.stage
    args.num_stages = r.num_stages
    if not is_first_stage():
        args.workers = 0
        args.synthetic_data = True

    if args.no_input_pipelining:
        num_versions = 1
    else:
        num_versions = r.num_warmup_minibatches + 1

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint_file_path = args.resume
        else:
            checkpoint_file_path = "%s.%d.pth.tar" % (args.resume, r.stage)
        assert os.path.isfile(checkpoint_file_path)
        print("=> loading checkpoint '{}'".format(checkpoint_file_path))
        checkpoint = torch.load(checkpoint_file_path, map_location=torch.device('cuda'))
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        r.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint_file_path, checkpoint['epoch']))

    optim_name = None
    optim_args = {'lr': 0.1}
    if args.optim_config is not None:
        json_optim_config = json.load(open(args.optim_config, 'r'))
        optim_name = json_optim_config['func']
        args.lr_policy = json_optim_config.get('lr_policy', None)
        args.lr_warmup = json_optim_config.get('lr_warmup', 0)
        for k, v in json_optim_config['args'].items():
            if isinstance(v, list):
                v = tuple(v)
            optim_args[k] = v

    if optim_name == 'Adam':
        from optim import adam
        optimizer = adam.AdamWithWeightStashing(r.modules(), r.master_parameters,
                        r.model_parameters, args.loss_scale,
                        num_versions=num_versions,
                        verbose_freq=args.verbose_frequency,
                        **optim_args)
    elif optim_name == 'RMSprop':
        from optim import rmsprop
        optimizer = rmsprop.RMSpropWithWeightStashing(r.modules(), r.master_parameters,
                        r.model_parameters, args.loss_scale,
                        num_versions=num_versions,
                        verbose_freq=args.verbose_frequency,
                        **optim_args)
    else:
        from optim import sgd
        optimizer = sgd.SGDWithWeightStashing(r.modules(), r.master_parameters,
                        r.model_parameters, args.loss_scale,
                        num_versions=num_versions,
                        verbose_freq=args.verbose_frequency,
                        **optim_args)
    args.lr = optim_args['lr']

    if args.resume:
        optim_state_dict = checkpoint['optimizer']
        param_groups = optim_state_dict['param_groups']
        # TODO: This loading might not be proper. It should label optim's
        # param_group with module_id for general parallel switching.
        # Now we support only switching DP to HP.
        if len(param_groups) > len(r.module_ids):
            new_state = {}
            new_param_groups = []
            for i in r.module_ids:
                assert i < len(param_groups)
                new_param_groups.append(param_groups[i])
                for p in param_groups[i]['params']:
                    new_state[p] = optim_state_dict['state'][p]
            optim_state_dict['state'] = new_state
            optim_state_dict['param_groups'] = new_param_groups
        optimizer.load_state_dict(optim_state_dict)

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.arch == 'inception_v3':
        if args.synthetic_data:
            train_dataset = SyntheticDataset((3, 299, 299), 1281167)
        else:
            traindir = os.path.join(args.data_dir, 'train')
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(299),
                    transforms.ToTensor(),
                    normalize,
                ])
            )
    else:
        if args.synthetic_data:
            train_dataset = SyntheticDataset((3, 224, 224), 1281167)
        else:
            traindir = os.path.join(args.data_dir, 'train')
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

    if args.synthetic_data:
        val_dataset = SyntheticDataset((3, 224, 224), 50000)
    else:
        valdir = os.path.join(args.data_dir, 'val')
        val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    distributed_sampler = False
    train_sampler = None
    val_sampler = None
    batch_size = args.batch_size
    if configuration_maps['stage_to_rank_map'] is not None:
        ranks_in_first_stage = configuration_maps['stage_to_rank_map'][0]
        num_ranks_in_first_stage = len(ranks_in_first_stage)
        if num_ranks_in_first_stage > 1 and args.rank in ranks_in_first_stage:
            assert batch_size % num_ranks_in_first_stage == 0
            batch_size = batch_size // num_ranks_in_first_stage
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=num_ranks_in_first_stage,
                rank=args.rank, shuffle=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, num_replicas=num_ranks_in_first_stage,
                rank=args.rank, shuffle=True)
            distributed_sampler = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)

    if args.resume:
        assert args.start_epoch > 0
        validate(val_loader, r, args.start_epoch-1)

    torch.cuda.synchronize()
    end = time.time()
    print("Initialize_time (Rank %d): %.6f seconds\n" % (args.rank, end-start))

    if args.end_epoch == 0:
        args.end_epoch = args.epochs
    avg_train_time = 0
    for epoch in range(args.start_epoch, args.end_epoch):
        if distributed_sampler:
            train_sampler.set_epoch(epoch)

        if args.forward_only:
            validate(val_loader, r, epoch)
        else:
            train_time = train(train_loader, r, optimizer, epoch)
            avg_train_time += train_time

            prec1 = validate(val_loader, r, epoch)

            best_prec1 = max(prec1, best_prec1)
            should_save_checkpoint = args.checkpoint_dir and r.rank_in_stage == 0
            if should_save_checkpoint:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': r.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, args.checkpoint_dir, r.stage, r.num_stages>1)

    torch.cuda.synchronize()
    end = time.time()
    total_time = end - start
    num_epoch = args.end_epoch - args.start_epoch
    print("Average epoch_time (Rank %d): %.6f seconds\n" % (args.rank, avg_train_time/num_epoch))
    print("Total time (Rank %d): %d minutes %.6f seconds\n" % (args.rank, total_time // 60, total_time % 60))



def train(train_loader, r: StageRuntime, optimizer, epoch):
    total_data = 0
    total_prec1 = 0
    total_prec5 = 0

    n = len(train_loader)
    r.train()
    if not is_first_stage(): train_loader = None
    r.set_loader(train_loader)
    r.set_optimizer(optimizer)

    if args.no_input_pipelining:
        num_warmup_minibatches = 0
    else:
        num_warmup_minibatches = r.num_warmup_minibatches

    if args.verbose_frequency > 0:
        print("Letting in %d warm-up minibatches\n" % num_warmup_minibatches)
        print("Running training for %d minibatches\n" % n)

    def img_cutout(tensors):
        x, y = cutout(tensors["input0"], tensors["target"], 224 // 4, prob=epoch / args.epochs)
        tensors["input0"] = x
        tensors["target"] = y

    preprocess = img_cutout if is_first_stage() else None

    torch.cuda.synchronize()
    start = time.time()

    r.run_warmups(num_warmup_minibatches, preprocess)

    for i in range(num_warmup_minibatches, n):
        adjust_learning_rate(r.optimizer, epoch, args.epochs, args.lr_policy, i-num_warmup_minibatches, n)
        if is_last_stage():
            target = r.target

        r.run_one_step(preprocess=preprocess, is_not_last=(i != n-1))

        if is_last_stage():
            output, loss = r.output, r.loss
            prec1, prec5, batch_size = accuracy(output, target, topk=(1, 5))
            total_prec1 += int(prec1)
            total_prec5 += int(prec5)
            total_data += int(batch_size)

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss: {loss:.4f}\t'
                      'Prec@1: {prec1}/{data} ({top1:.3f})\t'
                      'Prec@5: {prec5}/{data} ({top5:.3f})\n'.format(
                        epoch, i, n, loss=float(loss), prec1=total_prec1, prec5=total_prec5, data=total_data, top1=total_prec1/total_data, top5=total_prec5/total_data))

    for i in range(num_warmup_minibatches):
        adjust_learning_rate(r.optimizer, epoch, args.epochs, args.lr_policy, n-num_warmup_minibatches+i, n)
        r.run_cooldown()

    torch.cuda.synchronize()
    end = time.time()
    print("Epoch %d (Rank %d): %.6f seconds\n" % (epoch, args.rank, end - start))

    return end - start



def validate(val_loader, r, epoch):
    total_data = 0
    total_prec1 = 0
    total_prec5 = 0

    n = len(val_loader)
    r.eval()
    if not is_first_stage(): val_loader = None
    r.set_loader(val_loader)

    if args.no_input_pipelining:
        num_warmup_minibatches = 0
    else:
        num_warmup_minibatches = r.num_warmup_minibatches

    if args.verbose_frequency > 0:
        print("Letting in %d warm-up minibatches\n" % num_warmup_minibatches)
        print("Running training for %d minibatches\n" % n)

    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for i in range(n):
            r.run_forward()

            if is_last_stage():
                output, target = r.output, r.target
                prec1, prec5, batch_size = accuracy(output, target, topk=(1, 5))
                total_prec1 += int(prec1)
                total_prec5 += int(prec5)
                total_data += int(batch_size)

                if i % args.print_freq == 0:
                    print('Test: [{0}][{1}/{2}]\t'
                          'Prec@1: {prec1}/{data} ({top1:.3f})\t'
                          'Prec@5: {prec5}/{data} ({top5:.3f})\n'.format(
                           epoch, i, n, prec1=total_prec1, prec5=total_prec5, data=total_data, top1=total_prec1/total_data, top5=total_prec5/total_data))

        if is_last_stage():
            print(' * Prec@1: {prec1}/{data} ({top1:.3f})\t Prec@5: {prec5}/{data} ({top5:.3f})\n'
              .format(prec1=total_prec1, prec5=total_prec5, data=total_data, top1=total_prec1/total_data, top5=total_prec5/total_data))

    torch.cuda.synchronize()
    end = time.time()
    print("Test %d (Rank %d): %.6f seconds\n" % (epoch, args.rank, end - start))

    return total_prec1/total_data if is_last_stage() else 0



def save_checkpoint(state, checkpoint_dir, stage, hasStages=False):
    assert os.path.isdir(checkpoint_dir)
    file_name = "checkpoint.%d.pth.tar" % stage if hasStages else "checkpoint.pth.tar"
    checkpoint_file_path = os.path.join(checkpoint_dir, file_name)
    torch.save(state, checkpoint_file_path)
    print("Saved checkpoint to %s" % checkpoint_file_path)



def adjust_learning_rate(optimizer, epoch, total_epochs, lr_policy, step, epoch_length):
    """ Adjusts learning rate based on stage, epoch, and policy.
    Gets learning rate for stage from runtime and adjusts based on policy.
    Supported LR policies:
         - step
         - polynomial decay
         - exponential decay
    """
    stage_base_lr = args.lr

    if epoch < args.lr_warmup:
        lr = stage_base_lr * float(1 + step + epoch*epoch_length)/float(args.lr_warmup*epoch_length)

    else:
        if lr_policy == "step":
            lr = stage_base_lr * (0.1 ** (epoch // 30))
        elif lr_policy == "polynomial":
            power = 2.0
            lr = stage_base_lr * ((1.0 - (float(epoch) / float(total_epochs))) ** power)
        elif lr_policy == "exponential_decay":
            decay_rate = 0.97
            lr = stage_base_lr * (decay_rate ** (float(epoch) / float(total_epochs)))
        elif lr_policy == "cosine":
            min_lr = 0.0
            lr = (stage_base_lr - min_lr) * (1 + math.cos(float(epoch) / float(total_epochs) * math.pi)) / 2 + min_lr
        else:
            raise NotImplementedError

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k[0])
        res.append(batch_size)
        return res



if __name__ == '__main__':
    main()
