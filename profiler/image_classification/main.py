# Copyright (c) 2024 Ryb7532.
# Licensed under the MIT license.


import sys; sys.path = [".."] + sys.path
import torchmodules.torchgraph as torchgraph
import torchmodules.torchprofiler as torchprofiler
import torchmodules.torchsummary as torchsummary

import argparse
from collections import OrderedDict
import os
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import models.alexnet as alexnet
import models.vgg as vgg
import models.resnet as resnet
import models.densenet as densenet
import models.inception as inception
import models.mobilenet as mobilenet
import models.resnext as resnext
import models.squeezenet as squeezenet
import models.efficientnetv2 as efficientnetv2
import models.vision_transformer as vision_transformer

model_names = sorted(name for name in alexnet.__dict__
    if name.islower() and not name.startswith("_")
    and callable(alexnet.__dict__[name]))
model_names += sorted(name for name in vgg.__dict__
    if name.islower() and not name.startswith("_")
    and callable(vgg.__dict__[name]))
model_names += sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("_")
    and callable(resnet.__dict__[name]))
model_names += sorted(name for name in densenet.__dict__
    if name.islower() and not name.startswith("_")
    and callable(densenet.__dict__[name]))
model_names += sorted(name for name in inception.__dict__
    if name.islower() and not name.startswith("_")
    and callable(inception.__dict__[name]))
model_names += sorted(name for name in mobilenet.__dict__
    if name.islower() and not name.startswith("_")
    and callable(mobilenet.__dict__[name]))
model_names += sorted(name for name in resnext.__dict__
    if name.islower() and not name.startswith("_")
    and callable(resnext.__dict__[name]))
model_names += sorted(name for name in squeezenet.__dict__
    if name.islower() and not name.startswith("_")
    and callable(squeezenet.__dict__[name]))
model_names += sorted(name for name in efficientnetv2.__dict__
    if name.islower() and not name.startswith("_")
    and callable(efficientnetv2.__dict__[name]))
model_names += sorted(name for name in vision_transformer.__dict__
    if name.islower() and not name.startswith("_")
    and callable(vision_transformer.__dict__[name]))

parser = argparse.ArgumentParser(description='Profile Image Classification')
parser.add_argument('--data_dir', type=str,
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--profile_directory', default="profiles/",
                    help="Profile directory")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-s', '--synthetic_data', action='store_true',
                    help="Use synthetic data")
parser.add_argument('-v', '--verbose', action='store_true',
                    help="Controls verbosity while profiling")
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

module_whitelist = ["MultiheadAttention"]


def create_graph(model, train_loader, summary, directory):
    """Given a model, creates and visualizes the computation DAG
       of the model in the passed-in directory."""
    graph_creator = torchgraph.GraphCreator(model, summary, module_whitelist)
    graph_creator.hook_modules(model)
    for i, (input, _) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        model(input)
        if i >= 0:
            break
    graph_creator.unhook_modules()
    graph_creator.persist_graph(directory)


class SyntheticDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, input_size, length, num_classes=1000):
        self.tensor = Variable(torch.rand(*input_size)).type(torch.FloatTensor)
        self.target = torch.Tensor(1).random_(0, num_classes)[0].type(torch.LongTensor)
        self.length = length

    def __getitem__(self, index):
        return self.tensor, self.target

    def __len__(self):
        return self.length


def main():
    global args
    args = parser.parse_args()

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch.startswith('alexnet'):
            model = alexnet.__dict__[args.arch]()
        elif args.arch.startswith('vgg'):
            model = vgg.__dict__[args.arch]()
        elif args.arch.startswith('resnet'):
            model = resnet.__dict__[args.arch]()
        elif args.arch.startswith('densenet'):
            model = densenet.__dict__[args.arch]()
        elif args.arch.startswith('inception_v3'):
            model = inception.__dict__[args.arch]()
        elif args.arch.startswith('mobilenet'):
            model = mobilenet.__dict__[args.arch]()
        elif args.arch.startswith('resnext'):
            model = resnext.__dict__[args.arch]()
        elif args.arch.startswith('squeezenet'):
            model = squeezenet.__dict__[args.arch]()
        elif args.arch.startswith('efficientnetv2'):
            model = efficientnetv2.__dict__[args.arch]()
        elif args.arch.startswith('vit'):
            model = vision_transformer.__dict__[args.arch](dropout=0.1)
        else:
            if args.arch not in models.__dict__:
                raise Exception("Invalid model architecture")
            model = models.__dict__[args.arch]()

    model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.data_dir is not None:
        traindir = os.path.join(args.data_dir, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.arch == 'inception_v3':
        if args.synthetic_data:
            train_dataset = SyntheticDataset((3, 299, 299), 1000000)
        else:
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(299),
                    transforms.ToTensor(),
                    normalize,
                ]))
    else:
        if args.synthetic_data:
            train_dataset = SyntheticDataset((3, 224, 224), 1000000)
        else:
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)

    print("Collecting profile...\n")
    for i, (model_input, _) in enumerate(train_loader):
        model_input = model_input.cuda()
        if i >= 0:
            break
    summary = torchsummary.summary(model=model, module_whitelist=module_whitelist, model_input=(model_input,),
                                    verbose=args.verbose, device="cuda")
    per_layer_times, data_time, h2d_time = profile_train(train_loader, model, criterion, optimizer)

    summary_i = 0
    per_layer_times_i = 0
    last_summary_i = 0
    max_summary_i = len(summary)
    used_summary = [False for _ in range(max_summary_i)]
    while summary_i < max_summary_i and per_layer_times_i < len(per_layer_times):
        summary_elem = summary[summary_i]
        per_layer_time = per_layer_times[per_layer_times_i]
        if str(summary_elem['layer_name']) != str(per_layer_time[0]):
            if not used_summary[summary_i]:
                summary_elem['forward_time'] = 0.0
                summary_elem['backward_time'] = 0.0
            summary_i += 1
            continue
        summary_elem['forward_time'] = per_layer_time[1]
        summary_elem['backward_time'] = per_layer_time[2]
        used_summary[summary_i] = True
        if summary_i != last_summary_i:
            summary_i = last_summary_i
        else:
            while (summary_i < max_summary_i and used_summary[summary_i]):
                summary_i += 1
            last_summary_i = summary_i
        per_layer_times_i += 1
    summary.append(OrderedDict())
    summary[-1]['layer_name'] = 'Input0'
    summary[-1]['forward_time'] = h2d_time
    summary[-1]['backward_time'] = data_time
    summary[-1]['nb_params'] = 0.0
    summary[-1]['output_shape'] = [args.batch_size] + list(model_input.size()[1:])
    create_graph(model, train_loader, summary,
                    os.path.join(args.profile_directory, args.arch))
    print("\n...done!\n")
    return


def profile_train(train_loader, model, criterion, optimizer):
    batch_time_meter = AverageMeter()
    NUM_STEPS_TO_PROFILE = 100  # profile 100 steps or minibatches

    # switch to train mode
    model.train()

    layer_timestamps = []
    data_times = []
    h2d_times = []

    start_time = time.time()
    for i, (input, target) in enumerate(train_loader):
        start = time.time()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        torch.cuda.synchronize()
        h2d_time = time.time() - start

        optimizer.zero_grad()

        with torchprofiler.Profiling(model, module_whitelist=module_whitelist) as p:
            # compute output
            output = model(input)
            if isinstance(output, tuple):
                loss = sum((criterion(output_elem, target) for output_elem in output))
            else:
                loss = criterion(output, target)

            # compute gradient
            loss.backward()

        optimizer.step()

        end_time = time.time()
        iteration_time = end_time - start_time
        batch_time_meter.update(iteration_time)

        if i > NUM_STEPS_TO_PROFILE:
            break
        if i != 0:
            layer_timestamps.append(p.processed_times())
            h2d_times.append(h2d_time)

        if args.verbose:
            print('End-to-end time: {batch_time.val:.3f} s ({batch_time.avg:.3f} s)'.format(
                  batch_time=batch_time_meter))

        start_time = time.time()

    start_time = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time = time.time()
        data_times.append(data_time - start_time)
        start_time = data_time
        if i >= NUM_STEPS_TO_PROFILE:
            break

    layer_times = []
    tot_accounted_time = 0.0
    if args.verbose:
        print("\n==========================================================")
        print("Layer Type    Forward Time (ms)    Backward Time (ms)")
        print("==========================================================")

    for i in range(len(layer_timestamps[0])):
        layer_type = str(layer_timestamps[0][i][0])
        layer_forward_time_sum = 0.0
        layer_backward_time_sum = 0.0
        for j in range(len(layer_timestamps)):
            layer_forward_time_sum += (layer_timestamps[j][i][1] / 1000)
            layer_backward_time_sum += (layer_timestamps[j][i][3] / 1000)
        layer_times.append((layer_type, layer_forward_time_sum / len(layer_timestamps),
                                    layer_backward_time_sum / len(layer_timestamps)))
        if args.verbose:
            print(layer_times[-1][0], layer_times[-1][1], layer_times[-1][2])
        tot_accounted_time += (layer_times[-1][1] + layer_times[-1][2])

    print()
    print("Total accounted time: %.3f ms" % tot_accounted_time)
    return layer_times, (sum(data_times) * 1000.0) / len(data_times), (sum(h2d_times) * 1000.0 / len(h2d_times))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
