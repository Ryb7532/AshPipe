# Copyright (c) 2024 Ryb7532.
# Licensed under the MIT license.


import torch
import os


class Profiling(object):
    def __init__(self, model, module_whitelist):
        if isinstance(model, torch.nn.Module) is False:
            raise Exception("Not a valid model, please provide a 'nn.Module' instance.")

        self.model = model
        self.module_whitelist = module_whitelist
        self.record = {'forward':[], 'backward': []}
        self.profiling_on = True
        self.forward_original_methods = {}
        self.hook_handles = []
        self.hook_done = False
        self.unhook_done = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __str__(self):
        tot_time = 0.0

        ret = ""
        ret += "\n===============================================================\n"
        ret += "Forward Times\n"
        ret += "===============================================================\n"
        for i in range(len(self.record['forward'])):
            record_item = self.record['forward'][i]
            ret += "layer{:3d}:\t{:.6f} ms\t({})\n".format(
                i + 1, record_item[1].elapsed_time(record_item[3]), record_item[0])
            tot_time += (record_item[1].elapsed_time(record_item[3]))

        ret += "\n===============================================================\n"
        ret += "Backward Times\n"
        ret += "===============================================================\n"
        for i in range(len(self.record['forward'])):
            try:
                record_item = self.record['backward'][i]
                ret += "layer{:3d}:\t{:.6f} ms\t({})\n".format(
                    i + 1, record_item[1].elapsed_time(record_item[3]), record_item[0])
                tot_time += (record_item[1].elapsed_time(record_item[3]))
            except Exception as e:
                # Oops, this layer doesn't have metadata as needed.
                pass

        ret += ("\nTotal accounted time in forward and backward pass: %.6f ms" % tot_time)
        return ret

    def processed_times(self):
        torch.cuda.synchronize()
        processed_times = []
        forward_i = 0
        backward_i = 0
        last_forward_i = 0
        max_forward_i = len(self.record['forward'])
        used_forward = [False for _ in range(max_forward_i)]
        while forward_i < max_forward_i and backward_i < len(self.record['backward']):
            forward_record_item = self.record['forward'][forward_i]
            backward_record_item = self.record['backward'][backward_i]
            if backward_record_item[0] is None:
                backward_i += 1
            if forward_record_item[0] != backward_record_item[0]:
                forward_i += 1
                continue
            used_forward[forward_i] = True
            if forward_i != last_forward_i:
                forward_i = last_forward_i
            else:
                while (forward_i < max_forward_i and used_forward[forward_i]):
                    forward_i += 1
                last_forward_i = forward_i
            backward_i += 1
            forward_time = forward_record_item[1].elapsed_time(forward_record_item[3])
            backward_time = backward_record_item[1].elapsed_time(backward_record_item[3])
            processed_times.append((forward_record_item[0],
                                    forward_time * 1000, forward_record_item[2],
                                    backward_time * 1000, backward_record_item[2]))
        return processed_times

    def start(self):
        if self.hook_done is False:
            self.hook_done = True
            self.hook_modules(self.model)
        self.profiling_on = True
        return self

    def stop(self):
        self.profiling_on = False
        if self.unhook_done is False:
            self.unhook_done = True
            for handle in self.hook_handles:
                handle.remove()
            self.unhook_modules(self.model)
        self.record['backward'].reverse()
        return self

    def hook_modules(self, module):
        this_profiler = self
        sub_modules = module.__dict__['_modules']

        for _, sub_module in sub_modules.items():

            # nn.Module is the only thing we care about.
            if sub_module is None or isinstance(sub_module, torch.nn.Module) is False:
                break

            sub_module_name = sub_module.__class__.__name__
            sub_sub_modules = sub_module.__dict__['_modules']
            if "inplace" in sub_module.__dict__:
                sub_module.__dict__["inplace"] = False
            if len(sub_sub_modules) > 0 and sub_module_name not in self.module_whitelist:
                #
                # Recursively visit this module's descendants.
                #
                self.hook_modules(sub_module)
            else:
                # Wrapper function to "forward", with timer to record how long
                # forward pass takes.
                def forward_wrapper(self, *input, **kwargs):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    pid = os.getpid()
                    start.record()
                    result = this_profiler.forward_original_methods[self](*input, **kwargs)
                    end.record()

                    if (this_profiler.profiling_on):
                        global record
                        this_profiler.record['forward'].append((self, start, pid, end))

                    return result

                # Replace "forward" with "forward_wrapper".
                if sub_module not in this_profiler.forward_original_methods:
                    this_profiler.forward_original_methods.update({sub_module:
                                                                   sub_module.forward})
                    sub_module.forward = forward_wrapper.__get__(sub_module, sub_module.__class__)

                # Start timer for backward pass in pre_hook; then stop timer
                # for backward pass in post_hook.
                def backward_pre_hook(module, grad_output):
                    if (this_profiler.profiling_on):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        this_profiler.record['backward'].append((module, start, os.getpid(), end))
                    return

                def backward_post_hook(module, grad_input, grad_output):
                    idx = -1
                    if not this_profiler.profiling_on:
                        return
                    while module != this_profiler.record['backward'][idx][0]:
                        idx -= 1
                        if (-idx) == len(this_profiler.record['backward']):
                            return
                    this_profiler.record['backward'][idx][3].record()
                    return

                self.hook_handles.append(sub_module.register_full_backward_pre_hook(backward_pre_hook))
                self.hook_handles.append(sub_module.register_full_backward_hook(backward_post_hook))

    def unhook_modules(self, module):
        sub_modules = module.__dict__['_modules']

        for _, sub_module in sub_modules.items():
            # nn.Module is the only thing we care about.
            if sub_module is None or isinstance(sub_module, torch.nn.Module) is False:
                break

            sub_module_name = sub_module.__class__.__name__
            sub_sub_modules = sub_module.__dict__['_modules']
            if len(sub_sub_modules) > 0 and sub_module_name not in self.module_whitelist:
                #
                # Recursively visit this module's descendants.
                #
                self.unhook_modules(sub_module)
            else:
                sub_module.forward = self.forward_original_methods[sub_module]
