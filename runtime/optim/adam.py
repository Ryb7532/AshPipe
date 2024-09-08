# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch.optim.optimizer import required

from . import optimizer

class AdamWithWeightStashing(optimizer.OptimizerWithWeightStashing):
    """
    Adam optimizer with weight stashing.
    """
    def __init__(self, modules, master_parameters, model_parameters,
                 loss_scale, num_versions, lr=required, betas=(0.9, 0.999),
                 weight_decay=0, amsgrad = False, verbose_freq=0):
        super(AdamWithWeightStashing, self).__init__(
            optim_name='Adam',
            modules=modules, master_parameters=master_parameters,
            model_parameters=model_parameters, loss_scale=loss_scale,
            num_versions=num_versions, lr=lr, betas=betas,
            weight_decay=weight_decay, amsgrad=amsgrad, verbose_freq=verbose_freq,
        )
