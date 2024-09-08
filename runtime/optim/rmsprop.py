from torch.optim.optimizer import required

from . import optimizer

class RMSpropWithWeightStashing(optimizer.OptimizerWithWeightStashing):
    """
    RMSprop optimizer with weight stashing.
    """
    def __init__(self, modules, master_parameters, model_parameters,
                 loss_scale, num_versions, lr=required, alpha=0.99,
                 eps=1e-8, weight_decay=0, momentum=0, centered = False,
                 verbose_freq=0):
        super(RMSpropWithWeightStashing, self).__init__(
            optim_name='RMSprop',
            modules=modules, master_parameters=master_parameters,
            model_parameters=model_parameters, loss_scale=loss_scale,
            num_versions=num_versions, lr=lr, alpha=alpha, eps=eps,
            weight_decay=weight_decay, momentum=momentum, centered=centered,
            verbose_freq=verbose_freq,
        )
