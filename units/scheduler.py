import torch
import numpy as np
from functools import partial
from torch.optim.lr_scheduler import _LRScheduler

def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]

def compute_gaussian_product_coef(sigma1, sigma2):
    """
    Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
    return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var)
    """

    # denom = np.sqrt(sigma1**2 + sigma2**2)
    # coef1 = sigma2 / denom
    # coef2 = sigma1 / denom

    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2,
                       power: float = 1., inv_power: float = 1.):
    """
    betas for schrodinger bridge
    """
    betas = torch.linspace(
        linear_start ** inv_power,
        linear_end ** inv_power,
        n_timestep,
        dtype=torch.float64
    ) ** power
    return betas.numpy()

class SBSchedule():
    def __init__(
        self,
        timesteps: int = 1000,
        beta_max: float = 0.3,
        power: float = 1.,
        inv_power: float = 1.
    ):
        betas = make_beta_schedule(
            n_timestep=timesteps,
            linear_end=beta_max/timesteps,
            power=power,
            inv_power=inv_power,
        )
        betas = np.concatenate([betas[:timesteps//2], np.flip(betas[:timesteps//2])])       
        betas = (beta_max / timesteps) / np.max(betas) * betas * 0.5                        

        self.timesteps = betas.shape[0]

        # compute analytic std
        std_fwd = np.sqrt(np.cumsum(betas))                                                                       
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))                               

        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)

        std_sb = np.sqrt(var)
        # tensorize everything
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas)
        self.std_fwd = to_torch(std_fwd)
        self.std_bwd = to_torch(std_bwd)
        self.std_sb  = to_torch(std_sb)
        self.mu_x0 = to_torch(mu_x0)
        self.mu_x1 = to_torch(mu_x1)
        # self.alphas = to_torch(alphas)

    @staticmethod
    def inflate_batch_array(array, target):
        r"""
        Inflates the batch array (array) with only a single axis
        (i.e. shape = (batch_size,), or possibly more empty axes
        (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def get_std_fwd(self, step, xdim=None):
        device = self.mu_x0.device
        step=step.to(device)
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

class NoamLR(_LRScheduler):
    """
    Adapted from https://github.com/tugstugi/pytorch-saltnet/blob/master/utils/lr_scheduler.py

    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, model_size, warmup_steps):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        step = max(1, self._step_count)
        scale = self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps**(-1.5))

        return [base_lr * scale for base_lr in self.base_lrs]