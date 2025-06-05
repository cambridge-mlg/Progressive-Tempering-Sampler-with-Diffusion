import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, List
from fab.target_distributions import many_well
from fab.utils.plotting import plot_contours, plot_marginal_pair

from ptsd.targets.base_target import TargetDistribution


def get_target_log_prob_marginal_pair(log_prob, i: int, j: int, total_dim: int):
    def log_prob_marginal_pair(x_2d):
        x = torch.zeros((x_2d.shape[0], total_dim))
        x[:, i] = x_2d[:, 0]
        x[:, j] = x_2d[:, 1]
        return log_prob(x)

    return log_prob_marginal_pair


class ManyWellPotential(TargetDistribution):
    def __init__(
        self,
        dim,
        device="cpu",
    ):
        use_gpu = device != "cpu"
        self.many_well_energy = many_well.ManyWellEnergy(
            dim=dim, use_gpu=use_gpu, normalised=False, a=-0.5, b=-6.0, c=1.0
        )
        TargetDistribution.__init__(self, dim=dim, is_se3=False, n_particles=1)
        self.device = device
        self.all_metric_plots = {
            "marginal_pair": lambda samples, label, **kwargs: plt.scatter(
                samples[:, 0].detach().cpu(),
                samples[:, 2].detach().cpu(),
                label=label,
                **kwargs
            )
        }

    def log_prob(self, x: torch.Tensor):
        return self.many_well_energy.log_prob(x)

    def sample(self, num_samples: Optional[int]):
        if isinstance(num_samples, int):
            return self.many_well_energy.sample((num_samples,))
        elif isinstance(num_samples, list) or isinstance(num_samples, tuple):
            return self.many_well_energy.sample(num_samples)
        else:
            raise "Wrong values"

    def plot_samples(
        self,
        samples_list: List[torch.Tensor],
        labels_list: List[str],
        metric_to_plot="marginal_pair",
        **kwargs
    ):
        for label, samples in zip(labels_list, samples_list):
            if samples is None:
                continue
            self.all_metric_plots[metric_to_plot](samples, label, **kwargs)
