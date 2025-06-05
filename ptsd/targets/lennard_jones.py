from typing import List
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from scipy.interpolate import CubicSpline
from bgflow import Energy
from bgflow.utils import distance_vectors, distances_from_vectors
from ptsd.targets.base_target import TargetDistribution
from ptsd.utils.se3_utils import remove_mean, interatomic_dist, avg_distance_to_origin


def sample_from_array(array, size):
    idx = np.random.choice(array.shape[0], size=size)
    return array[idx]


def lennard_jones_energy_torch(r, eps=1.0, rm=1.0):
    p = 0.9
    lj = eps * ((rm / r) ** 12 - 2 * (rm / r) ** 6)
    return lj


def cubic_spline(x_new, x, c):
    x, c = x.to(x_new.device), c.to(x_new.device)
    intervals = torch.bucketize(x_new, x) - 1
    intervals = torch.clamp(intervals, 0, len(x) - 2)  # Ensure valid intervals
    # Calculate the difference from the left breakpoint of the interval
    dx = x_new - x[intervals]
    # Evaluate the cubic spline at x new
    y_new = (
        c[0, intervals] * dx**3
        + c[1, intervals] * dx**2
        + c[2, intervals] * dx
        + c[3, intervals]
    )
    return y_new


class LennardJonesPotential(Energy, TargetDistribution):
    def __init__(
        self,
        dim,
        n_particles,
        eps=1.0,
        rm=1.0,
        oscillator=True,
        oscillator_scale=1.0,
        two_event_dims=True,
        energy_factor=1.0,
        range_min=0.65,
        range_max=2.0,
        interpolation=1000,
        device='cpu',
    ):
        """Energy for a Lennard-Jones cluster.

        Parameters
        ----------
        dim : int
            Number of degrees of freedom ( = space dimension x n_particles)
        n_particles : int
            Number of Lennard-Jones particles
        eps : float
            LJ well depth epsilon
        rm : float
            LJ well radius R_min
        oscillator : bool
            Whether to use a harmonic oscillator as an external force
        oscillator_scale : float
            Force constant of the harmonic oscillator energy
        two_event_dims : bool
            If True, the energy expects inputs with two event dimensions (particle_id, coordinate).
            Else, use only one event dimension.
        """
        if two_event_dims:
            Energy.__init__(self, [n_particles, dim // n_particles])
        else:
            Energy.__init__(self, dim)
        TargetDistribution.__init__(self, dim=dim, is_se3=True, n_particles=n_particles)
        self._n_particles = n_particles
        self.n_spatial_dim = dim // n_particles
        self._eps = eps
        self._rm = rm
        self.oscillator = oscillator
        self._oscillator_scale = oscillator_scale

        # this is to match the eacf energy with the eq-fm energy
        # for lj13, to match the eacf set energy_factor=0.5
        self._energy_factor = energy_factor

        self.range_min = range_min
        self.range_max = range_max

        # fit spline cubic on these ranges
        interpolate_points = torch.linspace(range_min, range_max, interpolation)

        es = lennard_jones_energy_torch(interpolate_points, self._eps, self._rm)
        coeffs = CubicSpline(np.array(interpolate_points), np.array(es)).c
        self.splines = partial(
            cubic_spline, x=interpolate_points, c=torch.tensor(coeffs).float()
        )
        if self._n_particles == 13:
            self.data = torch.tensor(
                remove_mean(
                    self.load_data("data/train_split_LJ13-1000.npy"),
                    n_particles,
                    self.n_spatial_dim,
                ),
                device=device,
            )
        elif self._n_particles == 55:
            self.data = torch.tensor(
                remove_mean(
                    self.load_data("data/train_split_LJ55-1000-part1.npy"),
                    n_particles,
                    self.n_spatial_dim,
                ),
                device=device,
            )
        else:
            raise NotImplementedError
        self.device = device

        self.all_metric_plots = {
            "inter_dist": lambda samples, label, **kwargs: plt.hist(
                self.interatomic_dist(samples).detach().cpu().view(-1),
                bins=100,
                density=True,
                histtype="step",
                linewidth=4,
                label=label,
                range=(0, 9),
                **kwargs
            ),
            "avg_dist_to_origin": lambda samples, label, **kwargs: plt.hist(
                avg_distance_to_origin(samples, self._n_particles, self.n_spatial_dim)
                .detach()
                .cpu()
                .view(-1),
                bins=100,
                density=True,
                histtype="step",
                linewidth=4,
                label=label,
                range=(0, 9),
                **kwargs
            ),
        }

    def _energy(self, x, smooth_=False):
        if len(x.shape) == len(self.event_shape):
            x = x.unsqueeze(0)
        if x.shape[0] == 0:
            return torch.zeros([0, 1]).to(x.device)
        batch_shape = x.shape[: -len(self.event_shape)]
        x = x.view(*batch_shape, self._n_particles, self.n_spatial_dim)

        dists = distances_from_vectors(
            distance_vectors(x.view(-1, self._n_particles, self.n_spatial_dim))
        )

        lj_energies = lennard_jones_energy_torch(dists, self._eps, self._rm)

        if smooth_:
            lj_energies[dists < self.range_min] = self.splines(
                dists[dists < self.range_min]
            ).squeeze(-1)
        # lj_energies = lj_energies.view(*batch_shape, -1).sum(dim=-1) * self._energy_factor
        lj_energies = (
            lj_energies.view(*batch_shape, self._n_particles, -1).sum(dim=-1)
            * self._energy_factor
        )

        if self.oscillator:
            osc_energies = 0.5 * self._remove_mean(x).pow(2).sum(dim=(-1)).view(
                *batch_shape, self._n_particles
            )
            lj_energies = lj_energies + osc_energies * self._oscillator_scale
        return lj_energies  # [:, None]

    def _remove_mean(self, x):
        x = x.view(-1, self._n_particles, self.n_spatial_dim)
        return x - torch.mean(x, dim=1, keepdim=True)

    def _energy_numpy(self, x):
        x = torch.Tensor(x)
        return self._energy(x).cpu().numpy()

    def log_prob(self, x, smooth=True):
        return -self._energy(x, smooth_=smooth).sum(-1)

    def load_data(self, path, size=None):
        if path[-3:] == "npy":
            samples = np.load(path, allow_pickle=True)
            if size is not None:
                samples = samples[0][-size:]
        else:
            samples = torch.load(path)
        return samples

    def sample(self, num_samples):
        if isinstance(num_samples, int):
            indices = torch.randint(
                0, self.data.shape[0], (num_samples,), device=self.device
            )
        elif isinstance(num_samples, list) or isinstance(num_samples, tuple):
            indices = torch.randint(
                0, self.data.shape[0], num_samples, device=self.device
            )
        else:
            raise ValueError
        return self.data[indices].to(self.device)

    def interatomic_dist(self, x):
        batch_shape = x.shape[: -len(self.event_shape)]
        x = x.view(*batch_shape, self._n_particles, self.n_spatial_dim)
        return interatomic_dist(x)

    def plot_samples(
        self,
        samples_list: List[torch.Tensor],
        labels_list: List[str],
        metric_to_plot="inter_dist",
        **kwargs
    ):
        for label, samples in zip(labels_list, samples_list):
            if samples is None:
                continue
            self.all_metric_plots[metric_to_plot](samples, label, **kwargs)
