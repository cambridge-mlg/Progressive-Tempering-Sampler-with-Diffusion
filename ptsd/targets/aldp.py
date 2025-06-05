import torch
import numpy as np
import yaml
from PIL import Image
from typing import List
import os

from ptsd.targets.base_target import TargetDistribution
from ptsd.targets.aldp_boltzmann_dist import AldpBoltzmann
from ptsd.targets.aldp_utils import evaluate_aldp


class AldpPotential(AldpBoltzmann, TargetDistribution):
    def __init__(
        self,
        data_path=None,
        temperature=1000,
        energy_cut=1.0e8,
        energy_max=1.0e20,
        n_threads=4,
        transform='internal',
        ind_circ_dih=[],
        shift_dih=False,
        shift_dih_params={'hist_bins': 100},
        default_std={'bond': 0.005, 'angle': 0.15, 'dih': 0.2},
        env='vacuum',
        device='cpu',
        max_samples_for_data=None,
    ):
        AldpBoltzmann.__init__(
            self,
            data_path=data_path,
            temperature=temperature,
            energy_cut=energy_cut,
            energy_max=energy_max,
            n_threads=n_threads,
            transform=transform,
            ind_circ_dih=ind_circ_dih,
            shift_dih=shift_dih,
            shift_dih_params=shift_dih_params,
            default_std=default_std,
            env=env,
        )
        TargetDistribution.__init__(self, dim=60, is_se3=False, n_particles=1)
        self.dim = 60
        self.device = device
        self.to(device)
        self.is_molecule = False
        ind_circ_dih = [0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 16]

        ncarts = self.coordinate_transform.transform.len_cart_inds
        dih_ind_ = (
            self.coordinate_transform.transform.ic_transform.dih_indices.cpu().numpy()
        )
        ind = np.arange(60)
        ind = np.concatenate(
            [ind[: 3 * ncarts - 6], -np.ones(6, dtype=int), ind[3 * ncarts - 6 :]]
        )

        dih_ind = ind[dih_ind_]
        ind_circ = dih_ind[ind_circ_dih]

        self.dih_ind = dih_ind
        self.ind_circ = ind_circ
        self.data = torch.load('data/aldp/train.pt').to(device)[:max_samples_for_data]

    def log_prob(self, x):
        if x.ndim == 2:
            log_prob = (
                super().log_prob(x)
                - (x[:, self.ind_circ].abs() > np.pi).any(-1).float() * 1e8
            )
        else:
            x_shape = x.shape
            x_reshaped = x.reshape(-1, 60)
            log_prob = (
                super().log_prob(x).log_prob(x_reshaped)
                - (x[:, self.ind_circ].abs() > np.pi).any(-1).float() * 1e8
            )
            log_prob = log_prob.reshape(x_shape[:-1])
        return torch.where(log_prob.isnan(), 1e8, log_prob)

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
        return self.data[indices].to(self.device).float()  # shape (num_samples, 60)

    def eval(
        self,
        samples: torch.Tensor,
        true_samples: torch.Tensor,
        iter: int,
        metric_dir,
        plot_dir,
        batch_size: int,
    ):
        samples = samples.clone().detach()
        evaluate_aldp(
            samples,
            true_samples,
            self.log_prob,
            self.coordinate_transform,
            iter,
            metric_dir,
            plot_dir,
            batch_size,
        )

    def plot_samples(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        iter: int,
        metric_dir,
        plot_dir,
        batch_size: int,
    ):
        self.eval(source, target, iter, metric_dir, plot_dir, batch_size)
        marginal_angle = Image.open(
            os.path.join(plot_dir, 'marginals_%s_%07i.png' % ("angle", iter + 1))
        )
        marginal_bond = Image.open(
            os.path.join(plot_dir, 'marginals_%s_%07i.png' % ("bond", iter + 1))
        )
        marginal_dih = Image.open(
            os.path.join(plot_dir, 'marginals_%s_%07i.png' % ("dih", iter + 1))
        )
        phi_psi = Image.open(
            os.path.join(plot_dir, '%s_%07i.png' % ("phi_psi", iter + 1))
        )
        ramachandran = Image.open(
            os.path.join(plot_dir, '%s_%07i.png' % ("ramachandran", iter + 1))
        )
        images = {
            'marginal_angle': marginal_angle,
            'marginal_bond': marginal_bond,
            'marginal_dih': marginal_dih,
            'phi_psi': phi_psi,
            'ramachandran': ramachandran,
        }
        return images


def get_aldp_potential(config_path, device):

    with open(config_path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    # Target distribution
    transform_mode = (
        'mixed'
        if not 'transform' in config['system']
        else config['system']['transform']
    )
    shift_dih = (
        False if not 'shift_dih' in config['system'] else config['system']['shift_dih']
    )
    env = 'vacuum' if not 'env' in config['system'] else config['system']['env']
    ind_circ_dih = [0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 16]
    target = AldpPotential(
        data_path=config['data']['transform'],
        temperature=config['system']['temperature'],
        energy_cut=config['system']['energy_cut'],
        energy_max=config['system']['energy_max'],
        n_threads=config['system']['n_threads'],
        transform=transform_mode,
        ind_circ_dih=ind_circ_dih,
        shift_dih=shift_dih,
        env=env,
        device=device,
    )
    return target
