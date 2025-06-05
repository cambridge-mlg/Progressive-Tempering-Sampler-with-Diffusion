import torch
import yaml
from PIL import Image
import os
import mdtraj

from ptsd.targets.base_target import TargetDistribution
from ptsd.targets.aldp_boltzmann_dist import AldpBoltzmann
from ptsd.targets.aldp_utils import evaluate_aldp, filter_chirality
from ptsd.utils.se3_utils import remove_mean
import numpy as np


class AldpPotentialCart(AldpBoltzmann, TargetDistribution):
    def __init__(
        self,
        data_path=None,
        temperature=1000,
        energy_cut=1.0e8,
        energy_max=1.0e20,
        n_threads=4,
        transform='cartesian',
        ind_circ_dih=[],
        shift_dih=False,
        shift_dih_params={'hist_bins': 100},
        default_std={'bond': 0.005, 'angle': 0.15, 'dih': 0.2},
        env='vacuum',
        device='cpu',
        use_pt_data=False,
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
        TargetDistribution.__init__(self, dim=66, is_se3=True, n_particles=22)
        self.dim = 66
        self.device = device
        self.to(device)
        self.is_molecule = True
        print(f"PT Data: {use_pt_data}. Max samples: {max_samples_for_data}")
        if use_pt_data:
            print("Using PT data")
            internal_data = torch.load('data/aldp/train.pt').to(device)[
                :max_samples_for_data
            ]
            print(f"Loaded {internal_data.shape[0]} samples from {data_path}")
            self.data = remove_mean(
                self.coordinate_transform.forward(internal_data)[0],
                n_particles=22,
                n_dimensions=3,
            )
            traj = mdtraj.load("data/aldp/train.h5")  # just for the atom types etc
        else:
            traj = mdtraj.load("data/aldp/train.h5")
            traj.center_coordinates()
            ind = traj.top.select("backbone")
            traj.superpose(traj, 0, atom_indices=ind, ref_atom_indices=ind)
            data = traj.xyz
            self.min_energy_pos_path = data_path
            self.data = (
                torch.from_numpy(data.astype("float64"))
                .to(device)
                .reshape(-1, self.dim)
            )
            self.data = remove_mean(self.data, n_particles=22, n_dimensions=3)
        self.bonds = [
            (0, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (4, 5),
            (4, 6),
            (6, 7),
            (6, 8),
            (8, 9),
            (8, 10),
            (10, 11),
            (10, 12),
            (10, 13),
            (8, 14),
            (14, 15),
            (14, 16),
            (16, 17),
            (16, 18),
            (18, 19),
            (18, 20),
            (18, 21),
        ]
        structure_elements = [
            'C',
            'O',
            'N',
            'H',
            'H1',
            'H2',
            'H3',
            'CH3',
            'CA',
            'CB',
            'HA',
            'HB1',
            'HB2',
            'HB3',
        ]
        chemical_elements = ['H', 'C', 'N', 'O']
        table, _ = traj.topology.to_dataframe()
        structure_to_index = {
            element: idx for idx, element in enumerate(structure_elements)
        }
        chemical_to_index = {
            element: idx for idx, element in enumerate(chemical_elements)
        }
        atom_structure_elements = table["name"].values
        atom_chemical_elements = table["element"].values
        atom_structure_types = [
            structure_to_index[element] for element in atom_structure_elements
        ]
        atom_chemical_types = [
            chemical_to_index[element] for element in atom_chemical_elements
        ]
        self.atom_structure_types = torch.tensor(atom_structure_types).to(device)
        self.atom_chemical_types = torch.tensor(atom_chemical_types).to(device)

    def get_min_energy_position(self):
        x_init = torch.load(self.min_energy_pos_path, map_location=self.device)
        return x_init

    def log_prob(self, x):
        if x.ndim == 2:
            log_prob = super().log_prob(x)
        else:
            x_shape = x.shape
            x_reshaped = x.reshape(-1, self.dim)
            log_prob = super().log_prob(x).log_prob(x_reshaped)
            log_prob = log_prob.reshape(x_shape[:-1])
        return log_prob

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
        return self.data[indices].to(self.device).float()

    def eval(
        self,
        samples: torch.Tensor,
        true_samples: torch.Tensor,
        iter: int,
        metric_dir,
        plot_dir,
        batch_size: int,
    ):
        samples = self.coordinate_transform.inverse(samples.clone().detach())[0]
        true_samples = self.coordinate_transform.inverse(true_samples.clone().detach())[
            0
        ]
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

    def filter_chirality_cartesian(self, x):
        assert x.shape[-1] == self.dim, "Data should be in Cartesian coordinate"
        internal_x = self.coordinate_transform.inverse(x.clone().detach())[0]
        ind_L = filter_chirality(internal_x)
        filtered_x = self.coordinate_transform.forward(internal_x[ind_L])[0]
        return filtered_x

    def reflect_d_to_l_cartesian(self, x, reflect_ind: int = 0):
        assert x.shape[-1] == self.dim, "Data should be in Cartesian coordinate"
        internal_x = self.coordinate_transform.inverse(x.clone().detach())[0]
        ind_L = filter_chirality(internal_x)
        L_x = x[ind_L]
        D_x = x[~ind_L].view(-1, self.n_particles, self.n_dimensions)
        D_x[..., reflect_ind] *= -1.0  # reflect one axis
        reflected_x = torch.cat([L_x, D_x.reshape(-1, self.dim)], dim=0)
        return reflected_x


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
    target = AldpPotentialCart(
        data_path=config['data']['transform'],
        temperature=config['system']['temperature'],
        energy_cut=config['system']['energy_cut'],
        energy_max=config['system']['energy_max'],
        n_threads=config['system']['n_threads'],
        ind_circ_dih=ind_circ_dih,
        shift_dih=shift_dih,
        env=env,
        device=device,
    )
    return target
