import torch
import numpy as np
from functools import partial
from typing import Optional, Callable
import copy
from torch.distributions import Normal, Gumbel
from typing import Union
import pdb


class MCMCSampler(object):
    def __init__(
        self,
        x: torch.Tensor,
        energy_func: Callable,
        step_size: Union[float, torch.Tensor],
        mh: bool = True,
        device: str = 'cpu',
        point_estimator: bool = False,
    ):

        self.x = x
        self.step_size = step_size
        self.energy_func = energy_func
        self.mh = mh
        self.device = device
        self.point_estimator = point_estimator

    def sample(self) -> tuple:
        pass


class LangevinDynamics(MCMCSampler):

    def __init__(
        self,
        x: torch.Tensor,
        energy_func: callable,
        step_size: Union[float, torch.Tensor],
        mh: bool = True,
        device: str = 'cpu',
        point_estimator: bool = False,
    ):
        """
        Standard Langevin Dynamics Sampler
        """
        super(LangevinDynamics, self).__init__(
            x=x,
            energy_func=energy_func,
            step_size=step_size,
            mh=mh,
            device=device,
            point_estimator=point_estimator,
        )

        if self.mh:
            x_c = self.x.detach()
            x_c.requires_grad = True
            f_xc = self.energy_func(x_c)
            grad_xc = torch.autograd.grad(f_xc.sum(), x_c, create_graph=False)[0]

            self.f_x = f_xc.detach()
            self.grad_x = grad_xc.detach()

    def sample(self) -> tuple:
        if self.point_estimator == True:
            x_c = self.x.detach()
            x_c.requires_grad = True
            f_xc = self.energy_func(x_c)
            grad_xc = torch.autograd.grad(f_xc.sum(), x_c, create_graph=False)[0]
            x_p = x_c - self.step_size * grad_xc
            self.x = x_p.detach()
            return copy.deepcopy(x_p.detach()), None

        if self.mh == False:
            x_c = self.x.detach()
            x_c.requires_grad = True
            f_xc = self.energy_func(x_c)
            grad_xc = torch.autograd.grad(f_xc.sum(), x_c, create_graph=False)[0]

            x_p = (
                x_c
                - self.step_size * grad_xc
                + torch.sqrt(torch.tensor(2.0 * self.step_size, device=self.device))
                * torch.randn_like(x_c, device=self.device)
            )

            self.x = x_p.detach()
            return copy.deepcopy(x_p.detach()), f_xc.detach()

        else:
            x_c = self.x.detach()
            f_xc = self.f_x.detach()
            grad_xc = self.grad_x.detach()

            x_p = (
                x_c
                - self.step_size * grad_xc
                + torch.sqrt(torch.tensor(2.0 * self.step_size, device=self.device))
                * torch.randn_like(self.x, device=self.device)
            )
            x_p = x_p.detach()
            x_p.requires_grad = True
            f_xp = self.energy_func(x_p)
            grad_xp = torch.autograd.grad(f_xp.sum(), x_p, create_graph=False)[0]
            if isinstance(self.step_size, float):
                log_joint_prob_2 = -f_xc - torch.norm(
                    x_p - x_c + self.step_size * grad_xc, dim=-1
                ) ** 2 / (4 * self.step_size)
                log_joint_prob_1 = -f_xp - torch.norm(
                    x_c - x_p + self.step_size * grad_xp, dim=-1
                ) ** 2 / (4 * self.step_size)
            else:
                log_joint_prob_2 = -f_xc - torch.norm(
                    x_p - x_c + self.step_size * grad_xc, dim=-1
                ) ** 2 / (4 * self.step_size.squeeze(-1))
                log_joint_prob_1 = -f_xp - torch.norm(
                    x_c - x_p + self.step_size * grad_xp, dim=-1
                ) ** 2 / (4 * self.step_size.squeeze(-1))

            log_accept_rate = log_joint_prob_1 - log_joint_prob_2
            is_accept = torch.rand_like(log_accept_rate).log() <= log_accept_rate
            is_accept = is_accept.unsqueeze(-1)

            self.x = torch.where(is_accept, x_p.detach(), self.x)
            self.f_x = torch.where(is_accept.squeeze(-1), f_xp.detach(), self.f_x)
            self.grad_x = torch.where(is_accept, grad_xp.detach(), self.grad_x)

            acc_rate = torch.minimum(
                torch.ones_like(log_accept_rate), log_accept_rate.exp()
            )

            return copy.deepcopy(self.x.detach()), acc_rate.detach()


class ParallelTempering(LangevinDynamics):
    def __init__(
        self,
        x: torch.Tensor,
        energy_func: Callable,
        step_size: Union[float, torch.Tensor],
        swap_interval: int,
        temperatures: torch.Tensor,
        mh: bool = True,
        device: str = 'cpu',
        point_estimator: bool = False,
    ):
        """
        x: torch.Tensor([num_temperatures, N, dim])
        temperatures = torch.Tensor([num_temperatures])
        """
        super(ParallelTempering, self).__init__(
            x=x.reshape(-1, x.shape[-1]),
            energy_func=lambda samples: energy_func(samples)
            / temperatures.repeat_interleave(x.shape[1]),
            step_size=step_size,
            mh=mh,
            device=device,
            point_estimator=point_estimator,
        )
        self.base_energy = energy_func
        assert (
            len(x.shape) == 3
            and len(temperatures.shape) == 1
            and x.shape[0] == len(temperatures)
        )
        self.temperatures = temperatures
        self.num_temperatures = x.shape[0]
        self.swap_rate = 0.0
        self.swap_rates = []
        self.swap_interval = swap_interval
        self.counter = 0

    def sample_per_temp(self):
        new_samples, acc = super(ParallelTempering, self).sample()
        return new_samples, acc

    def get_chain(self, chain_index):
        chain_samples = (
            self.x.view(self.num_temperatures, -1, self.x.shape[-1])[chain_index]
            .clone()
            .detach()
        )
        chain_energy = (
            self.f_x.view(self.num_temperatures, -1)[chain_index].clone().detach()
        )
        chain_scores = (
            self.grad_x.view(self.num_temperatures, -1, self.x.shape[-1])[chain_index]
            .clone()
            .detach()
        )
        return chain_samples, chain_energy, chain_scores

    def add_chain(self, chain_index, samples, energies, scores):
        self.x.view(self.num_temperatures, -1, self.x.shape[-1])[
            chain_index
        ] = samples.clone().detach()
        self.f_x.view(self.num_temperatures, -1)[
            chain_index
        ] = energies.clone().detach()
        self.grad_x.view(self.num_temperatures, -1, self.x.shape[-1])[
            chain_index
        ] = scores.clone().detach()

    def attempt_swap(self, chain_a_index, chain_b_index):
        temp_a, temp_b = (
            self.temperatures[chain_a_index],
            self.temperatures[chain_b_index],
        )
        chain_a, chain_a_energy, chain_a_score = self.get_chain(chain_a_index)
        chain_b, chain_b_energy, chain_b_score = self.get_chain(chain_b_index)
        log_prob_a = -chain_a_energy * temp_a  # this is untempered log prob
        log_prob_b = -chain_b_energy * temp_b  # this is untempered log prob

        log_acceptance_ratio = (1.0 / temp_a - 1.0 / temp_b) * (log_prob_b - log_prob_a)
        is_accept = (
            torch.rand_like(log_acceptance_ratio, device=self.device).log()
            < log_acceptance_ratio
        )
        is_accept = is_accept.unsqueeze(-1)

        new_chain_a = torch.where(
            is_accept, chain_b.detach().clone(), chain_a.detach().clone()
        )
        new_chain_b = torch.where(
            is_accept, chain_a.detach().clone(), chain_b.detach().clone()
        )

        new_chain_a_score = torch.where(
            is_accept,
            chain_b_score.detach().clone() * temp_b / temp_a,
            chain_a_score.detach().clone(),
        )
        new_chain_b_score = torch.where(
            is_accept,
            chain_a_score.detach().clone() * temp_a / temp_b,
            chain_b_score.detach().clone(),
        )

        new_chain_a_energy = torch.where(
            is_accept.squeeze(-1),
            chain_b_energy.detach().clone() * temp_b / temp_a,
            chain_a_energy.detach().clone(),
        )
        new_chain_b_energy = torch.where(
            is_accept.squeeze(-1),
            chain_a_energy.detach().clone() * temp_a / temp_b,
            chain_b_energy.detach().clone(),
        )

        self.add_chain(
            chain_a_index, new_chain_a, new_chain_a_energy, new_chain_a_score
        )
        self.add_chain(
            chain_b_index, new_chain_b, new_chain_b_energy, new_chain_b_score
        )

        return is_accept.float().mean().item()

    def swap_samples(self):
        swap_rates = []
        for i in range(self.num_temperatures - 1, 0, -1):
            swap_rate = self.attempt_swap(i, i - 1)
            swap_rates.append(swap_rate)
        self.swap_rate = np.mean(swap_rates)
        self.swap_rates = swap_rates

    def sample(self):
        _, acc = self.sample_per_temp()
        self.counter += 1
        if self.counter % self.swap_interval == 0:
            self.swap_samples()
        return (
            self.x.clone()
            .detach()
            .reshape(self.num_temperatures, -1, self.x.shape[-1]),
            acc,
        )
