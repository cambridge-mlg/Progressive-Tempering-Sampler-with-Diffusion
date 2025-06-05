import torch
from tqdm import tqdm
from typing import Callable, Optional

from ptsd.sampler.sampler import MCMCSampler
from ptsd.sampler.dyn_mcmc_warp import DynSamplerWrapper


def sample_mcmc_chain(
    sampler: MCMCSampler,
    num_chains: int,
    num_steps: int,
    num_interval: int,
    init_pos: torch.Tensor,
    eval_func: Optional[Callable] = None,
):
    """
    Simulate multiple MCMC chains. Draw samples from each chain for every fixed intervel.
        init_pos: torch.Tensor([num_chains, dim]) or torch.Tensor([1, dim]) or torch.tensor([dim])

    Return: torch.Tensor([num_chains, K, dim])
    """
    if len(init_pos.shape) == 1 or (
        len(init_pos.shape) == 2 and init_pos.shape[0] == 1
    ):
        init_pos = init_pos.repeat(num_chains, 1)
    sampler.x = init_pos.clone().detach()

    samples = init_pos.clone().detach().unsqueeze(1)
    progress_bar = tqdm(range(num_steps), desc="MCMC simulation")
    for i in progress_bar:
        x, acc, *_ = sampler.sample()
        if i % num_interval == 0:
            samples = torch.concat((samples, x.clone().detach().unsqueeze(1)), dim=1)
            if eval_func:
                eval_func(samples.reshape(-1, init_pos.shape[1]), i)
        progress_bar.set_postfix({"acc rate": f"{acc:.2f}"})
    return samples
