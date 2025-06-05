import torch
from tqdm import tqdm
import pdb

from ptsd.sampler.sampler import ParallelTempering
from ptsd.sampler.dyn_mcmc_warp import DynSamplerWrapper


def bilevel_bootstrap_sampling(
    log_prob_func,
    all_temps,
    init_samples,
    init_buffer=None,
    init_mcmc_steps=1000,
    mcmc_steps=50,
    mcmc_step_size=1e-4,
    replica_exchange_intervel=10,
):
    """
    samples: (2, N, dim)
    """
    num_chains = init_samples.shape[1]
    device = init_samples.device
    temps = all_temps[-2:]
    new_step_size = torch.tensor([mcmc_step_size, mcmc_step_size], device=device)
    print(f"temp low: {temps[0]}, temp high: {temps[1]}")
    all_temp_samples = []
    if init_buffer is None:
        pt = ParallelTempering(
            init_samples,
            energy_func=lambda x: -log_prob_func(x) * 1.0,
            step_size=new_step_size.repeat_interleave(num_chains).unsqueeze(-1),
            swap_interval=replica_exchange_intervel,
            temperatures=temps,
            mh=True,
            device=device,
            point_estimator=False,
        )
        pt = DynSamplerWrapper(
            pt, per_temp=True, total_n_temp=2, target_acceptance_rate=0.6, alpha=0.25
        )
        progress_bar = tqdm(range(init_mcmc_steps))
        CHAIN_SAMPLES = []
        for i in progress_bar:
            new_samples, acc, new_step_size = pt.sample()
            CHAIN_SAMPLES.append(new_samples.clone().detach())
            progress_bar.set_postfix({"acc": acc, "replica_rate": pt.sampler.swap_rate})
    else:
        new_samples = init_buffer.clone().detach()

    new_step_size[1] = new_step_size[0]

    progress_bar = tqdm(range(len(all_temps) - 1, 1, -1), desc="Bilevel Sampling")
    for temp_idx in progress_bar:
        low_temp_samples = new_samples[0].clone().detach()
        temps = all_temps[temp_idx - 2 : temp_idx]
        print(f"temp low: {temps[0]}, temp high: {temps[1]}")
        pt = ParallelTempering(
            x=torch.stack(
                [low_temp_samples.clone().detach(), low_temp_samples.clone().detach()],
                dim=0,
            ),
            energy_func=lambda x: -log_prob_func(x) * 1.0,
            step_size=new_step_size.repeat_interleave(num_chains).unsqueeze(-1),
            swap_interval=replica_exchange_intervel,
            temperatures=temps,
            mh=True,
            device=device,
            point_estimator=False,
        )
        pt = DynSamplerWrapper(
            pt, per_temp=True, total_n_temp=2, target_acceptance_rate=0.6, alpha=0.25
        )
        CHAIN_SAMPLES = []
        progress_bar_lg = tqdm(range(mcmc_steps), desc="Langevin MCMC fine-tuning")
        for _ in progress_bar_lg:
            new_samples, acc, new_step_size = pt.sample()
            CHAIN_SAMPLES.append(new_samples.clone().detach())
            progress_bar_lg.set_postfix(
                {"acc:": acc.mean().item(), "replica_acc:": pt.sampler.swap_rate}
            )
        all_temp_samples.append(torch.stack(CHAIN_SAMPLES, dim=1).clone().detach())
        new_step_size[1] = new_step_size[
            0
        ]  # init the step size at lower temp as the one at higher temp
        progress_bar.set_postfix({"temp low": temps[0], "temp high": temps[1]})
    all_temp_samples = torch.stack(all_temp_samples, dim=0).clone().detach()
    return new_samples[0].clone().detach(), all_temp_samples
