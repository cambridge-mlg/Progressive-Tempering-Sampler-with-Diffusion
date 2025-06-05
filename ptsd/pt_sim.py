from omegaconf import DictConfig
import wandb
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from ptsd.sampler.sampler import ParallelTempering
from ptsd.sampler.dyn_mcmc_warp import DynSamplerWrapper
from ptsd.targets.aldp import AldpPotential
from ptsd.targets.aldp_cartesian import AldpPotentialCart
from ptsd.targets.base_target import TargetDistribution


"""
Simulate samples with shape (num_chains, num_steps, dim) using Parallel Tempering
"""

torch.manual_seed(0)
# torch.set_default_dtype(torch.float64)


def parallel_tempering_simulation(
    target: TargetDistribution, x_init: torch.Tensor, config: DictConfig
):
    all_temps = torch.tensor(config["all_temps"]).to(config["device"])
    dim = target.dim
    pt = ParallelTempering(
        x=x_init,
        energy_func=lambda x: -target.log_prob(x),
        step_size=torch.tensor(
            [config["step_size"]] * (config["total_n_temp"] * config["num_chains"]),
            device=config["device"],
        ).unsqueeze(-1),
        swap_interval=config["swap_interval"],
        temperatures=all_temps,
        mh=True,
        device=config["device"],
    )
    pt = DynSamplerWrapper(
        pt,
        per_temp=True,
        total_n_temp=config["total_n_temp"],
        target_acceptance_rate=0.6,
        alpha=0.25,
    )
    progress_bar = tqdm(
        range(config["num_steps"]),
        desc=f"Parallel Tempering for {config['name']} (dim={dim})",
    )
    swap_rates = []
    traj = []
    for i in progress_bar:
        new_samples, acc, *_ = pt.sample()
        traj.append(new_samples.clone().detach().cpu().float())
        if pt.sampler.swap_rates:
            swap_rates.append(pt.sampler.swap_rates)
            for j in range(len(all_temps) - 1):
                wandb.log(
                    {
                        f"swap_rates/{all_temps[j].item():.2f}~{all_temps[j + 1].item():.2f}": pt.sampler.swap_rates[
                            j
                        ]
                    },
                    step=i,
                )
        if (i + 1) % config["check_interval"] == 0:
            os.makedirs(config["save_fold"], exist_ok=True)
            torch.save(
                torch.stack(traj, dim=2)[0].detach().cpu().float(),
                f"{config['save_fold']}/pt_{config['name']}.pt",
            )
            if isinstance(target, AldpPotential) or isinstance(
                target, AldpPotentialCart
            ):
                sub_plot_path = f"{config['plot_path']}/{i + 1}"
                os.makedirs(sub_plot_path, exist_ok=True)
                images_dict = target.plot_samples(
                    torch.stack(traj[-config["check_interval"] :], dim=2)[0]
                    .reshape(-1, dim)
                    .to(config["device"]),
                    target.sample(1000000),
                    i,
                    sub_plot_path,
                    sub_plot_path,
                    1000000,
                )
                for key, value in images_dict.items():
                    wandb.log({f"{key}": wandb.Image(value)}, step=i)
            else:
                plt.figure(figsize=(6, 6))
                target.plot_samples(
                    samples_list=[
                        torch.stack(traj[-config["check_interval"] :], dim=2)[
                            0
                        ].reshape(-1, dim),
                        target.sample([config["check_interval"]]),
                    ],
                    labels_list=["pt", "gt"],
                    alpha=0.5,
                )
                plt.legend()
                wandb.log({"samples": wandb.Image(plt)}, step=i)
                plt.close()
        for j in range(len(all_temps)):
            wandb.log({f"acc_rates/{all_temps[j].item():.2f}": acc[j].item()}, step=i)
        progress_bar.set_postfix_str(f"acc rate: {acc.mean().item()}")
    traj = torch.stack(traj, dim=2)[0]
    os.makedirs(config["save_fold"], exist_ok=True)
    torch.save(
        traj.detach().cpu().float(),
        f"{config['save_fold']}/pt_{config['name']}_{dim}.pt",
    )
