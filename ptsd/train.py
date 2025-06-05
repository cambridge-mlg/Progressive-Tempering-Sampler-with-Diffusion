import torch
import numpy as np
from typing import Optional, List
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import wandb

from ptsd.targets.base_target import TargetDistribution
from ptsd.targets.aldp import AldpPotential
from ptsd.targets.aldp_cartesian import AldpPotentialCart
from ptsd.sampler.sampler import ParallelTempering
from ptsd.sampler.dyn_mcmc_warp import DynSamplerWrapper
from ptsd.utils.diffusion_utils import (
    Eular_solve,
    importance_sample_with_reweighting,
    Euler_solve_with_log_pdf,
    Euler_solve_wrapper,
)
from ptsd.utils.metrics import Metric
from ptsd.utils.se3_utils import remove_mean
from ptsd.utils.data_parallel_utils import unwrap_data_parallel

import pdb


def get_attr(model, attr_name):
    """Safely get attribute from model, whether wrapped in DataParallel or not."""
    if hasattr(model, 'module'):
        return getattr(model.module, attr_name)
    return getattr(model, attr_name)


def set_attr(model, attr_name, value):
    """Safely set attribute on model, whether wrapped in DataParallel or not."""
    if hasattr(model, 'module'):
        setattr(model.module, attr_name, value)
    else:
        setattr(model, attr_name, value)


def DSM_loss(
    samples: torch.Tensor, models: torch.nn.Module, noise: torch.Tensor, t: torch.Tensor
):
    sigma_data = get_attr(models, 'data_sigma')
    w = (t**2 + sigma_data**2) / (t**2 * sigma_data**2)
    x_hat = models(t.log(), samples + noise * t[:, None])
    dsm_loss = ((samples - x_hat) ** 2 * w[:, None]).mean()
    return dsm_loss


def resize_sample_buffer(samples, target_size, dim=1):
    """
    Resize a tensor of samples to contain exactly target_size samples along the specified dimension.

    Args:
        samples: Tensor of samples
        target_size: The desired number of samples
        dim: The dimension along which to sample (default=1)

    Returns:
        Tensor with exactly target_size samples along the specified dimension
    """
    current_size = samples.shape[dim]

    if current_size > target_size:
        # Case 1: Too many samples - randomly subsample without replacement
        indices = torch.randperm(current_size, device=samples.device)[:target_size]
        print("Too many samples, randomly subsampling")
        return torch.index_select(samples, dim, indices)
    elif current_size < target_size:
        # Case 2: Too few samples - sample with replacement to fill the buffer
        print("Not enough samples, sampling with replacement")
        indices = torch.randint(
            0, current_size, (target_size - current_size,), device=samples.device
        )
        additional_samples = torch.index_select(samples, dim, indices)
        return torch.cat([samples, additional_samples], dim=dim)
    else:
        # Case 3: Exactly right number of samples - no adjustment needed
        print(
            "Exactly right number of samples -> no subsampling or sampling with replacement needed"
        )
        return samples


def evaluate(
    target: TargetDistribution,
    low_temp_samples: torch.Tensor,
    high_temp_samples: torch.Tensor,
    extrapolate_samples: torch.Tensor,
    mcmc_refined_extrapolate_samples: torch.Tensor,
    low_temp_buffer: torch.Tensor,
    high_temp_buffer: torch.Tensor,
    gt_samples: torch.Tensor,
    low_temp,
    high_temp,
    next_temp,
    plot_path,
):

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    target.plot_samples(
        samples_list=[high_temp_buffer, high_temp_samples],
        labels_list=["buffer", "model"],
        alpha=0.7,
    )
    plt.legend()
    plt.title(f"temp={high_temp}")
    plt.subplot(1, 4, 2)
    target.plot_samples(
        samples_list=[low_temp_buffer, low_temp_samples],
        labels_list=["buffer", "model"],
        alpha=0.7,
    )
    plt.legend()
    plt.title(f"temp={low_temp}")
    plt.subplot(1, 4, 3)
    target.plot_samples(
        samples_list=[extrapolate_samples, gt_samples if next_temp == 1.0 else None],
        labels_list=["extrapolation", "gt"],
        alpha=0.7,
    )
    plt.legend()
    plt.title(f"temp={next_temp}")
    plt.subplot(1, 4, 4)
    target.plot_samples(
        samples_list=[
            mcmc_refined_extrapolate_samples,
            gt_samples if next_temp == 1.0 else None,
        ],
        labels_list=["mcmc_refined_extrapolation", "gt"],
        alpha=0.7,
    )
    plt.legend()
    plt.title(f"temp={next_temp}")

    plt.savefig(plot_path)
    if wandb.run is not None:
        wandb.log(
            {
                f"plots/temp_{high_temp: .3f}_{low_temp: .3f}": wandb.Image(plt),
            }
        )
    plt.close()


def train_single_dm(
    target: TargetDistribution,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    train_set: Optional[torch.Tensor],
    Epochs: int,
    bsz: int,
    normalization: float = 1.0,
    max_grad_norm: float = 1.0,
    loss_log_path: str = "loss",
    adaptive_data_sigma: bool = False,
    device: str = "cuda",
):
    model.train()
    if adaptive_data_sigma:
        set_attr(model, 'data_sigma', (train_set / normalization).std())
    progress_bar = tqdm(range(Epochs), desc=f"Training")
    skipped_batches = 0
    for iter in progress_bar:
        t = (torch.randn((bsz,), device=device) * 1.2 - 1.1).exp()
        if train_set is None:
            batch = target.sample([bsz]) / normalization
        else:
            if train_set.shape[0] >= bsz:
                # If we have enough samples, sample without replacement
                sample_idx = torch.multinomial(
                    torch.ones(train_set.shape[0]), bsz, replacement=False
                )
            else:
                # If we don't have enough samples, sample with replacement
                sample_idx = torch.multinomial(
                    torch.ones(train_set.shape[0]), bsz, replacement=True
                )
            batch = train_set[sample_idx, :].to(device) / normalization
        noise = torch.randn_like(batch)
        if target.is_se3:
            noise = remove_mean(
                noise, n_particles=target.n_particles, n_dimensions=target.n_dimensions
            )
        dsm_loss = DSM_loss(batch, model, noise, t)
        dsm_loss.backward()

        # Check for NaN gradients
        has_nan_grad = False
        for param in model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan_grad = True
                break

        if has_nan_grad:
            print(f"Skipping batch due to NaN gradients")
            skipped_batches += 1
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()
        opt.zero_grad()
        progress_bar.set_postfix({"loss": dsm_loss.item()})
        if wandb.run is not None:
            wandb.log(
                {
                    loss_log_path: dsm_loss.item(),
                }
            )

    print(f"Skipped {skipped_batches} batches due to NaN gradients")

    return model, opt


def train(
    target: TargetDistribution,
    models: List[torch.nn.Module],
    opts: List[torch.optim.Optimizer],
    hparams,
    init_buffer: Optional[torch.Tensor] = None,
):
    """
    hparams:
        Epochs
        total_n_temp
        device
        bsz
        all_temps
        LG_step_size
        LG_steps
        update_buffer_intervel
        tmin
        tmax
        BUFFER_MAX_SIZE  # Renamed from BUFFER_MAX_SIZE
        swap_buffer_intervel
        check_intervel
        num_plot_samples
        plot_fold
        min_inter_dist
        max_inter_dist

    For the chains at the init phase, the important parameters are:
        init_LG_steps
        init_num_abundance
        init_LG_pick_interval
        init_num_chains

    And for the chains in the later phases, the equivalent parameters are:
        LG_steps
        num_abundance
        LG_pick_interval
        num_chains

    New parameters:
        num_dm_samples: number of samples to generate from diffusion model

    Calculation for how many samples we get per level from the chains for the init phase:
        total_samples = init_num_chains * (init_LG_steps - init_num_abundance) / init_LG_pick_interval
        e.g., init_num_chains = 100, init_LG_steps = 1000, init_num_abundance = 100, init_LG_pick_interval = 9
        total_samples = 100 * (1000 - 100) / 9 = 10000
    Calculation for how many samples we get per level from the chains for the latter phase:
        total_samples = num_chains * (LG_steps - num_abundance) / LG_pick_interval
        e.g., num_chains = 100, LG_steps = 1000, num_abundance = 100, LG_pick_interval = 9
        total_samples = 100 * (1000 - 100) / 9 = 10000
    """
    all_temps = hparams.all_temps
    if not isinstance(all_temps, torch.Tensor):
        all_temps = torch.tensor(all_temps).to(hparams.device)
    if not isinstance(hparams.LG_step_size, torch.Tensor):
        new_step_size = torch.tensor(hparams.LG_step_size[-2:]).to(hparams.device)
    else:
        new_step_size = hparams.LG_step_size[-2:]

    assert hparams.init_num_abundance < hparams.init_LG_steps
    assert hparams.num_abundance < hparams.LG_steps
    assert (
        hparams.init_LG_steps - hparams.init_num_abundance
    ) // hparams.init_LG_pick_interval > 0
    assert (hparams.LG_steps - hparams.num_abundance) // hparams.LG_pick_interval > 0

    # Initialize sample storage for each temperature level
    SAMPLES_BY_TEMP = {}  # Dictionary to store samples for each temperature

    # Define energy function that preserves gradients and includes debugging
    def energy_func(x):
        # Don't detach from computation graph
        # Use x directly without creating a new detached tensor
        log_prob = target.log_prob(x)
        return -log_prob

    if init_buffer is None:
        # Run PT to initialize the highest two temps' samples if init_buffer is not given
        temps = all_temps[-2:]
        # TODO: make it compatible to ALDP internal coordiante
        if isinstance(target, AldpPotentialCart):
            x_init = target.get_min_energy_position().to(hparams.device)
            init_samples = (
                x_init.unsqueeze(0)
                .repeat(2, hparams.init_num_chains, 1)
                .to(torch.float32)
            )
        else:
            init_samples = (
                torch.randn((2, hparams.init_num_chains, target.dim)).to(hparams.device)
                * 0.01
            )

        # Make sure tensors require grad and are on the correct device
        init_samples = (
            init_samples.detach().clone().to(hparams.device).requires_grad_(True)
        )

        pt = ParallelTempering(
            init_samples,
            energy_func=energy_func,
            step_size=new_step_size.repeat_interleave(
                hparams.init_num_chains
            ).unsqueeze(-1),
            swap_interval=hparams.swap_buffer_interval,
            temperatures=temps,
            mh=True,
            device=hparams.device,
            point_estimator=False,
        )
        pt = DynSamplerWrapper(
            pt, per_temp=True, total_n_temp=2, target_acceptance_rate=0.6, alpha=0.25
        )
        progress_bar = tqdm(
            range(hparams.init_LG_steps),
            desc="Langevin MCMC for highest 2 temperatures",
        )
        pt_collected_samples = []
        for iter in progress_bar:
            new_samples, acc, new_step_size = pt.sample()
            if (
                iter + 1 > hparams.init_num_abundance
                and (iter + 1) % hparams.init_LG_pick_interval == 0
            ):
                pt_collected_samples.append(new_samples.clone().detach().cpu())
            progress_bar.set_postfix(
                {"acc:": acc.mean().item(), "replica_acc:": pt.sampler.swap_rate}
            )
        new_step_size[1] = new_step_size[
            0
        ]  # init the step size at lower temp as the one at higher temp

        # Concatenate all collected samples
        all_collected_samples = torch.cat(
            pt_collected_samples, dim=1
        )  # shape (2, total_collected, dim)

        # Store samples for the two highest temperatures, resizing as needed
        SAMPLES_BY_TEMP[all_temps[-1].item()] = resize_sample_buffer(
            all_collected_samples[1], hparams.BUFFER_MAX_SIZE, dim=0
        )
        SAMPLES_BY_TEMP[all_temps[-2].item()] = resize_sample_buffer(
            all_collected_samples[0], hparams.BUFFER_MAX_SIZE, dim=0
        )

        # Save initial samples
        saved_samples = (
            torch.stack(
                [
                    SAMPLES_BY_TEMP[all_temps[-2].item()],
                    SAMPLES_BY_TEMP[all_temps[-1].item()],
                ]
            )
            .clone()
            .detach()
            .cpu()
            .float()
        )

        torch.save(
            {
                "data": saved_samples,
                "config": {
                    "temp_low": hparams.temp_low,
                    "temp_high": hparams.temp_high,
                    "total_n_temp": hparams.total_n_temp,
                    "temp_schedule": hparams.temp_schedule,
                },
            },
            f"{hparams.model_save_path}/init_samples.pt",
        )
    else:
        # Use provided init_buffer
        SAMPLES_BY_TEMP[all_temps[-1].item()] = init_buffer[1].clone().detach().cpu()
        SAMPLES_BY_TEMP[all_temps[-2].item()] = init_buffer[0].clone().detach().cpu()

    # Compute ranges for clipping based on highest temperature samples
    box_min = (
        SAMPLES_BY_TEMP[all_temps[-1].item()].to(hparams.device).min(dim=0).values * 2
    )
    box_max = (
        SAMPLES_BY_TEMP[all_temps[-1].item()].to(hparams.device).max(dim=0).values * 2
    )

    progress_bar = tqdm(range(hparams.total_n_temp - 1, 0, -1), desc="Taylor Sweep")
    for temp_idx in progress_bar:
        high_temp, low_temp = all_temps[temp_idx], all_temps[temp_idx - 1]
        model_high_temp = models[temp_idx]
        model_low_temp = models[temp_idx - 1]
        opt_high_temp = opts[temp_idx]
        opt_low_temp = opts[temp_idx - 1]

        if temp_idx < hparams.total_n_temp - 1:
            # Initialize model_low_temp from model_high_temp
            model_low_temp.load_state_dict(model_high_temp.state_dict())
            opt_low_temp.load_state_dict(opt_high_temp.state_dict())

        print(f"Training for temps {low_temp:.2f} and {high_temp:.2f}")
        # Use samples from SAMPLES_BY_TEMP for training
        high_temp_samples = SAMPLES_BY_TEMP[high_temp.item()].to(hparams.device)
        low_temp_samples = SAMPLES_BY_TEMP[low_temp.item()].to(hparams.device)

        # Make sure we have the right number of samples for training
        high_temp_samples_train = resize_sample_buffer(
            high_temp_samples, hparams.TRAINING_SET_SIZE, dim=0
        )
        low_temp_samples_train = resize_sample_buffer(
            low_temp_samples, hparams.TRAINING_SET_SIZE, dim=0
        )

        model_high_temp, opt_high_temp = train_single_dm(
            target=target,
            model=model_high_temp,
            opt=opt_high_temp,
            train_set=high_temp_samples_train,
            Epochs=(
                hparams.init_Epochs
                if temp_idx == hparams.total_n_temp - 1
                else hparams.Epochs
            ),
            bsz=hparams.bsz,
            normalization=hparams.normalization,
            max_grad_norm=hparams.max_grad_norm,
            loss_log_path=f"loss/temp_{high_temp: .2f}",
            adaptive_data_sigma=hparams.adaptive_data_sigma,
            device=hparams.device,
        )

        model_low_temp, opt_low_temp = train_single_dm(
            target=target,
            model=model_low_temp,
            opt=opt_low_temp,
            train_set=low_temp_samples_train,
            Epochs=(
                hparams.init_Epochs
                if temp_idx == hparams.total_n_temp - 1
                else hparams.Epochs
            ),
            bsz=hparams.bsz,
            normalization=hparams.normalization,
            max_grad_norm=hparams.max_grad_norm,
            loss_log_path=f"loss/temp_{low_temp: .2f}",
            adaptive_data_sigma=hparams.adaptive_data_sigma,
            device=hparams.device,
        )

        # Save checkpoint
        checkpoint = {
            'high_temp_model_state_dict': unwrap_data_parallel(
                model_high_temp
            ).state_dict(),
            'low_temp_model_state_dict': unwrap_data_parallel(
                model_low_temp
            ).state_dict(),
            'high_temp_optimizer_state_dict': opt_high_temp.state_dict(),
            'low_temp_optimizer_state_dict': opt_low_temp.state_dict(),
            'high_temp_samples': SAMPLES_BY_TEMP[high_temp.item()],
            'low_temp_samples': SAMPLES_BY_TEMP[low_temp.item()],
        }

        os.makedirs(hparams.model_save_path, exist_ok=True)
        torch.save(
            checkpoint,
            f"{hparams.model_save_path}/model_{low_temp: .2f}_{high_temp: .2f}.pt",
        )

        if temp_idx == 1:
            break

        # Sample from the model with Taylor guidance for the next buffer
        def taylor_guided_model(t, x):
            d_temp = high_temp - low_temp
            next_temp = all_temps[temp_idx - 2]
            low_temp_model_output = models[temp_idx - 1](t, x)
            high_temp_model_output = models[temp_idx](t, x)
            temp_gradient = (high_temp_model_output - low_temp_model_output) / d_temp
            return low_temp_model_output + temp_gradient * (next_temp - low_temp)

        with torch.no_grad():
            # Generate samples for the next temperature using the diffusion model
            if hparams.importance_resample and (temp_idx > 2 or hparams.is_last_step):
                sample_func = Euler_solve_wrapper(
                    Euler_solve_with_log_pdf,
                    num_samples_per_batch=hparams.num_samples_to_generate_per_batch,
                )
                print(f"Generating {hparams.num_dm_samples} samples, with IS weights")
                generate_samples, generate_log_probs = sample_func(
                    target,
                    model=taylor_guided_model,
                    start_samples=torch.randn(
                        hparams.num_dm_samples, target.dim, device=hparams.device
                    )
                    * hparams.tmax,
                    tmax=hparams.tmax,
                    tmin=hparams.tmin,
                    num_steps=hparams.is_integration_steps,
                )
                generate_samples = generate_samples * hparams.normalization
                if isinstance(target, AldpPotentialCart):
                    generate_samples = target.reflect_d_to_l_cartesian(generate_samples)
                target_log_probs = (
                    target.log_prob(generate_samples) / all_temps[temp_idx - 2]
                )
                # Cap the maximum IS weight
                log_prob_threshold = torch.quantile(
                    target_log_probs - generate_log_probs,
                    hparams.is_weight_max_quantile,
                )
                log_prob_diff = target_log_probs - generate_log_probs
                capped_diff = torch.minimum(log_prob_diff, log_prob_threshold)
                generate_log_probs = target_log_probs - capped_diff
                dm_samples, ess_ratio, weights = importance_sample_with_reweighting(
                    samples=generate_samples,
                    model_log_probs=generate_log_probs,
                    target_log_probs=target_log_probs,
                )
            else:
                sample_func = Euler_solve_wrapper(
                    Eular_solve,
                    num_samples_per_batch=hparams.num_samples_to_generate_per_batch,
                )
                print(
                    f"Generating {hparams.num_dm_samples} samples, without IS weights"
                )
                dm_samples = (
                    sample_func(
                        target,
                        taylor_guided_model,
                        torch.randn(
                            hparams.num_dm_samples, target.dim, device=hparams.device
                        )
                        * hparams.tmax,
                        tmax=hparams.tmax,
                        tmin=hparams.tmin,
                    )
                    * hparams.normalization
                )

            if isinstance(target, AldpPotentialCart):
                dm_samples = target.reflect_d_to_l_cartesian(dm_samples)
            dm_samples = torch.clamp(dm_samples, box_min, box_max)
            print(f"amount of nan samples: {torch.isnan(dm_samples).any(dim=1).sum()}")
            dm_samples = dm_samples[~torch.isnan(dm_samples).any(dim=1)]
            print(f"amount of samples after filtering nan: {dm_samples.shape[0]}")

            # Generate samples for visualization and evaluation
            sample_func = Euler_solve_wrapper(
                Eular_solve,
                num_samples_per_batch=hparams.num_samples_to_generate_per_batch,
            )
            generated_high_temp_samples = (
                sample_func(
                    target,
                    model_high_temp,
                    torch.randn(
                        hparams.num_plot_samples, target.dim, device=hparams.device
                    )
                    * hparams.tmax,
                )
                * hparams.normalization
            )

            generated_low_temp_samples = (
                sample_func(
                    target,
                    model_low_temp,
                    torch.randn(
                        hparams.num_plot_samples, target.dim, device=hparams.device
                    )
                    * hparams.tmax,
                )
                * hparams.normalization
            )

            if isinstance(target, AldpPotentialCart):
                generated_high_temp_samples = target.reflect_d_to_l_cartesian(
                    generated_high_temp_samples
                )
                generated_low_temp_samples = target.reflect_d_to_l_cartesian(
                    generated_low_temp_samples
                )

        generated_high_temp_samples = torch.clamp(
            generated_high_temp_samples, box_min, box_max
        )
        generated_low_temp_samples = torch.clamp(
            generated_low_temp_samples, box_min, box_max
        )

        # Now we need to prepare the next temperature level
        next_temp = all_temps[temp_idx - 2]

        # Select samples for PT initialization from both diffusion-generated samples and current buffer
        num_chains_pt = min(hparams.num_chains, dm_samples.shape[0])

        # Prepare initial PT samples: a subset from extrapolated samples and a subset from low_temp samples
        if dm_samples.shape[0] <= num_chains_pt:
            dm_init_samples = dm_samples
        else:
            # Sample without replacement
            indices = torch.randperm(dm_samples.shape[0])[:num_chains_pt]
            dm_init_samples = dm_samples[indices]

        low_temp_samples = SAMPLES_BY_TEMP[low_temp.item()].to(hparams.device)
        if low_temp_samples.shape[0] <= num_chains_pt:
            low_init_samples = low_temp_samples
        else:
            indices = torch.randperm(low_temp_samples.shape[0])[:num_chains_pt]
            low_init_samples = low_temp_samples[indices]

        print("Start fine-tuning samples for next level training")
        temps = all_temps[temp_idx - 2 : temp_idx]

        # Run PT with the initial samples
        pt_init_samples = torch.stack(
            [
                dm_init_samples.clone().detach().requires_grad_(True),  # For next_temp
                low_init_samples.clone().detach().requires_grad_(True),  # For low_temp
            ],
            dim=0,
        )

        # Ensure tensors require grad and are on the correct device
        pt_init_samples = (
            pt_init_samples.detach().clone().to(hparams.device).requires_grad_(True)
        )

        # Use a wrapper that forces manual gradient preservation
        pt = ParallelTempering(
            x=pt_init_samples,  # Already has requires_grad=True
            energy_func=energy_func,
            step_size=new_step_size.repeat_interleave(
                pt_init_samples.shape[1]
            ).unsqueeze(-1),
            swap_interval=hparams.swap_buffer_interval,
            temperatures=temps,
            mh=True,
            device=hparams.device,
            point_estimator=False,
        )
        pt = DynSamplerWrapper(
            pt, per_temp=True, total_n_temp=2, target_acceptance_rate=0.6, alpha=0.25
        )

        if hparams.LG_steps > 0:
            """This part has some interesting features:
            - If the PT result is exactly one sample for one sample in the buffer (pt_samples size is BUFFER_MAX_SIZE),
            then we just use those samples
            - Otherwise, we mix the resulting samples with the direct diffusion model samples
            """
            progress_bar_lg = tqdm(
                range(hparams.LG_steps), desc="Langevin MCMC fine-tuning"
            )
            pt_collected_samples = []

            for iter in progress_bar_lg:
                new_samples, acc, new_step_size = pt.sample()
                if (
                    iter + 1 > hparams.num_abundance
                    and (iter + 1) % hparams.LG_pick_interval == 0
                ):
                    pt_collected_samples.append(new_samples.clone().detach())
                progress_bar_lg.set_postfix(
                    {"acc:": acc.mean().item(), "replica_acc:": pt.sampler.swap_rate}
                )

            new_step_size[1] = new_step_size[
                0
            ]  # init the step size at lower temp as the one at higher temp

            # Collect PT samples
            if len(pt_collected_samples) > 0:
                pt_samples = torch.cat(
                    pt_collected_samples, dim=1
                )  # Shape: [2, num_collected, dim]
                next_temp_pt_samples = pt_samples[0]  # Samples for next_temp from PT
                current_low_temp_samples = pt_samples[1]
            else:
                next_temp_pt_samples = new_samples[0]
                current_low_temp_samples = new_samples[1]

            # Make sure both tensors are on the same device before concatenation
            next_temp_pt_samples = next_temp_pt_samples.to(hparams.device)
            current_low_temp_samples = current_low_temp_samples.to(hparams.device)
            dm_samples = dm_samples.to(hparams.device)

            # The following option is used in the case where we run PT on a subset of the buffer
            if num_chains_pt != hparams.BUFFER_MAX_SIZE:
                # TODO: in the middle ground where next_temp_pt_samples size is slightly less than BUFFER_MAX_SIZE, this does not work very well
                # Simply combine PT samples with diffusion-generated samples and then use resize_sample_buffer
                print(f"Running PT on {num_chains_pt} chains!")
                combined_next_temp_samples = torch.cat(
                    [next_temp_pt_samples, dm_samples], dim=0
                )
                # Resize the combined samples to get the desired number of samples
                next_temp_samples = resize_sample_buffer(
                    combined_next_temp_samples, hparams.BUFFER_MAX_SIZE, dim=0
                )
                SAMPLES_BY_TEMP[next_temp.item()] = next_temp_samples.cpu()
            else:  # used when we run PT on the whole buffer
                print("Running PT on the exact whole buffer!")
                next_temp_samples = next_temp_pt_samples
                current_low_temp_samples = current_low_temp_samples
                SAMPLES_BY_TEMP[next_temp.item()] = next_temp_samples.cpu()
                SAMPLES_BY_TEMP[low_temp.item()] = current_low_temp_samples.cpu()
        else:
            # If no PT steps, just use the diffusion samples for next temp
            SAMPLES_BY_TEMP[next_temp.item()] = resize_sample_buffer(
                dm_samples, hparams.BUFFER_MAX_SIZE, dim=0
            ).cpu()

        # Plot or evaluate results
        if isinstance(target, AldpPotential) or isinstance(target, AldpPotentialCart):
            save_dir = f'{hparams.plot_fold}/training/{all_temps[temp_idx]:.2f}-{all_temps[temp_idx-1]:.2f}'
            os.makedirs(save_dir, exist_ok=True)

            # Get samples for visualization
            next_temp_viz_samples = resize_sample_buffer(
                SAMPLES_BY_TEMP[next_temp.item()].to(hparams.device),
                hparams.num_plot_samples,
                dim=0,
            )
            low_temp_viz_samples = resize_sample_buffer(
                SAMPLES_BY_TEMP[low_temp.item()].to(hparams.device),
                hparams.num_plot_samples,
                dim=0,
            )
            high_temp_viz_samples = resize_sample_buffer(
                SAMPLES_BY_TEMP[high_temp.item()].to(hparams.device),
                hparams.num_plot_samples,
                dim=0,
            )

            for subdir, source_sample, target_sample in zip(
                [f'{low_temp: .2f}', f'{high_temp: .2f}', 'extrapolation'],
                [
                    generated_low_temp_samples,
                    generated_high_temp_samples,
                    next_temp_viz_samples,
                ],
                [
                    low_temp_viz_samples,
                    high_temp_viz_samples,
                    target.sample([next_temp_viz_samples.shape[0]]),
                ],
            ):

                os.makedirs(f'{save_dir}/{subdir}', exist_ok=True)
                images_dict = target.plot_samples(
                    source_sample.to(hparams.device),
                    target_sample.to(hparams.device),
                    iter=0,
                    metric_dir=f'{save_dir}/{subdir}',
                    plot_dir=f'{save_dir}/{subdir}',
                    batch_size=1000000,
                )
                for key, value in images_dict.items():
                    wandb.log(
                        {
                            f"temp_{all_temps[temp_idx]:.2f}-{all_temps[temp_idx-1]:.2f}/{subdir}/{key}": wandb.Image(
                                value
                            )
                        }
                    )
        else:
            next_temp_dm_samples = dm_samples.to(hparams.device)
            next_temp_viz_samples = resize_sample_buffer(
                SAMPLES_BY_TEMP[next_temp.item()].to(hparams.device),
                hparams.num_plot_samples,
                dim=0,
            )
            low_temp_viz_samples = resize_sample_buffer(
                SAMPLES_BY_TEMP[low_temp.item()].to(hparams.device),
                hparams.num_plot_samples,
                dim=0,
            )
            high_temp_viz_samples = resize_sample_buffer(
                SAMPLES_BY_TEMP[high_temp.item()].to(hparams.device),
                hparams.num_plot_samples,
                dim=0,
            )

            evaluate(
                target,
                low_temp_samples=generated_low_temp_samples[: hparams.num_plot_samples],
                high_temp_samples=generated_high_temp_samples[
                    : hparams.num_plot_samples
                ],
                extrapolate_samples=next_temp_dm_samples,
                mcmc_refined_extrapolate_samples=next_temp_viz_samples,
                low_temp_buffer=low_temp_viz_samples,
                high_temp_buffer=high_temp_viz_samples,
                gt_samples=target.sample([hparams.num_plot_samples]),
                low_temp=low_temp,
                high_temp=high_temp,
                next_temp=next_temp,
                plot_path=f'{hparams.plot_fold}/training/loss_{temp_idx}_{temp_idx-1}.png',
            )

    # Train final model at temperature 1
    final_temp = all_temps[0]
    final_samples = SAMPLES_BY_TEMP[final_temp.item()].to(hparams.device)
    final_samples_train = resize_sample_buffer(
        final_samples, hparams.TRAINING_SET_SIZE, dim=0
    )

    models[0], opts[0] = train_single_dm(
        target=target,
        model=models[0],
        opt=opts[0],
        train_set=final_samples_train,
        Epochs=hparams.Epochs,
        bsz=hparams.bsz,
        normalization=hparams.normalization,
        max_grad_norm=hparams.max_grad_norm,
        loss_log_path=f"loss/temp_{final_temp: .2f}",
        adaptive_data_sigma=hparams.adaptive_data_sigma,
        device=hparams.device,
    )

    # Save final model
    checkpoint = {
        "model_state_dict": unwrap_data_parallel(models[0]).state_dict(),
        "opt_state_dict": opts[0].state_dict(),
    }
    torch.save(checkpoint, f"{hparams.model_save_path}/model_{final_temp}.pt")

    # Generate target samples from trained T1 model
    with torch.no_grad():
        sample_func = Euler_solve_wrapper(
            Eular_solve, num_samples_per_batch=hparams.num_samples_to_generate_per_batch
        )
        samples = (
            sample_func(
                target,
                models[0],
                torch.randn(hparams.BUFFER_MAX_SIZE, target.dim, device=hparams.device)
                * hparams.tmax,
            )
            * hparams.normalization
        )

        samples = torch.clamp(samples, box_min, box_max)

        if isinstance(target, AldpPotentialCart):
            samples = target.reflect_d_to_l_cartesian(samples)

    # Save all samples for evaluation
    all_samples_dict = {
        temp.item(): SAMPLES_BY_TEMP[temp.item()]
        for temp in all_temps
        if temp.item() in SAMPLES_BY_TEMP
    }
    checkpoint = {
        "target_samples": samples.clone().detach().float().cpu(),
        "all_temp_samples": all_samples_dict,
    }
    os.makedirs(os.path.dirname(hparams.model_save_path), exist_ok=True)
    torch.save(checkpoint, f"{hparams.model_save_path}/eval_set.pt")

    # Evaluate the final model
    if isinstance(target, AldpPotential) or isinstance(target, AldpPotentialCart):
        save_dir = f'{hparams.plot_fold}/training/{all_temps[0]:.2f}/model'
        os.makedirs(save_dir, exist_ok=True)
        target_sample = target.sample([samples.shape[0]])
        images_dict = target.plot_samples(
            samples.to(hparams.device),
            target_sample.to(hparams.device),
            iter=0,
            metric_dir=f'{save_dir}',
            plot_dir=f'{save_dir}',
            batch_size=1000000,
        )
        for key, value in images_dict.items():
            wandb.log({f"temp_{all_temps[0]:.2f}/model/{key}": wandb.Image(value)})

        save_dir = f'{hparams.plot_fold}/training/{all_temps[0]:.2f}/samples'
        os.makedirs(save_dir, exist_ok=True)
        final_viz_samples = resize_sample_buffer(
            SAMPLES_BY_TEMP[final_temp.item()].to(hparams.device),
            min(samples.shape[0], hparams.num_plot_samples),
            dim=0,
        )
        images_dict = target.plot_samples(
            final_viz_samples,
            target_sample[: final_viz_samples.shape[0]],
            iter=0,
            metric_dir=f'{save_dir}',
            plot_dir=f'{save_dir}',
            batch_size=1000000,
        )
        for key, value in images_dict.items():
            wandb.log({f"temp_{all_temps[0]:.2f}/samples/{key}": wandb.Image(value)})
    else:
        plt.figure(figsize=(6, 6))
        target.plot_samples(
            samples_list=[samples, target.sample([samples.shape[0]])],
            labels_list=["model", "gt"],
        )
        plt.legend()
        plt.savefig(f'{hparams.plot_fold}/training/final_model.png')
        if wandb.run is not None:
            wandb.log(
                {
                    f"plots/final": wandb.Image(plt),
                }
            )
        plt.close()

    return (
        models,
        opts,
        {
            temp.item(): SAMPLES_BY_TEMP[temp.item()]
            for temp in all_temps
            if temp.item() in SAMPLES_BY_TEMP
        },
    )


def train_ptdm(
    target: TargetDistribution,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    pt_samples: torch.Tensor,
    metric_list: List[Metric],
    hparams,
):
    """
    pt_samples: removed burn in, with shape [num_chains, num_steps, dim]
    check_intervals:
    """
    # num_training_steps = 1#(pt_samples.shape[0] * pt_samples.shape[1]) // hparams.energy_call_per_training_step
    # progress_bar = tqdm(range(num_training_steps), desc=f"Training PT+DM")
    # for iter in progress_bar:
    train_set = pt_samples  # [:, :(iter + 1) * hparams.energy_call_per_training_step // pt_samples.shape[0], :].reshape(-1, pt_samples.shape[-1])
    model, opt = train_single_dm(
        target=target,
        model=model,
        opt=opt,
        train_set=train_set,
        Epochs=hparams.Epochs,
        bsz=hparams.bsz,
        normalization=hparams.normalization,
        max_grad_norm=hparams.max_grad_norm,
        adaptive_data_sigma=hparams.adaptive_data_sigma,
        device=hparams.device,
    )
    # Save the trained model
    if hparams.model_save_path:
        os.makedirs(hparams.model_save_path, exist_ok=True)
        model_save_file = os.path.join(hparams.model_save_path, 'model.pt')
        torch.save(model.state_dict(), model_save_file)
        print(f"Model saved to {model_save_file}")

    with torch.no_grad():
        sample_func = Euler_solve_wrapper(
            Eular_solve, num_samples_per_batch=hparams.num_samples_to_generate_per_batch
        )
        samples = (
            sample_func(
                target,
                model,
                torch.randn(hparams.num_eval_samples, target.dim, device=hparams.device)
                * hparams.tmax,
            )
            * hparams.normalization
        )
        print(f"samples shape: {samples.shape}")
        if isinstance(target, AldpPotentialCart):
            samples = target.reflect_d_to_l_cartesian(samples)
        target_sample = target.sample([samples.shape[0]])
        print(f"target_sample shape: {target_sample.shape}")
        # num_current_energy_calls = int((iter + 1) * hparams.energy_call_per_training_step)
        if isinstance(target, AldpPotential) or isinstance(target, AldpPotentialCart):
            save_dir = f'{hparams.plot_fold}/'
            os.makedirs(save_dir, exist_ok=True)
            images_dict = target.plot_samples(
                samples.to(hparams.device),
                target_sample.to(hparams.device),
                iter=0,
                metric_dir=f'{save_dir}',
                plot_dir=f'{save_dir}',
                batch_size=1000000,
            )
            for key, value in images_dict.items():
                wandb.log({f"{key}/": wandb.Image(value)})
        else:
            print(f"Start plotting samples")
            plt.figure(figsize=(6, 6))
            target.plot_samples(
                samples_list=[samples, target_sample], labels_list=["model", "gt"]
            )
            plt.legend()
            plt.savefig(f'{hparams.plot_fold}/training/samples.png')
            wandb.log({f"plots/": wandb.Image(plt)})
            plt.close()

        print(f"Start evaluating metrics")
        for metric_func in metric_list:
            metric = metric_func(
                samples[: hparams.num_metric_samples],
                target_sample[: hparams.num_metric_samples],
            )
            wandb.log({f"metrics/{metric_func.__name__}": metric})


def train_gtdm(
    target: TargetDistribution,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    metric_list: List[Metric],
    hparams,
):
    num_train_steps = hparams.Epochs // hparams.check_interval
    progress_bar = tqdm(range(num_train_steps), desc=f"Training GT+DM")
    for iter in progress_bar:
        model, opt = train_single_dm(
            target=target,
            model=model,
            opt=opt,
            train_set=None,
            Epochs=hparams.check_interval,
            bsz=hparams.bsz,
            normalization=hparams.normalization,
            max_grad_norm=hparams.max_grad_norm,
            adaptive_data_sigma=hparams.adaptive_data_sigma,
            device=hparams.device,
        )

        with torch.no_grad():
            sample_func = Euler_solve_wrapper(
                Eular_solve,
                num_samples_per_batch=hparams.num_samples_to_generate_per_batch,
            )
            samples = (
                sample_func(
                    target,
                    model,
                    torch.randn(
                        (
                            hparams.num_val_samples
                            if iter < num_train_steps - 1
                            else hparams.num_test_samples
                        ),
                        target.dim,
                        device=hparams.device,
                    )
                    * hparams.tmax,
                )
                * hparams.normalization
            )
            if isinstance(target, AldpPotentialCart):
                samples = target.reflect_d_to_l_cartesian(samples)
            target_sample = target.sample([samples.shape[0]])
            for metric_func in metric_list:
                metric = metric_func(
                    samples[: hparams.num_metric_samples],
                    target_sample[: hparams.num_metric_samples],
                )
                wandb.log({f"val/{metric_func.__name__}": metric})
            if isinstance(target, AldpPotential) or isinstance(
                target, AldpPotentialCart
            ):
                save_dir = f'{hparams.plot_fold}/iter_{iter+1}'
                os.makedirs(save_dir, exist_ok=True)
                images_dict = target.plot_samples(
                    samples.to(hparams.device),
                    target_sample.to(hparams.device),
                    iter=0,
                    metric_dir=f'{save_dir}',
                    plot_dir=f'{save_dir}',
                    batch_size=1000000,
                )
                for key, value in images_dict.items():
                    wandb.log({f"{key}/": wandb.Image(value)})
            else:
                plt.figure(figsize=(6, 6))
                target.plot_samples(
                    samples_list=[samples, target_sample], labels_list=["model", "gt"]
                )
                plt.legend()
                plt.savefig(f'{hparams.plot_fold}/training/iter_{iter+1}.png')
                wandb.log({f"plots/": wandb.Image(plt)})
                plt.close()
