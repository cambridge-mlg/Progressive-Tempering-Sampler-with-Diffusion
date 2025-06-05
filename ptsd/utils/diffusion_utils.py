import torch
import numpy as np
from typing import Callable, Tuple
from ptsd.utils.se3_utils import remove_mean
from ptsd.targets.base_target import TargetDistribution
from tqdm import tqdm


def Eular_solve(
    target: TargetDistribution,
    model: torch.nn.Module,
    start_samples: torch.Tensor,
    tmax: float = 40.0,
    tmin: float = 1e-3,
    rho: float = 7,
    num_steps: int = 500,
):

    ts = tmin ** (1 / rho) + np.arange(num_steps) / (num_steps - 1) * (
        tmax ** (1 / rho) - tmin ** (1 / rho)
    )
    ts = ts**rho
    progress_bar = tqdm(
        range(ts.shape[0] - 1, 0, -1),
        desc="Sampling from diffusion. Batch size {}".format(start_samples.shape[0]),
    )
    with torch.no_grad():
        samples = (
            start_samples
            if not target.is_se3
            else remove_mean(start_samples, target.n_particles, target.n_dimensions)
        )
        for i in progress_bar:
            t = torch.ones(samples.shape[0], 1).to(samples.device) * ts[i]
            t_1 = torch.ones(samples.shape[0], 1).to(samples.device) * ts[i - 1]
            x_hat = model(t.squeeze(-1).log(), samples)

            samples = samples * t_1 / t + (1 - t_1 / t) * x_hat
            if target.is_se3:
                samples = remove_mean(samples, target.n_particles, target.n_dimensions)
        return samples


def compute_logprob_score(log_prob_func, x: torch.Tensor) -> Tuple:
    """
    It returns
    """
    with torch.set_grad_enabled(True):
        samples = x.clone().detach().requires_grad_(True)
        log_probs = log_prob_func(samples)
        scores = torch.autograd.grad(
            log_probs.sum(),
            inputs=samples,
        )[0]
    return log_probs.clone().detach(), scores.clone().detach()


def Euler_solve_with_energy_guidance(
    target: TargetDistribution,
    model: torch.nn.Module,
    start_samples: torch.Tensor,
    tmax: float = 40.0,
    tmin: float = 1e-3,
    rho: float = 7,
    num_steps: int = 500,
    samples: torch.Tensor = None,
    log_prob_differences: torch.Tensor = None,
):

    ts = tmin ** (1 / rho) + np.arange(num_steps) / (num_steps - 1) * (
        tmax ** (1 / rho) - tmin ** (1 / rho)
    )
    ts = ts**rho

    with torch.no_grad():
        current_samples = (
            start_samples
            if not target.is_se3
            else remove_mean(start_samples, target.n_particles, target.n_dimensions)
        )
        for i in range(ts.shape[0] - 1, 0, -1):
            t = (
                torch.ones(current_samples.shape[0], 1).to(current_samples.device)
                * ts[i]
            )
            t_1 = (
                torch.ones(current_samples.shape[0], 1).to(current_samples.device)
                * ts[i - 1]
            )

            # Get model prediction
            x_hat = model(t.squeeze(-1).log(), current_samples)

            # Calculate guidance weights based on equation
            if samples is not None and log_prob_differences is not None:
                # Calculate distances to reference samples
                diff = current_samples.unsqueeze(1) - samples.unsqueeze(
                    0
                )  # Shape: [batch, num_samples, dim]
                squared_distances = (diff**2).sum(dim=-1)  # Shape: [batch, num_samples]

                # Calculate weights with corrected sign for log_prob_differences
                log_weights = log_prob_differences - squared_distances / (2 * t**2)
                weights = torch.softmax(
                    log_weights, dim=1
                )  # Shape: [batch, num_samples]

                # Calculate guidance term
                guidance = (weights.unsqueeze(-1) * diff).sum(
                    dim=1
                ) / t**2  # Shape: [batch, dim]

                # Add guidance to model prediction, scaled by t**2 since x_hat = x + t**2 * score
                x_hat = x_hat + guidance * t**2

            # Euler step
            current_samples = current_samples * t_1 / t + (1 - t_1 / t) * x_hat

            # Center molecules if needed
            if target.is_se3:
                samples = remove_mean(samples, target.n_particles, target.n_dimensions)

        return current_samples


def sample_in_batches(sampling_func, init_sample_generator, batch_size, total_samples):
    with torch.no_grad():
        output_list = []
        for i in range(0, total_samples, batch_size):
            batch_size_to_use = min(batch_size, total_samples - i)
            batch_samples = sampling_func(init_sample_generator(batch_size_to_use))
            output_list.append(batch_samples)
            print(f"Processed {i} samples out of {total_samples}")
    return output_list


def Euler_solve_with_log_pdf(
    target: TargetDistribution,
    model: torch.nn.Module,
    start_samples: torch.Tensor,
    tmax: float = 40.0,
    tmin: float = 1e-3,
    rho: float = 7,
    num_steps: int = 2000,
):

    ts = tmin ** (1 / rho) + np.arange(num_steps) / (num_steps - 1) * (
        tmax ** (1 / rho) - tmin ** (1 / rho)
    )
    ts = ts**rho

    # Calculate effective dimensionality for centered molecules
    if target.is_se3:
        d = (
            np.prod(start_samples.shape[1:]) - target.n_dimensions
        )  # subtract degrees of freedom from centering
    else:
        d = np.prod(start_samples.shape[1:])

    # Initialize with centered Gaussian log pdf
    log_prob = -0.5 * d * np.log(2 * np.pi) - d * np.log(tmax)
    if target.is_se3:
        # For centered configurations, use the norm in the centered subspace
        start_samples = remove_mean(
            start_samples, target.n_particles, target.n_dimensions
        )
        log_prob = log_prob - torch.sum(
            start_samples**2, dim=tuple(range(1, len(start_samples.shape)))
        ) / (2 * tmax**2)
    else:
        log_prob = log_prob - torch.sum(
            start_samples**2, dim=tuple(range(1, len(start_samples.shape)))
        ) / (2 * tmax**2)

    with torch.enable_grad():
        samples = start_samples.requires_grad_(True)

        for i in range(ts.shape[0] - 1, 0, -1):
            t = torch.ones(samples.shape[0], 1).to(samples.device) * ts[i]
            t_1 = torch.ones(samples.shape[0], 1).to(samples.device) * ts[i - 1]

            x_hat = model(t.squeeze(-1).log(), samples)
            f = -(x_hat - samples) / t

            epsilon = (
                torch.randint(0, 2, samples.shape, device=samples.device).float() * 2
                - 1
            )

            # For molecules, we need to account for the centering constraint in our Jacobian calculations.
            # Let P be the projection matrix that centers the molecules (removes mean).
            # We want to estimate Tr(PJP^T) where J is the Jacobian of our drift.
            # Using the Hutchinson estimator, E[ε^T PJP^T ε] = Tr(PJP^T).
            # This can be computed by simply projecting (centering) ε before using it
            # in our trace estimation.
            if target.is_se3:
                # Center the epsilon to ensure we're estimating trace in the correct subspace
                epsilon = remove_mean(epsilon, target.n_particles, target.n_dimensions)

            vjp = torch.autograd.grad(
                f, samples, epsilon, create_graph=False, retain_graph=False
            )[0]
            trace_est = (vjp * epsilon).sum(dim=tuple(range(1, len(samples.shape))))

            dt = t_1 - t
            log_prob = log_prob - trace_est * dt.squeeze(-1)

            # Euler step
            samples = samples * t_1 / t + (1 - t_1 / t) * x_hat
            if target.is_se3:
                samples = remove_mean(samples, target.n_particles, target.n_dimensions)

            samples = samples.detach().requires_grad_(True)

        return samples.detach(), log_prob


def Euler_solve_with_log_pdf_full_hessian(
    target: TargetDistribution,
    model: torch.nn.Module,
    start_samples: torch.Tensor,
    tmax: float = 40.0,
    tmin: float = 1e-3,
    rho: float = 7,
    num_steps: int = 2000,
):

    ts = tmin ** (1 / rho) + np.arange(num_steps) / (num_steps - 1) * (
        tmax ** (1 / rho) - tmin ** (1 / rho)
    )
    ts = ts**rho

    # Calculate effective dimensionality for centered molecules
    if target.is_se3:
        d = (
            np.prod(start_samples.shape[1:]) - target.n_dimensions
        )  # subtract degrees of freedom from centering
    else:
        d = np.prod(start_samples.shape[1:])

    # Initialize with centered Gaussian log pdf
    log_prob = -0.5 * d * np.log(2 * np.pi) - d * np.log(tmax)
    if target.is_se3:
        # For centered configurations, use the norm in the centered subspace
        start_samples = remove_mean(
            start_samples, target.n_particles, target.n_dimensions
        )
        log_prob = log_prob - torch.sum(
            start_samples**2, dim=tuple(range(1, len(start_samples.shape)))
        ) / (2 * tmax**2)
    else:
        log_prob = log_prob - torch.sum(
            start_samples**2, dim=tuple(range(1, len(start_samples.shape)))
        ) / (2 * tmax**2)

    with torch.enable_grad():
        samples = start_samples.requires_grad_(True)

        for i in range(ts.shape[0] - 1, 0, -1):
            t = torch.ones(samples.shape[0], 1).to(samples.device) * ts[i]
            t_1 = torch.ones(samples.shape[0], 1).to(samples.device) * ts[i - 1]

            x_hat = model(t.squeeze(-1).log(), samples)
            f = -(x_hat - samples) / t

            # Instead of Hutchinson estimation, compute full Jacobian
            if target.is_se3:
                # For molecules, we need to compute trace in the centered subspace
                # Create basis for the centered subspace
                batch_size = samples.shape[0]
                n_particles = target.n_particles
                n_dim = target.n_dimensions

                # Compute Jacobian for each sample in batch
                trace_list = []
                for b in range(batch_size):
                    sample_f = (
                        lambda x: (
                            -model(t.squeeze(-1).log(), x.unsqueeze(0)) - x.unsqueeze(0)
                        )
                        / t
                    )

                    # Compute full Jacobian
                    J = torch.autograd.functional.jacobian(sample_f, samples[b])
                    J = J.squeeze(0)  # Remove batch dimension

                    if target.is_se3:
                        # Project Jacobian to centered subspace
                        # J_centered = P J P, where P is centering projection

                        #  Construct centering projection matrix P = I - 1/N (1 1^T ⊗ I_d)
                        # We can apply it efficiently without constructing full matrix
                        bs = J.shape[0]
                        J_reshaped = J.reshape(
                            bs, n_particles, n_dim, n_particles, n_dim
                        )

                        # Apply P from right: J P
                        J_right = J_reshaped - J_reshaped.mean(dim=3, keepdim=True)

                        # Apply P from left: P (J P)
                        J_centered = J_right - J_right.mean(dim=1, keepdim=True)

                        # Reshape back and take trace
                        J_centered = J_centered.reshape(
                            bs, n_particles * n_dim, n_particles * n_dim
                        )

                        trace = torch.trace(J_centered, dim1=1, dim2=2)
                    else:
                        trace = torch.trace(J)

                    trace_list.append(trace)

                trace_est = torch.stack(trace_list)
            else:
                # For non-molecular cases, directly compute trace for each sample
                trace_list = []
                for b in range(samples.shape[0]):
                    sample_f = (
                        lambda x: (
                            -model(t.squeeze(-1).log(), x.unsqueeze(0)) - x.unsqueeze(0)
                        )
                        / t
                    )
                    J = torch.autograd.functional.jacobian(sample_f, samples[b])
                    trace_list.append(torch.trace(J.squeeze(0)))
                trace_est = torch.stack(trace_list)

            dt = t_1 - t
            log_prob = log_prob - trace_est * dt.squeeze(-1)

            # Euler step
            samples = samples * t_1 / t + (1 - t_1 / t) * x_hat
            if target.is_se3:
                samples = remove_mean(samples, target.n_particles, target.n_dimensions)

            samples = samples.detach().requires_grad_(True)

        return samples.detach(), log_prob


def importance_sample_with_reweighting(
    samples: torch.Tensor,
    model_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    normalize_target: bool = True,
    target_shift: float = -5.0,
) -> tuple[torch.Tensor, float]:
    """
    Perform importance sampling with reweighting to correct for distribution mismatch.

    Args:
        samples: Tensor of samples from the model distribution
        model_log_probs: Log probabilities of samples under the model distribution
        target_log_probs: Log probabilities of samples under the target distribution
        normalize_target: Whether to normalize target log probs by subtracting max (default: True)
        target_shift: Constant shift applied to normalized target log probs (default: -5.0)

    Returns:
        tuple containing:
            - resampled_samples: Tensor of resampled points
            - ess_ratio: Effective sample size ratio (between 0 and 1)
            - weights: Importance weights
    """
    # Check for invalid values
    valid_mask = ~(
        torch.isnan(model_log_probs)
        | torch.isinf(model_log_probs)
        | torch.isnan(target_log_probs)
        | torch.isinf(target_log_probs)
    )

    if not torch.any(valid_mask):
        raise ValueError(
            "No valid samples found - all log probabilities are nan or inf"
        )

    # Filter out invalid samples
    valid_samples = samples[valid_mask]
    valid_model_log_probs = model_log_probs[valid_mask]
    valid_target_log_probs = target_log_probs[valid_mask]

    # Normalize target log probs if requested
    if normalize_target:
        valid_target_log_probs = (
            valid_target_log_probs - valid_target_log_probs.max() + target_shift
        )

    # Calculate importance weights in log space
    log_weights = valid_target_log_probs - valid_model_log_probs

    # Normalize weights for numerical stability
    log_weights = log_weights - log_weights.max()
    normalized_weights = torch.softmax(log_weights, dim=0)

    # Resample indices according to normalized weights
    num_samples = len(samples)  # Note: we still return original number of samples
    resampled_indices = torch.multinomial(
        normalized_weights, num_samples, replacement=True
    )

    # Get resampled samples
    resampled_samples = valid_samples[resampled_indices]

    # Calculate effective sample size ratio
    ess = (normalized_weights.sum() ** 2) / (normalized_weights**2).sum()
    ess_ratio = ess.item() / len(
        valid_samples
    )  # Note: using valid sample count for ratio

    return resampled_samples, ess_ratio, normalized_weights


def Euler_solve_wrapper(sample_func: Callable, num_samples_per_batch: int):
    def wrapped_sample_func(
        target: TargetDistribution,
        model: torch.nn.Module,
        start_samples: torch.Tensor,
        tmax: float = 40.0,
        tmin: float = 1e-3,
        rho: float = 7,
        num_steps: int = 500,
    ):
        num_generate_samples = start_samples.shape[0]
        num_iters = num_generate_samples // num_samples_per_batch
        outputs1, outputs2 = [], []
        print(
            "Sampling from diffusion. Total samples {}. Batch size {}. num_iters {}".format(
                start_samples.shape[0], num_samples_per_batch, num_iters
            )
        )
        for i in range(num_iters):
            suboutputs = sample_func(
                target=target,
                model=model,
                start_samples=start_samples[
                    i * num_samples_per_batch : (i + 1) * num_samples_per_batch
                ],
                tmax=tmax,
                tmin=tmin,
                rho=rho,
                num_steps=num_steps,
            )
            if isinstance(suboutputs, tuple):
                outputs1.append(suboutputs[0])
                outputs2.append(suboutputs[1])
            else:
                outputs1.append(suboutputs)
        if num_iters * num_samples_per_batch < num_generate_samples:
            suboutputs = sample_func(
                target=target,
                model=model,
                start_samples=start_samples[num_iters * num_samples_per_batch :],
                tmax=tmax,
                tmin=tmin,
                rho=rho,
                num_steps=num_steps,
            )
            if isinstance(suboutputs, tuple):
                outputs1.append(suboutputs[0])
                outputs2.append(suboutputs[1])
            else:
                outputs1.append(suboutputs)
        outputs1 = torch.cat(outputs1, dim=0)
        if outputs2:
            outputs2 = torch.cat(outputs2, dim=0)
            return outputs1, outputs2
        else:
            return outputs1

    return wrapped_sample_func


def evaluate_log_likelihood(
    target: TargetDistribution,
    model: torch.nn.Module,
    samples: torch.Tensor,
    tmax: float = 40.0,
    tmin: float = 1e-3,
    rho: float = 7,
    num_steps: int = 2000,
):
    """
    Evaluate the log likelihood of given samples under the diffusion model.

    Args:
        target: Target distribution
        model: Score-based model that approximates the score function
        samples: Input samples to evaluate [batch_size, ...]
        tmax: Maximum diffusion time
        tmin: Minimum diffusion time
        rho: Parameter controlling the time schedule
        num_steps: Number of integration steps

    Returns:
        log_likelihood: Log likelihood of the input samples
    """
    # Generate time steps from tmin to tmax
    ts = tmin ** (1 / rho) + np.arange(num_steps) / (num_steps - 1) * (
        tmax ** (1 / rho) - tmin ** (1 / rho)
    )
    ts = ts**rho

    # Calculate effective dimensionality for centered molecules
    if target.is_se3:
        d = (
            np.prod(samples.shape[1:]) - target.n_dimensions
        )  # subtract degrees of freedom from centering
    else:
        d = np.prod(samples.shape[1:])

    with torch.enable_grad():
        current_samples = samples.clone().detach().requires_grad_(True)
        if target.is_se3:
            current_samples = remove_mean(
                current_samples, target.n_particles, target.n_dimensions
            )

        # Start with the log probability under the data distribution
        # Initialize this to zero as we'll track the change in log probability
        log_likelihood = torch.zeros(samples.shape[0], device=samples.device)

        # Forward diffusion process (ODE from data to noise)
        for i in range(1, ts.shape[0]):
            t_prev = (
                torch.ones(current_samples.shape[0], 1).to(current_samples.device)
                * ts[i - 1]
            )
            t = (
                torch.ones(current_samples.shape[0], 1).to(current_samples.device)
                * ts[i]
            )

            # Get score estimate from the model
            x_hat = model(t_prev.squeeze(-1).log(), current_samples)
            f = -(x_hat - current_samples) / t_prev

            # Estimate trace of Jacobian using Hutchinson's estimator
            epsilon = (
                torch.randint(
                    0, 2, current_samples.shape, device=current_samples.device
                ).float()
                * 2
                - 1
            )
            if target.is_se3:
                # Center the epsilon to ensure we're estimating trace in the correct subspace
                epsilon = remove_mean(epsilon, target.n_particles, target.n_dimensions)

            vjp = torch.autograd.grad(
                f, current_samples, epsilon, create_graph=False, retain_graph=False
            )[0]
            trace_est = (vjp * epsilon).sum(
                dim=tuple(range(1, len(current_samples.shape)))
            )

            # Update log likelihood (negative trace because we're going forward)
            dt = t - t_prev
            log_likelihood = log_likelihood + trace_est * dt.squeeze(-1)

            # Forward ODE step (deterministic, no noise)
            with torch.no_grad():
                current_samples = current_samples + f * dt

                if target.is_se3:
                    current_samples = remove_mean(
                        current_samples, target.n_particles, target.n_dimensions
                    )

            current_samples = current_samples.detach().requires_grad_(True)

        # Add log probability of the final noise distribution (Gaussian with variance tmax^2)
        log_likelihood = (
            log_likelihood
            - 0.5 * d * np.log(2 * np.pi * tmax**2)
            - torch.sum(
                current_samples**2, dim=tuple(range(1, len(current_samples.shape)))
            )
            / (2 * tmax**2)
        )

        return log_likelihood
