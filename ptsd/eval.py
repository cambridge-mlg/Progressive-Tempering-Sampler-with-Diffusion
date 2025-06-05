import torch
from typing import List
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import torch
from ptsd.utils.metrics import (
    MMD_energy,
    TVD_energy,
    Wasserstain2_energy,
    Wasserstain2_data,
)
from ptsd.utils.diffusion_utils import Eular_solve, Euler_solve_wrapper
from ptsd.utils.metrics import Metric
import os
from ptsd.targets.aldp import AldpPotential


def get_samples_from_all_temps(
    target_name, temp_schedule, total_n_temp, temp_low, temp_high, device
):
    """
    Load all the samples for a given temperature schedule.
    """
    all_temps_reference_file = f"data/pt/{target_name}_schedule_{temp_schedule}_total_n_temp_{total_n_temp}_temp_low_{temp_low}_temp_high_{temp_high}.pt"
    if os.path.exists(all_temps_reference_file):
        samples = torch.load(all_temps_reference_file, map_location=device)
        return samples
    else:
        return None


def evaluate_samples(target_func, t1_samples, pt_samples, save_path):
    """
    Evaluate generated samples against baseline and ground truth.
    A wrapper function to combine all the evaluation metrics.
    """
    eval(
        gt_samples=target_func.sample([10000]),
        sample_dict=(
            {"ptsd": t1_samples.unsqueeze(0)} | {"pt": pt_samples.unsqueeze(0)}
            if pt_samples is not None
            else {}
        ),
        metric_list=[
            MMD_energy(target_func, kernel_num=5, fix_sigma=1),
            TVD_energy(target_func),
            Wasserstain2_energy(target_func),
            Wasserstain2_data(target_func),
        ],
        save_path=save_path,
    )


def get_eval_metrics(target_func, samples, reference_samples):
    """
    samples: shape [N, dim]
    reference_samples: shape [M, dim]
    output: list of metrics comparing these two sets of samples
    """
    metrics = (
        {
            "MMD_energy": MMD_energy(target_func, kernel_num=5, fix_sigma=1),
            "TVD_energy": TVD_energy(target_func),
            "Wasserstain2_energy": Wasserstain2_energy(target_func),
            "Wasserstain2_data": Wasserstain2_data(target_func),
        }
        if not isinstance(target_func, AldpPotential)
        else {"Wasserstain2_data": Wasserstain2_data(target_func)}
    )
    results = {}
    for metric_name, metric in metrics.items():
        results[metric_name] = (
            metric(samples.clone().detach(), reference_samples.clone().detach())
            .detach()
            .cpu()
            .item()
        )
    return results


def write_metrics_to_csv(metrics_file: str, eval_metrics: dict):
    """
    Write evaluation metrics to a CSV file. Creates header if file doesn't exist,
    otherwise appends metrics as a new row.

    Args:
        metrics_file (str): Path to the CSV file
        eval_metrics (dict): Dictionary containing metric names and values
    """
    # Create header if file doesn't exist
    if not os.path.exists(metrics_file):
        with open(metrics_file, "w") as f:
            f.write(",".join(eval_metrics.keys()) + "\n")

    # Append metrics as new row
    with open(metrics_file, "a") as f:
        f.write(",".join(str(v) for v in eval_metrics.values()) + "\n")


def generate_samples(
    target_func, model, target_name, dim, device, hparams, num_samples=10000
):
    """
    Generate samples using the appropriate sampling method for the target.
    A wrapper function to unify the interface for sampling for different targets.
    """
    with torch.no_grad():
        if target_name == 'lj55':
            sample_func = Euler_solve_wrapper(
                Eular_solve,
                num_samples_per_batch=hparams.num_samples_to_generate_per_batch,
            )
            return (
                sample_func(
                    target_func,
                    model,
                    torch.randn([num_samples, dim], device=device) * hparams.tmax,
                )
                * hparams.normalization
            )
        else:
            return (
                Eular_solve(
                    target_func,
                    model,
                    torch.randn([num_samples, dim], device=device) * hparams.tmax,
                )
                * hparams.normalization
            )


def load_pt_samples(pt_samples_path, target, device):
    # Load baseline PT samples
    if pt_samples_path is None:
        return None
    pt_samples = torch.load(pt_samples_path, map_location=device)
    if target == 'lj13':
        pt_samples = pt_samples[0, 10000:, :]
    elif target in ['lj55', 'mw32']:
        pt_samples = pt_samples[0]
    pt_samples = pt_samples[-10000:]
    return pt_samples


def eval(
    gt_samples: torch.Tensor,
    sample_dict: dict[torch.Tensor],
    metric_list: List[Metric],
    save_path: str,
    file_name="eval_metrics.txt",
):
    """
    sample_dict: each sample in sample_dict should be of shape [k, N, dim], where k can be varied but N should be the same
    """
    assert gt_samples.ndim == 2
    num_metrics = len(metric_list)
    plt.figure(figsize=(6 * num_metrics, 6))
    pg_metric = tqdm(enumerate(metric_list), desc="metric")

    # Create a dictionary to store all metrics for saving to file
    all_metrics = {}

    for i, metric in pg_metric:
        print(f"running {metric.__name__}")
        plt.subplot(1, num_metrics, i + 1)
        plt.title(metric.__name__)
        pg_sample = tqdm(sample_dict.items(), desc="sample")

        for sample_name, sample in pg_sample:
            assert sample.ndim == 3
            metrics = []
            for step in range(sample.shape[0]):
                value = (
                    metric(source=sample[step].clone().detach(), target=gt_samples)
                    .detach()
                    .cpu()
                    .item()
                )
                metrics.append(value)
                if wandb.run is not None:
                    wandb.log({f"eval/{metric.__name__}/{sample_name}": value})

            # Store metrics in dictionary
            metric_key = f"{metric.__name__}_{sample_name}"
            all_metrics[metric_key] = metrics

            plt.plot(metrics, 'o', label=sample_name)
            pg_sample.set_postfix_str(f"{sample_name}")
        plt.legend()

    plt.savefig(f"{save_path}/eval.png")
    plt.close()

    # Save metrics to text file
    with open(f"{save_path}/{file_name}", "a") as f:
        for metric_name, values in all_metrics.items():
            f.write(f"{metric_name}: {values}\n")
            # Also save the final value
            if values:
                f.write(f"{metric_name}_final: {values[-1]}\n")


def setup_quadratic_function(samples, seed=0):
    """
    Setup the quadratic function f(x) = (x - x_shift)^T A (x - x_shift) + b^T (x - x_shift)
    for ESS calculation."""
    # torch.manual_seed(seed)
    # x_shift = torch.randn(samples.shape[1])
    # A = torch.randn(samples.shape[1], samples.shape[1])
    # b = torch.randn(samples.shape[1])

    x_shift = torch.zeros(samples.shape[1])
    A = torch.eye(samples.shape[1])
    b = torch.zeros(samples.shape[1])

    return x_shift, A, b


def evaluate_quadratic_function(samples, seed=0):
    """
    Evaluate the quadratic function f(x) = (x - x_shift)^T A (x - x_shift) + b^T (x - x_shift)
    for ESS calculation.

    Args:
        samples: Tensor of shape [num_samples, dim]
        seed: Random seed for reproducibility

    Returns:
        Tensor of shape [num_samples] containing function values
    """
    device = samples.device
    x_shift, A, b = setup_quadratic_function(samples, seed)
    x_shift = x_shift.to(device)
    A = A.to(device)
    b = b.to(device)

    # Compute (x - x_shift) for all samples
    centered_x = samples - x_shift

    # Compute (x - x_shift)^T A (x - x_shift) + b^T (x - x_shift)
    quadratic_term = torch.sum(centered_x * (torch.matmul(centered_x, A.T)), dim=1)
    linear_term = torch.sum(centered_x * b, dim=1)

    return quadratic_term + linear_term


# Function to calculate ESS
def calculate_ess(samples, target, n_repetitions=100, seed=0):
    """
    Calculate the effective sample size (ESS) using importance sampling.

    Args:
        samples: Tensor of shape [num_samples, dim]
        target: Target distribution
        n_repetitions: Number of repetitions for averaging
        seed: Random seed for reproducibility

    Returns:
        Average ESS ratio
    """
    torch.manual_seed(seed)

    ess_ratios = []
    for i in range(n_repetitions):
        # Set up quadratic function with different seed for each repetition
        function_values = evaluate_quadratic_function(samples, seed=i)

        # Calculate log weights
        log_weights = target.log_prob(samples)

        # Normalize weights
        weights = torch.exp(log_weights - torch.logsumexp(log_weights, dim=0))

        # Calculate ESS
        ess = 1.0 / torch.sum(weights**2)
        ess_ratio = ess / len(samples)

        ess_ratios.append(ess_ratio.item())

    return sum(ess_ratios) / len(ess_ratios)


def calculate_avg_mae(target, samples, target_samples, use_importance_weights=False):
    """
    Calculate the average MAE of the estimation of E[f(x)] over all the samples.
    """
    mae_list = []
    # for i in range(1):
    mae_list.append(
        calculate_mae(
            target,
            samples,
            target_samples,
            use_importance_weights=use_importance_weights,
        )
    )
    return torch.tensor(mae_list).mean().item()


# Function to calculate MAE
def calculate_mae(
    target, samples, target_samples, use_importance_weights=False, seed=0
):
    """
    Calculate the mean absolute error (MAE) in the estimation of E[f(x)].

    Args:
        samples: Tensor of shape [num_samples, dim]
        target: Target distribution
        use_importance_weights: Whether to use importance weights
        n_samples: Number of ground truth samples to use

    Returns:
        MAE as a percentage of the true expectation
    """
    # ground truth samples
    gt_samples = target_samples

    # Evaluate function on ground truth samples
    gt_function_values = evaluate_quadratic_function(gt_samples, seed=seed)
    true_expectation = torch.mean(gt_function_values)

    # Evaluate function on our samples
    function_values = evaluate_quadratic_function(samples, seed=seed)

    if use_importance_weights:
        # Calculate importance weights
        log_weights = target.log_prob(samples)
        weights = torch.exp(log_weights - torch.logsumexp(log_weights, dim=0))
        estimated_expectation = torch.sum(weights * function_values)
    else:
        estimated_expectation = torch.mean(function_values)

    # Calculate MAE as percentage
    mae_percentage = (
        100
        * torch.abs(estimated_expectation - true_expectation)
        / torch.abs(true_expectation)
    )
    return mae_percentage
