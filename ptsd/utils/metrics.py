import torch
import ot as pot
import numpy as np
import math

from ptsd.targets.base_target import TargetDistribution
from ptsd.utils.se3_utils import calculate_rmsd_matrix


def maximum_mean_discrepancy(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 1,
    fix_sigma=None,
):
    assert source.shape == target.shape and len(source.shape) <= 2
    # assert source.ndim == target.ndim and source.ndim <= 2
    if len(source.shape) == 1:
        source = source.unsqueeze(-1)
        target = target.unsqueeze(-1)

    def gaussian_kernel(x, y):
        n_samples = int(x.size()[0]) + int(y.size()[0])
        total = torch.cat([x, y], dim=0)

        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        #         print(bandwidth_list)
        kernel_val = [
            torch.exp(-L2_distance / bandwidth_temp)
            for bandwidth_temp in bandwidth_list
        ]
        return sum(kernel_val)

    batch_size = int(source.size()[0])
    kernels = gaussian_kernel(source, target)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    mmd = torch.mean(XX + YY - XY - YX)
    return mmd


def total_variation(source: torch.Tensor, target: torch.Tensor, num_bins: int = 200):
    """for 1d inputs now"""
    # assert source.shape == target.shape and len(source.shape) <= 2
    assert source.ndim == target.ndim and source.ndim <= 2
    if len(source.shape) == 1:
        H_data_set, x_data_set = np.histogram(target.cpu(), bins=num_bins)
        H_generated_samples, _ = np.histogram(source.cpu(), bins=(x_data_set))
        total_var = (
            0.5
            * np.abs(
                H_data_set / H_data_set.sum()
                - H_generated_samples / H_generated_samples.sum()
            ).sum()
        )
        total_var = torch.tensor(total_var, device=source.device)
    else:
        dim = math.prod(source.shape[1:])
        bins = (200,) * dim
        all_data = torch.cat([target, source], dim=0)
        min_vals, _ = all_data.min(dim=0)
        max_vals, _ = all_data.max(dim=0)
        ranges = tuple(
            (min_vals[i].item(), max_vals[i].item()) for i in range(dim)
        )  # tuple of (min, max) for each dimension
        ranges = tuple(item for subtuple in ranges for item in subtuple)
        hist_p, _ = torch.histogramdd(target.cpu(), bins=bins, range=ranges)
        hist_q, _ = torch.histogramdd(source.cpu(), bins=bins, range=ranges)

        p_dist = hist_p / hist_p.sum()
        q_dist = hist_q / hist_q.sum()

        total_var = 0.5 * torch.abs(p_dist - q_dist).sum()

    return total_var


def wasserstain2_dist(source: torch.Tensor, target: torch.Tensor):
    """
    For different length of source's shape:
        1: W2 distance on energy or interatomic distance
        2: W2 distance on non-SE(3) data
        3: W2 distance on SE(3) data
    """
    assert source.ndim == target.ndim
    if len(source.shape) == 1:
        return torch.tensor(
            pot.emd2_1d(source.cpu().numpy(), target.cpu().numpy()),
            device=source.device,
        )
    elif len(source.shape) == 2:
        distance_matrix = pot.dist(
            target.cpu().numpy(), source.cpu().numpy(), metric='euclidean'
        )
    elif len(source.shape) == 3:
        # For molecules
        distance_matrix = calculate_rmsd_matrix(target, source).cpu().numpy()
    else:
        raise "Incorrect data shape..."
    src, dist = np.ones(len(target)) / len(target), np.ones(len(source)) / len(source)
    G = pot.emd(src, dist, distance_matrix)
    w2_dist = np.sum(G * distance_matrix) / G.sum()
    w2_dist = torch.tensor(w2_dist, device=target.device)
    return w2_dist


class Metric:
    def __init__(self, name, target: TargetDistribution):
        self.__name__ = name
        self.target = target

    def __call__(self, source: torch.Tensor, target: torch.Tensor):
        pass


class MMD_energy(Metric):
    def __init__(
        self,
        target: TargetDistribution,
        kernel_mul: float = 2.0,
        kernel_num: int = 1,
        fix_sigma=None,
    ):
        super().__init__("MMD_energy", target)
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def __call__(self, source: torch.Tensor, target: torch.Tensor):
        return maximum_mean_discrepancy(
            -self.target.log_prob(source),
            -self.target.log_prob(target),
            self.kernel_mul,
            self.kernel_num,
            self.fix_sigma,
        )


class MMD_interatomic(Metric):
    def __init__(
        self,
        target: TargetDistribution,
        kernel_mul: float = 2.0,
        kernel_num: int = 1,
        fix_sigma=None,
    ):
        super().__init__("MMD_interatomic", target)
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
        if not hasattr(self.target, "interatomic_dist"):
            raise "Energy function does not have interatomic_dist method..."

    def __call__(self, source: torch.Tensor, target: torch.Tensor):
        return maximum_mean_discrepancy(
            self.target.interatomic_dist(source),
            self.target.interatomic_dist(target),
            self.kernel_mul,
            self.kernel_num,
            self.fix_sigma,
        )


class TVD_energy(Metric):
    def __init__(self, target: TargetDistribution, num_bins: int = 200):
        super().__init__("TVD_energy", target)
        self.num_bins = num_bins

    def __call__(self, source: torch.Tensor, target: torch.Tensor):
        return total_variation(
            -self.target.log_prob(source), -self.target.log_prob(target), self.num_bins
        )


class TVD_interatomic(Metric):
    def __init__(self, target: TargetDistribution, num_bins: int = 200):
        super().__init__("TVD_interatomic", target)
        self.num_bins = num_bins
        if not hasattr(self.target, "interatomic_dist"):
            raise "Energy function does not have interatomic_dist method..."

    def __call__(self, source: torch.Tensor, target: torch.Tensor):
        return total_variation(
            self.target.interatomic_dist(source),
            self.target.interatomic_dist(target),
            self.num_bins,
        )


class Wasserstain2_energy(Metric):
    def __init__(self, target):
        super().__init__("Wasserstain2_energy", target)

    def __call__(self, source: torch.Tensor, target: torch.Tensor):
        return wasserstain2_dist(
            -self.target.log_prob(source), -self.target.log_prob(target)
        )


class Wasserstain2_data(Metric):
    def __init__(self, target: TargetDistribution):
        super().__init__("Wasserstain2_data", target)

    def __call__(self, source: torch.Tensor, target: torch.Tensor):
        return wasserstain2_dist(source, target)


class MMD_data(Metric):
    def __init__(
        self,
        target: TargetDistribution,
        kernel_mul: float = 2.0,
        kernel_num: int = 1,
        fix_sigma=None,
    ):
        super().__init__("MMD_data", target)
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def __call__(self, source: torch.Tensor, target: torch.Tensor):
        return maximum_mean_discrepancy(
            source, target, self.kernel_mul, self.kernel_num, self.fix_sigma
        )
