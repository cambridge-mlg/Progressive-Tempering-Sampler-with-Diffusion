import abc
import torch
from typing import Optional, List


class TargetDistribution(abc.ABC):
    def __init__(self, dim, is_se3, n_particles):
        super().__init__()
        if not hasattr(self, "dim"):
            self.dim = dim
        self.is_se3 = is_se3
        self.n_particles = n_particles
        self.n_dimensions = dim // n_particles
        assert (is_se3 and n_particles > 1) or (not is_se3 and n_particles == 1)
        assert self.n_particles * self.n_dimensions == self.dim

    @abc.abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """returns (unnormalised) log probability of samples x"""
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, num_samples: Optional[int]) -> torch.Tensor:
        """returns samples from target distribution"""
        raise NotImplementedError

    @abc.abstractmethod
    def plot_samples(
        self, samples_list: List[torch.Tensor], labels_list: List[str], **kwargs
    ):
        """plot samples with customized labels"""
        raise NotImplementedError
