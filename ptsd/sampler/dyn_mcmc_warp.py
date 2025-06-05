from ptsd.sampler.sampler import MCMCSampler
from typing import Optional


class DynSamplerWrapper:
    def __init__(
        self,
        sampler: MCMCSampler,
        per_temp: bool = False,
        per_chain: bool = False,
        total_n_temp: Optional[int] = None,
        target_acceptance_rate: float = 0.6,
        alpha: float = 0.25,
    ):
        self.sampler = sampler
        self.target_acceptance_rate = target_acceptance_rate
        self.alpha = alpha
        self.per_temp = per_temp
        self.per_chain = per_chain
        self.total_n_temp = total_n_temp
        assert (per_temp and total_n_temp) or not per_temp
        # Cannot use per_chain without per_temp as we need temperature structure information
        assert not (per_chain and not per_temp)

    def sample(self) -> tuple:
        """Sample from the wrapped MCMC sampler and dynamically adjust step size.

        Returns:
            tuple: Contains:
                - new_samples (torch.Tensor): Samples with shape (total_n_temp, num_chains, dim) where:
                    - total_n_temp is the number of temperatures (e.g. 10)
                    - num_chains is the number of chains per temperature (e.g. 500)
                    - dim is the dimension of the problem (varies by target: 2 for GMM, 32 for MW, etc.)
                - acc_temp (torch.Tensor or float): Acceptance rates, per temperature if per_temp=True
                                                    or per chain if per_chain=True
                - org_step_size (torch.Tensor): Updated step sizes
        """
        new_samples, acc = self.sampler.sample()

        if self.per_chain:
            # Handle per-chain adaptation (requires per_temp to be True)
            assert len(self.sampler.step_size) > 0

            # Reshape step sizes to match temperature and chain structure
            chain_shape = acc.shape
            # Reshape step size to match chain structure (total_n_temp, chains_per_temp)
            org_step_size = self.sampler.step_size.squeeze(-1).view(chain_shape)

            # Apply adaptation to each chain individually
            above_target = acc > self.target_acceptance_rate
            org_step_size[above_target] = org_step_size[above_target] * (1 + self.alpha)
            org_step_size[~above_target] = org_step_size[~above_target] * (
                1 - self.alpha
            )

            # Update the sampler's step size, preserving per-chain information
            self.sampler.step_size = org_step_size.view(-1).unsqueeze(-1)

            # Return the per-chain acceptance rates
            acc_temp = acc

        elif self.per_temp:
            assert len(self.sampler.step_size) > 0
            acc_temp = acc.reshape(self.total_n_temp, -1).mean(dim=1)
            org_step_size = (
                self.sampler.step_size.squeeze(-1)
                .view(self.total_n_temp, -1)
                .mean(dim=1)
            )
            org_step_size[acc_temp > self.target_acceptance_rate] = org_step_size[
                acc_temp > self.target_acceptance_rate
            ] * (1 + self.alpha)
            org_step_size[acc_temp <= self.target_acceptance_rate] = org_step_size[
                acc_temp <= self.target_acceptance_rate
            ] * (1 - self.alpha)
            self.sampler.step_size = org_step_size.repeat_interleave(
                self.sampler.step_size.shape[0] // self.total_n_temp
            ).unsqueeze(-1)

        else:
            acc_temp = acc.mean().item()
            org_step_size = self.sampler.step_size
            if acc.mean().item() > self.target_acceptance_rate:
                org_step_size *= 1 + self.alpha  # Increase step size
            else:
                org_step_size *= 1 - self.alpha  # Decrease step size
            self.sampler.step_size = org_step_size

        return new_samples, acc_temp, org_step_size
