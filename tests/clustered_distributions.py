"""Generate clustered distributions for testing.

Code modified from
https://github.com/VincentStimper/normalizing-flows repository.
"""

import numpy as np
import torch
from torch import nn


class Target(nn.Module):
    """Sample target distributions to test models."""

    def __init__(
        self,
        prop_scale: torch.Tensor = torch.tensor(6.0),
        prop_shift: torch.Tensor = torch.tensor(-3.0),
    ) -> None:
        """Constructor.

        Args:
          prop_scale (tf.Tensor): Scale for the uniform proposal
          prop_shift (tf.Tensor): Shift for the uniform proposal
        """
        super().__init__()
        self.register_buffer("prop_scale", prop_scale)
        self.register_buffer("prop_shift", prop_shift)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Base log probability function.

        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        raise NotImplementedError(
            "The log probability is not implemented yet."
        )

    def rejection_sampling(self, num_steps: int = 1) -> torch.Tensor:
        """Perform rejection sampling on image distribution.

        Args:
          num_steps: Number of rejection sampling steps to perform

        Returns:
          Accepted samples
        """
        eps = torch.rand(
            (num_steps, self.n_dims),
            dtype=self.prop_scale.dtype,
            device=self.prop_scale.device,
        )
        z_ = self.prop_scale * eps + self.prop_shift
        prob = torch.rand(
            num_steps,
            dtype=self.prop_scale.dtype,
            device=self.prop_scale.device,
        )
        prob_ = torch.exp(self.log_prob(z_) - self.max_log_prob)
        accept = prob_ > prob
        z = z_[accept, :]
        return z

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """Sample from image distribution through rejection sampling.

        Args:
          num_samples: Number of samples to draw

        Returns:
          Samples
        """
        z = torch.zeros(
            (0, self.n_dims),
            dtype=self.prop_scale.dtype,
            device=self.prop_scale.device,
        )
        while len(z) < num_samples:
            z_ = self.rejection_sampling(num_samples)
            ind = np.min([len(z_), num_samples - len(z)])
            z = torch.cat([z, z_[:ind, :]], 0)
        return z


class TwoMoons(Target):
    """Bimodal two-dimensional distribution."""

    def __init__(self) -> None:
        """Constructor."""
        super().__init__()
        self.n_dims = 2
        self.max_log_prob = 0.0

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Log probability function for the two moons distribution.

        ```
        log(p) = - 1/2 * ((norm(z) - 2) / 0.2) ** 2
                 + log(  exp(-1/2 * ((z[0] - 2) / 0.3) ** 2)
                       + exp(-1/2 * ((z[0] + 2) / 0.3) ** 2))
        ```

        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        a = torch.abs(z[:, 0])
        log_prob = (
            -0.5 * ((torch.norm(z, dim=1) - 2) / 0.2) ** 2
            - 0.5 * ((a - 2) / 0.3) ** 2
            + torch.log(1 + torch.exp(-4 * a / 0.09))
        )
        return log_prob
