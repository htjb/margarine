"""KDE implementation using JAX."""

import jax.numpy as jnp
import jax.scipy.stats as stats

from margarine.base.baseflow import BaseDensityEstimator
from margarine.utils import approximate_bounds


class KDE(BaseDensityEstimator):
    """Kernel Density Estimator (KDE) using JAX."""

    def __init__(
        self,
        theta: jnp.ndarray,
        weights: jnp.ndarray | None = None,
        theta_ranges: jnp.ndarray | None = None,
        bandwidth: float | str = "silverman",
    ) -> None:
        """Initialize the KDE.

        Args:
            theta: Parameters of the density estimator.
            weights: Optional weights for the parameters.
            theta_ranges: Optional ranges for the parameters.
            bandwidth: Bandwidth for the KDE.
        """
        super().__init__(theta)
        self.weights = weights
        self.theta_ranges = theta_ranges
        self.bandwidth = bandwidth

        if self.weights is None:
            self.weights = jnp.ones_like(self.theta)

        if theta_ranges is None:
            self.theta_ranges = jnp.concatenate(
                approximate_bounds(self.theta, self.weights)
            )

    def train(self) -> stats.gaussian_kde:
        """Generates a weighted KDE."""
        return None
