"""KDE implementation using JAX."""

import jax.numpy as jnp
import jax.scipy.stats as stats
from tensorflow_probability.substrates import jax as tfp

from margarine.base.baseflow import BaseDensityEstimator
from margarine.utils import (
    approximate_bounds,
    forward_transform,
    inverse_transform,
)

tfb = tfp.bijectors
tfd = tfp.distributions


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
            self.weights = jnp.ones(len(self.theta))

        if theta_ranges is None:
            self.theta_ranges = jnp.concatenate(
                approximate_bounds(self.theta, self.weights)
            )

    def train(self) -> stats.gaussian_kde:
        """Generates a weighted KDE."""
        phi = forward_transform(
            self.theta, self.theta_ranges[1], self.theta_ranges[0]
        )
        weights = self.weights / jnp.sum(self.weights)
        print(weights)
        self.kde = stats.gaussian_kde(
            phi.T, weights=weights, bw_method=self.bandwidth
        )
        return self.kde

    def sample(self, key: jnp.ndarray, num_samples: int) -> jnp.ndarray:
        """Sample from the KDE.

        Args:
            key: JAX random key for sampling.
            num_samples: Number of samples to draw.

        Returns:
            jnp.ndarray: Samples drawn from the KDE.
        """
        x = self.kde.resample(key, (num_samples,)).T
        x = inverse_transform(x, self.theta_ranges[1], self.theta_ranges[0])
        return x

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the log-probability of given samples.

        While the density estimator has its own built in log probability
        function, a correction has to be applied for the transformation of
        variables that is used to improve accuracy when learning. The
        correction is implemented here.

        Args:
            x: Samples for which to compute the log-probability.

        Returns:
            jnp.ndarray: Log-probabilities of the samples.
        """
        transformed_x = forward_transform(
            x, self.theta_ranges[1], self.theta_ranges[0]
        )

        transform_chain = tfb.Chain(
            [
                tfb.Invert(tfb.NormalCDF()),
                tfb.Scale(1 / (self.theta_ranges[1] - self.theta_ranges[0])),
                tfb.Shift(-self.theta_ranges[0]),
            ]
        )

        def norm_jac(y: jnp.ndarray) -> jnp.ndarray:
            """Calculate the normalising jacobian for the transformation."""
            return transform_chain.inverse_log_det_jacobian(y, event_ndims=0)

        correction = norm_jac(x).sum(axis=1)

        log_probs = jnp.log(self.kde.pdf(transformed_x.T)) + correction
        return log_probs
