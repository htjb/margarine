"""KDE implementation using JAX."""

import pickle

import jax.numpy as jnp
import jax.scipy.stats as stats
from tensorflow_probability.substrates import jax as tfp

from margarine.base.baseflow import BaseDensityEstimator
from margarine.utils.utils import (
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
        self.theta = theta
        self.weights = weights
        self.theta_ranges = theta_ranges
        self.bandwidth = bandwidth

        if self.weights is None:
            self.weights = jnp.ones(len(self.theta))

        if theta_ranges is None:
            self.theta_ranges = approximate_bounds(self.theta, self.weights)

    def train(self) -> stats.gaussian_kde:
        """Generates a weighted KDE."""
        phi = forward_transform(
            self.theta, self.theta_ranges[0], self.theta_ranges[1]
        )
        weights = self.weights / jnp.sum(self.weights)
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
        x = inverse_transform(x, self.theta_ranges[0], self.theta_ranges[1])
        return x

    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        r"""Transform samples from the unit hypercube to samples on the KDE.

        Uses the Rosenblatt transformation
        (conditional inverse transform sampling)
        to map uniform samples to a Gaussian mixture KDE distribution.

        Args:
            u (jnp.ndarray): Samples from the unit hypercube [0,1]^d
                Shape: (n_samples, n_dims)

        Returns:
            jnp.ndarray: The transformed samples following the
                KDE distribution.
                Shape: (n_samples, n_dims)
        """
        return NotImplementedError("KDE __call__ method is not implemented.")

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
            x, self.theta_ranges[0], self.theta_ranges[1]
        )

        transform_chain = tfb.Chain(
            [
                tfb.Invert(tfb.NormalCDF()),
                tfb.Scale(1 / (self.theta_ranges[0] - self.theta_ranges[1])),
                tfb.Shift(-self.theta_ranges[1]),
            ]
        )

        def norm_jac(y: jnp.ndarray) -> jnp.ndarray:
            """Calculate the normalising jacobian for the transformation."""
            return transform_chain.inverse_log_det_jacobian(y, event_ndims=0)

        correction = norm_jac(transformed_x).sum(axis=1)

        log_probs = jnp.log(self.kde.pdf(transformed_x.T)) + correction
        return log_probs

    def log_like(
        self,
        x: jnp.ndarray,
        logevidence: float,
        prior_density: jnp.ndarray | BaseDensityEstimator,
    ) -> jnp.ndarray:
        """Compute the marginal log-likelihood of given samples.

        Args:
            x: Samples for which to compute the log-likelihood.
            logevidence: Log-evidence term.
            prior_density: Prior density or density estimator.

        Returns:
            jnp.ndarray: Log-likelihoods of the samples.
        """
        if isinstance(prior_density, BaseDensityEstimator):
            prior_density = prior_density.log_prob(x)

        return self.log_prob(x) + logevidence - prior_density

    def save(self, filepath: str) -> None:
        """Save the KDE to a file.

        Args:
            filepath: Path to the file where the KDE will be saved.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self.kde, f)

    @classmethod
    def load(cls, filepath: str) -> "KDE":
        """Load a KDE from a file.

        Args:
            filepath: Path to the file from which to load the KDE.

        Returns:
            KDE: Loaded KDE instance.
        """
        with open(filepath, "rb") as f:
            kde = pickle.load(f)

        instance = cls(jnp.array([]))  # Dummy initialization
        instance.kde = kde
        return instance
