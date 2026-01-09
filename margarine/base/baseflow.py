r"""Base density estimator for margarine package.

Defines a base class for density estimators with common interface methods
including:

- train
- sample
- \_\_call\_\_
- log_prob
- log_like
- save
- load

"""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp


class BaseDensityEstimator(ABC):
    """Base class for density estimators in the margarine package."""

    @abstractmethod
    def train(self) -> None:
        """Train the density estimator on the provided data."""
        raise NotImplementedError("Train method must be implemented.")

    @abstractmethod
    def sample(self, key: jnp.ndarray, num_samples: int) -> jnp.ndarray:
        """Generate samples from the density estimator.

        Args:
            key: JAX random key for sampling.
            num_samples: Number of samples to generate.

        Returns:
            jnp.ndarray: Generated samples as a JAX array.
        """
        u = jax.random.uniform(key, shape=(num_samples, self.theta.shape[1]))
        raise self(u)

    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the density estimator at given points.

        Args:
            u: Samples from the unit hypercube.

        Returns:
            jnp.ndarray: samples from the density estimator.
        """
        raise NotImplementedError("Call method must be implemented.")

    @abstractmethod
    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the log-probability of given samples.

        Args:
            x: Samples for which to compute the log-probability.

        Returns:
            jnp.ndarray: Log-probabilities of the samples.
        """
        raise NotImplementedError("log_prob method must be implemented.")

    @abstractmethod
    def log_like(
        self,
        x: jnp.ndarray,
        logevidence: float,
        prior_density: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the log-likelihood of given samples.

        Args:
            x: Samples for which to compute the log-likelihood.
            logevidence: Log-evidence value.
            prior_density: Prior density estimator or densities.

        Returns:
            jnp.ndarray: Log-likelihoods of the samples.
        """
        return NotImplementedError("log_like method must be implemented.")

    def save(self, filepath: str) -> None:
        """Save the density estimator to a file.

        Args:
            filepath: Path to the file where the estimator will be saved.
        """
        raise NotImplementedError("save method must be implemented.")

    @classmethod
    def load(cls, filepath: str) -> "BaseDensityEstimator":
        """Load a density estimator from a file.

        Args:
            filepath: Path to the file from which to load the estimator.

        Returns:
            BaseDensityEstimator: Loaded density estimator instance.
        """
        raise NotImplementedError("load method must be implemented.")
