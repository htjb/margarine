"""KDE implementation using JAX."""

import pickle
import shutil
from pathlib import Path
from zipfile import ZipFile

import jax.numpy as jnp
import jax.scipy.stats as stats
import yaml
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

    def save(self, filename: str) -> None:
        """Save the KDE to a file.

        Args:
            filename (str): Path to the file where the KDE will be saved.
        """
        path = Path(filename).resolve()
        if path.exists():
            shutil.rmtree(path)

        with open(f"{path}/kde.pkl", "wb") as f:
            pickle.dump(self.kde, f)

        config = {
            "bandwidth": self.bandwidth,
        }

        with open(f"{path}/config.yaml", "w") as f:
            yaml.dump(config, f)

        metadata = {
            "theta": self.theta,
            "weights": self.weights,
            "theta_ranges": self.theta_ranges,
        }

        with open(f"{path}/metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        with ZipFile(filename + ".marg", "w") as z:
            for subpath in path.rglob("*"):
                if subpath.is_file():
                    z.write(subpath, arcname=subpath.relative_to(path))

        shutil.rmtree(path)

    @classmethod
    def load(cls, filename: str) -> "KDE":
        """Load a KDE from a file.

        Args:
            filename (str): Path to the file from which to load the KDE.

        Returns:
            KDE: Loaded KDE instance.
        """
        zip_path = Path(f"{filename}.marg")
        path = Path(filename + ".tmp").resolve()
        with ZipFile(zip_path) as z:
            # Extract all files to a folder
            z.extractall(path)

        with open(f"{path}/config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        instance = cls(
            theta=jnp.array([]),  # Placeholder, will be overwritten
            weights=jnp.array([]),  # Placeholder, will be overwritten
            theta_ranges=jnp.array([]),  # Placeholder, will be overwritten
            bandwidth=config["bandwidth"],
        )

        with open(f"{path}/metadata.yaml") as f:
            metadata = yaml.load(f, Loader=yaml.FullLoader)

        instance.theta = jnp.array(metadata["theta"])
        instance.weights = jnp.array(metadata["weights"])
        instance.theta_ranges = jnp.array(metadata["theta_ranges"])

        with open(f"{path}/kde.pkl", "rb") as f:
            kde = pickle.load(f)

        instance.kde = kde

        shutil.rmtree(path)
        return instance
