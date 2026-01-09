"""KDE implementation using JAX."""

import os
import pickle
import shutil
from pathlib import Path
from zipfile import ZipFile

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import yaml
from tensorflow_probability.substrates import jax as tfp

from margarine import _version
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

        In detail, for a d-dimensional KDE, each output
        dimension is computed sequentially:

            x_1 = F_{X_1}^{-1}(u_1)
            x_2 = F_{X_2 | X_1}^{-1}(u_2 | x_1)
            ...
            x_d = F_{X_d | X_{1:d-1}}^{-1}(u_d | x_1, ..., x_{d-1})

        where \(u_i \sim \text{Uniform}[0,1]\), \(F^{-1}\)
        denotes the inverse CDF, and
        the conditional CDFs are computed
        from the Gaussian mixture KDE. Since these
        inverse CDFs do not have a closed form, we
        solve for each x_i using a
        root-finding algorithm (e.g., Newton-Raphson).

        Args:
            u (jnp.ndarray): Samples from the unit hypercube [0,1]^d
                Shape: (n_samples, n_dims)

        Returns:
            jnp.ndarray: The transformed samples following the
                KDE distribution.
                Shape: (n_samples, n_dims)
        """
        # generate useful parameters for __call__ function to transform
        # hypercube into samples on the KDE.
        S = self.kde.covariance
        mu = self.kde.dataset.T
        steps, s = [], []
        for i in range(mu.shape[-1]):
            if i == 0:
                step = jnp.array([])
                s_i = jnp.sqrt(S[0, 0])
            else:
                step = jnp.linalg.solve(S[:i, :i].T, S[i, :i]).T
                s_i = jnp.sqrt(S[i, i] - step @ S[:i, i])
            steps.append(step)  # conditional means
            s.append(s_i)  # conditional stddevs

        @jax.jit
        def newton_root(
            m: jnp.ndarray,
            s_i: float,
            target: float,
            init: float,
            weights: jnp.ndarray,
            maxiter: int = 20,
        ) -> float:
            """Solve for x in the conditional CDF equation with Newton-Raphson.

            Finds the scalar root of:

                sum_k w_k * Φ((x - m_k) / s_i) - target = 0

            where Φ is the standard normal CDF, m is the
            conditional mean vector
            for the current dimension, s_i is the
            conditional standard deviation,
            weights are the mixture weights of the KDE, and target ∈ [0,1] is
            the uniform sample to invert.

            Args:
                m (jnp.ndarray): Conditional means of
                    the mixture components, shape (n_kernels,).
                s_i (float): Conditional standard deviation for this dimension.
                target (float): Uniform sample in [0,1] to map to the KDE.
                init (float): Initial guess for the Newton-Raphson iteration.
                weights (jnp.ndarray): Mixture weights, shape (n_kernels,).
                maxiter (int, optional): Maximum number of iterations.
                    Default is 20.

            Returns:
                float: The root x such that the conditional CDF equals target.
            """

            def f(x: float) -> float:
                """Function for which we want to find the root."""
                return (
                    tfd.Normal(0, 1).cdf((x - m) / s_i) * weights
                ).sum() - target

            def f_prime(x: float) -> float:
                """Derivative of the function f."""
                pdf = tfd.Normal(0, 1).prob((x - m) / s_i) / s_i
                return (pdf * weights).sum()

            x = init
            for _ in range(maxiter):
                x = x - f(x) / f_prime(x)
            return x

        @jax.jit
        def transform_one(x: jnp.ndarray) -> jnp.ndarray:
            """Transform a single sample from unit hypercube to KDE."""
            y = jnp.zeros_like(x)
            for i in range(len(x)):
                m_i = (
                    mu[:, i] + steps[i] @ (y[:i] - mu[:, :i]).T
                    if i > 0
                    else mu[:, i]
                )
                init = mu[:, i].mean()
                y = y.at[i].set(
                    newton_root(m_i, s[i], x[i], init, self.kde.weights)
                )
            return inverse_transform(
                y, self.theta_ranges[0], self.theta_ranges[1]
            )

        transformed_samples = jax.vmap(transform_one)(u)
        return transformed_samples

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
                tfb.Scale(1 / (self.theta_ranges[1] - self.theta_ranges[0])),
                tfb.Shift(-self.theta_ranges[0]),
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

        os.makedirs(path)

        with open(f"{path}/kde.pkl", "wb") as f:
            pickle.dump(self.kde, f)

        config = {
            "bandwidth": self.bandwidth,
            "theta_ranges": self.theta_ranges,
        }

        with open(f"{path}/config.yaml", "w") as f:
            yaml.dump(config, f)

        metadata = {
            "theta": self.theta,
            "weights": self.weights,
            "margarine_version": _version.__version__,
        }

        with open(f"{path}/metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        with ZipFile(filename + ".marg", "w") as z:
            for subpath in path.rglob("*"):
                if subpath.is_file():
                    z.write(subpath, arcname=subpath.relative_to(path))

        shutil.rmtree(path)

    @classmethod
    def load(cls, filename: str) -> BaseDensityEstimator | None:
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

        with open(f"{path}/metadata.yaml") as f:
            metadata = yaml.unsafe_load(f)

        version = metadata.get("margarine_version", None)
        if version is None:
            print(
                "Warning: The KDE was saved with a version of margarine ",
                " < 2.0.0. In order to load it you will need to downgrade ",
                "margarine to a version < 2.0.0. e.g. ",
                "pip install margarine<2.0.0",
            )
            return
        if version != _version.__version__:
            print(
                f"Warning: The KDE was saved with margarine version "
                f"{version}, but you are loading it with version "
                f"{_version.__version__}. This may lead to "
                f"incompatibilities."
            )

        with open(f"{path}/config.yaml") as f:
            config = yaml.unsafe_load(f)

        instance = cls(
            theta=jnp.array([]),  # Placeholder, will be overwritten
            weights=jnp.array([]),  # Placeholder, will be overwritten
            bandwidth=config["bandwidth"],
            theta_ranges=config["theta_ranges"],
        )

        instance.theta = jnp.array(metadata["theta"])
        instance.weights = jnp.array(metadata["weights"])

        with open(f"{path}/kde.pkl", "rb") as f:
            kde = pickle.load(f)

        instance.kde = kde

        shutil.rmtree(path)
        return instance
