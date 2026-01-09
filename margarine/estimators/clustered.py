"""Module for clustered mixture of density estimators."""

import os
import shutil
import warnings
from pathlib import Path
from zipfile import ZipFile

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
import yaml
from jax.scipy.special import logsumexp

from margarine.estimators.kde import KDE
from margarine.estimators.nice import NICE
from margarine.estimators.realnvp import RealNVP
from margarine.utils.kmeans import kmeans, silhouette_score
from margarine.utils.utils import approximate_bounds

tfd = tfp.distributions


class cluster:
    """Create clustered mixture of MAFs to model multi-modal distributions."""

    def __init__(
        self,
        theta: jnp.ndarray,
        base_estimator: NICE | KDE | RealNVP,
        weights: jnp.ndarray | None = None,
        theta_ranges: jnp.ndarray | None = None,
        clusters: jnp.ndarray | None | int = None,
        max_cluster_number: int = 10,
        **kwargs: object,
    ) -> None:
        r"""Piecewise normalizing flow built from masked autoregressive flows.

        This class is a wrapper around the MAF class with additional clustering
        functionality. It trains, loads, and
        calls a piecewise density estimator where
        different base estimators are trained on
        different clusters of the sample space.

        Args:
            theta (jnp.ndarray): Samples to train the clustered MAF on.
            base_estimator (NICE | KDE | RealNVP): The base density estimator
                to use for each cluster.
            weights (jnp.ndarray | None, optional): Weights for the samples.
                Defaults to None.
            theta_ranges (jnp.ndarray | None, optional): Ranges for the
                parameters. Should have shape
                (nparams, 2). Defaults to None.
            clusters (jnp.ndarray | None | int, optional): Predefined cluster
                labels for each sample or an integer
                corresponding to the number of expected clusters.
                If None, k-means clustering is used.
                Defaults to None.
            max_cluster_number (int, optional): Maximum number of clusters
                to consider when using k-means clustering. Defaults to 10.
            **kwargs: Additional keyword arguments for the base estimator.
        """
        self.clusters = clusters
        self.base_estimator = base_estimator
        self.kwargs = kwargs

        self.theta = theta
        self.weights = weights
        self.theta_ranges = theta_ranges

        if self.weights is None:
            self.weights = jnp.ones(len(self.theta))

        if self.theta_ranges is None:
            self.theta_ranges = approximate_bounds(self.theta, self.weights)

        if self.clusters is None:
            ks = jnp.arange(2, max_cluster_number + 1)
            losses = []
            for k in ks:
                labels = kmeans(
                    self.theta,
                    k=k,
                )
                losses.append(-silhouette_score(self.theta, labels))
            losses = jnp.array(losses)
            minimum_index = jnp.argmin(losses)
            self.cluster_number = ks[minimum_index]

            self.clusters = kmeans(
                self.theta,
                k=self.cluster_number,
            )
        elif isinstance(self.clusters, int):
            self.cluster_number = self.clusters
            self.clusters = kmeans(
                self.theta,
                k=self.cluster_number,
            )
        else:
            self.cluster_number = len(jnp.unique(self.clusters))

        # count the number of times a cluster label appears in cluster_labels
        self.cluster_count = jnp.bincount(self.clusters)

        # While loop to make sure clusters are not too small
        while self.cluster_count.min() < 100:
            warnings.warn(
                "One or more clusters are too small "
                + "(n_cluster < 100). "
                + "Reducing the number of clusters by 1."
            )
            minimum_index -= 1
            self.cluster_number = ks[minimum_index]
            self.clusters = kmeans(
                self.theta, k=self.cluster_number, num_iters=25
            )
            self.cluster_count = jnp.bincount(self.clusters)
            if self.cluster_number == 2:
                # break if two clusters
                warnings.warn(
                    "The number of clusters is 2. This is the "
                    + "minimum number of clusters that can be used. "
                    + "Some clusters may be too small and the "
                    + "train/test split may fail."
                    + "Try running without clusting. "
                )
                break

        split_theta = []
        split_weights = []
        for i in range(self.cluster_number):
            split_theta.append(self.theta[self.clusters == i])
            split_weights.append(self.weights[self.clusters == i])

        self.split_theta = split_theta
        self.split_weights = split_weights

        self.estimators = []
        for i in range(len(split_theta)):
            self.estimators.append(
                base_estimator(
                    split_theta[i],
                    weights=split_weights[i],
                    theta_ranges=self.theta_ranges,
                    **self.kwargs,
                )
            )

    def train(self, **kwargs: object) -> None:
        r"""Train the cluster estimator.

        Args:
            **kwargs: Additional keyword arguments for training.
        """
        for i in range(len(self.estimators)):
            self.estimators[i].train(**kwargs)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Log-probability for a given set of parameters.

        While each density estimator has its own built in log probability
        function, a correction has to be applied for the transformation of
        variables that is used to improve accuracy when learning and we
        have to sum probabilities over the series of flows. The
        correction and the sum are implemented here.

        Args:
            x (jnp.ndarray): The set of samples for which to
                calculate the log probability.

        Returns:
            jnp.ndarray: The log-probabilities of the provided samples.
        """
        estimator_importance = jnp.array(
            [jnp.sum(weights) for weights in self.split_weights]
        )
        estimator_importance = estimator_importance / jnp.sum(
            estimator_importance
        )

        logprob = []
        for estimator, importance in zip(
            self.estimators, estimator_importance
        ):
            prob = estimator.log_prob(x)
            probs = prob + jnp.log(importance)
            logprob.append(probs)
        logprob = jnp.array(logprob)
        logprob = logsumexp(logprob, axis=0)

        return logprob

    def log_like(
        self,
        x: jnp.ndarray,
        logevidence: float,
        prior_density: jnp.ndarray | NICE | KDE | RealNVP,
    ) -> jnp.ndarray:
        """Compute the marginal log-likelihood of given samples.

        Args:
            x: Samples for which to compute the log-likelihood.
            logevidence: Log-evidence term.
            prior_density: Prior density or density estimator.

        Returns:
            jnp.ndarray: Log-likelihoods of the samples.
        """
        if isinstance(prior_density, NICE | KDE | RealNVP):
            prior_density = prior_density.log_prob(x)

        return self.log_prob(x) + logevidence - prior_density

    def __call__(self, key: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        r"""Transform samples from the unit hypercube to the cluster.

        This function is used when calling the cluster class to transform
        samples from the unit hypercube to samples on the
        clustered distribution.

        Args:
            key (jnp.ndarray): JAX random key for sampling.
            u (jnp.ndarray): Samples on the unit hypercube.

        Returns:
           jnp.ndarray: Samples transformed to the clusterMAF.
        """
        estimator_importance = jnp.array(
            [jnp.sum(weights) for weights in self.split_weights]
        )
        probabilities = estimator_importance / jnp.sum(estimator_importance)
        options = jnp.arange(0, self.cluster_number)

        choice = jax.random.choice(
            key, options, p=probabilities, shape=(len(u),)
        )

        totals = jnp.array(
            [len(choice[choice == options[i]]) for i in range(len(options))]
        )
        totals = jnp.hstack([0, jnp.cumsum(totals)])

        values = []
        for i in range(len(options)):
            x = self.estimators[i](u[totals[i] : totals[i + 1]])
            values.append(x)

        return jnp.vstack(values)

    def sample(self, key: jnp.ndarray, num_samples: int = 1000) -> jnp.ndarray:
        r"""Generate samples on the cluster.

        Args:
            key (jnp.ndarray): JAX random key for sampling.
            num_samples (int, optional): The number of samples to generate.
                Defaults to 1000.

        Returns:
            jnp.ndarray: Samples generated on the clusterMAF.
        """
        estimator_importance = jnp.array(
            [jnp.sum(weights) for weights in self.split_weights]
        )
        probabilities = estimator_importance / jnp.sum(estimator_importance)
        options = jnp.arange(0, self.cluster_number)

        choice = jax.random.choice(
            key, options, p=probabilities, shape=(num_samples,)
        )

        totals = jnp.array(
            [len(choice[choice == options[i]]) for i in range(len(options))]
        )

        values = []
        for i in range(len(options)):
            key, subkey = jax.random.split(key)
            x = self.estimators[i].sample(subkey, totals[i])
            values.append(x)

        return jnp.vstack(values)

    def save(self, filename: str) -> None:
        """Save the clustered estimator to a file.

        Args:
            filename (str): The name of the file to save the estimator to.
        """
        path = Path(filename).resolve()
        if path.exists():
            shutil.rmtree(path)

        os.makedirs(path)

        config = {
            "base_estimator": self.base_estimator.__name__,
            "kwargs": self.kwargs,
            "theta_ranges": self.theta_ranges,
            "clusters": self.clusters,
        }

        with open(path / "config.yaml", "w") as f:
            yaml.dump(config, f)

        for i, estimator in enumerate(self.estimators):
            estimator.save(str(path) + f"/estimator_{i}")

        metadata = {
            "theta": self.theta,
            "weights": self.weights,
        }

        with open(path / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        with ZipFile(filename + ".clumarg", "w") as z:
            for subpath in path.rglob("*"):
                if subpath.is_file():
                    z.write(subpath, arcname=subpath.relative_to(path))

        shutil.rmtree(path)

    @classmethod
    def load(cls, filename: str) -> "cluster":
        """Load a clustered estimator from a file.

        Args:
            filename (str): The name of the file to load the estimator from.

        Returns:
            cluster: The loaded clustered estimator.
        """
        zip_path = Path(f"{filename}.clumarg")
        path = Path(filename + ".tmp").resolve()
        with ZipFile(zip_path) as z:
            # Extract all files to a folder
            z.extractall(path)

        with open(path / "config.yaml") as f:
            config = yaml.unsafe_load(f)

        with open(path / "metadata.yaml") as f:
            metadata = yaml.unsafe_load(f)

        base_estimator_class = globals()[config["base_estimator"]]

        instance = cls(
            theta=metadata["theta"],  # Placeholder, will be overwritten
            base_estimator=base_estimator_class,
            weights=metadata["weights"],  # Placeholder, will be overwritten
            theta_ranges=config["theta_ranges"],
            clusters=config["clusters"],
            **config["kwargs"],
        )

        instance.estimators = []
        for i in range(instance.cluster_number):
            estimator = base_estimator_class.load(
                str(path) + f"/estimator_{i}"
            )
            instance.estimators.append(estimator)

        shutil.rmtree(path)

        return instance
