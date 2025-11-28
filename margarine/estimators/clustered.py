"""Module for clustered mixture of density estimators."""

import warnings

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax.scipy.special import logsumexp

from margarine.base.baseflow import BaseDensityEstimator
from margarine.utils.kmeans import kmeans, silhouette_score
from margarine.utils.utils import (
    approximate_bounds,
)

tfd = tfp.distributions


class cluster:
    """Create clustered mixture of MAFs to model multi-modal distributions."""

    def __init__(
        self,
        theta: jnp.ndarray,
        base_estimator: BaseDensityEstimator,
        weights: jnp.ndarray | None = None,
        theta_ranges: jnp.ndarray | None = None,
        clusters: jnp.ndarray | None = None,
        max_cluster_number: int = 10,
        **kwargs: dict,
    ) -> None:
        r"""Piecewise normalizing flow built from masked autoregressive flows.

        This class is a wrapper around the MAF class with additional clustering
        functionality. It trains, loads, and
        calls a piecewise density estimator where
        different base estimators are trained on
        different clusters of the sample space.

        Args:
            theta (jnp.ndarray): Samples to train the clustered MAF on.
            base_estimator (BaseDensityEstimator): The base density estimator
                to use for each cluster.
            weights (jnp.ndarray | None, optional): Weights for the samples.
                Defaults to None.
            theta_ranges (jnp.ndarray | None, optional): Ranges for the
                parameters in each cluster. Should have shape
                (nclusters, nparams, 2). Defaults to None.
            clusters (jnp.ndarray | None, optional): Predefined cluster
                labels for each sample. If None, k-means clustering is used.
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

        if self.clusters is None:
            ks = jnp.arange(2, max_cluster_number + 1)
            losses = []
            for k in ks:
                labels = kmeans(self.theta, k=k, num_iters=25)
                losses.append(-silhouette_score(self.theta, labels))
            losses = jnp.array(losses)
            minimum_index = jnp.argmin(losses)
            self.cluster_number = ks[minimum_index]

            self.clusters = kmeans(
                self.theta, k=self.cluster_number, num_iters=25
            )

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

        if self.theta_ranges is None:
            self.theta_ranges = []
            for i in range(self.cluster_number):
                self.theta_ranges.append(
                    approximate_bounds(
                        self.split_theta[i], self.split_weights[i]
                    )
                )

        self.estimators = []
        for i in range(len(split_theta)):
            self.estimators.append(
                base_estimator(
                    split_theta[i],
                    weights=split_weights[i],
                    theta_ranges=self.theta_ranges[i],
                    **self.kwargs,
                )
            )

    def train(self, **kwargs: dict) -> None:
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
        if prior_density is isinstance(prior_density, BaseDensityEstimator):
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
