"""Tests for clustered MAF flows and their marginal statistics."""

import numpy as np
import pytest
import tensorflow as tf
from anesthetic import MCMCSamples
from clustered_distributions import TwoMoons
from numpy.testing import assert_allclose, assert_equal

from margarine.clustered import clusterMAF
from margarine.marginal_stats import calculate


def D_KL(
    logL: np.ndarray | tf.Tensor, weights: np.ndarray | tf.Tensor
) -> float:
    """Calculate the Kullback-Leibler divergence."""
    return -np.average(logL, weights=weights)


def d_g(
    logL: np.ndarray | tf.Tensor, weights: np.ndarray | tf.Tensor
) -> float:
    """Calculate the BMD statistic."""
    return 2 * np.cov(logL, aweights=weights)


nsamples = 2500

tm = TwoMoons()

samples = tm.sample(nsamples)
theta = samples.numpy()
weights = np.ones(len(theta))

logL = tm.log_prob(samples).numpy()

samples = MCMCSamples(data=theta)
samples_kl = D_KL(logL, weights)
samples_d = d_g(logL, weights)
names = [i for i in range(theta.shape[-1])]


def test_maf_clustering() -> None:
    """Test clustered MAF marginal statistics calculation."""
    bij = clusterMAF(theta, weights=weights)
    bij.train(10000, early_stop=True, patience=400)
    file = "saved_maf_cluster.pkl"
    bij.save(file)

    loaded_bijector = clusterMAF.load("saved_maf_cluster.pkl")
    for f in range(len(bij.flow)):
        for i in range(len(bij.flow[f].mades)):
            assert_equal(
                bij.flow[f].mades[i].get_weights(),
                loaded_bijector.flow[f].mades[i].get_weights(),
            )

    def check_stats(label: str) -> None:
        """Check calculated statistics against expected values."""
        if label == "KL Divergence":
            value = samples_kl
            assert_allclose(stats[label], value, rtol=1, atol=1)
        else:
            value = samples_d
            assert_allclose(stats[label], value, rtol=1, atol=1)

    stats_label = ["KL Divergence", "BMD"]

    stats = calculate(bij).statistics()
    [check_stats(stat_label) for stat_label in stats_label]


def test_maf_cluster_kwargs() -> None:
    """Test clustered MAF with incorrect keyword arguments."""
    with pytest.raises(ValueError):
        clusterMAF(theta, weights=weights, cluster_number=3)
    with pytest.raises(ValueError):
        clusterMAF(theta, weightes=weights, cluster_labels=np.ones(len(theta)))

    labels = np.ones(len(theta))
    labels[: len(theta) // 2] = 0


def test_cluster_size() -> None:
    """Test clustered MAF with small cluster sizes."""

    # testing the while loop for cluster size
    def draw(ndims: int) -> np.ndarray:
        """Draw samples from a multivariate normal distribution."""
        return np.random.multivariate_normal(
            np.zeros(ndims), np.eye(ndims), ndims * 10
        )

    samples = draw(3)
    clusterMAF(samples)
