"""Tests for marginal statistics calculation using MAF and KDE methods."""

import numpy as np
import pytest
import tensorflow as tf
from anesthetic import MCMCSamples
from numpy.testing import assert_allclose, assert_equal
from scipy.stats import norm

from margarine.clustered import clusterMAF
from margarine.kde import KDE
from margarine.maf import MAF
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


norm = norm(loc=4.2, scale=0.3)
theta = norm.rvs(size=(1000, 2))
logL = norm.logpdf(theta).sum(axis=1)
weights = np.ones(len(theta))

mcmc_samples = MCMCSamples(data=theta, logL=logL)
samples_kl = D_KL(logL, weights)
samples_d = d_g(logL, weights)
names = [i for i in range(theta.shape[-1])]


def test_maf() -> None:
    """Test MAF marginal statistics calculation."""

    def check_stats(label: str) -> None:
        """Check calculated statistics against expected values."""
        if label == "KL Divergence":
            value = samples_kl
            assert_allclose(stats[label], value, rtol=1, atol=1)
        else:
            value = samples_d
            assert_allclose(stats[label], value, rtol=1, atol=1)

    bij = MAF(theta, weights=weights)
    bij.train(10000, early_stop=True, patience=20)

    stats_label = ["KL Divergence", "BMD"]

    stats = calculate(bij).statistics()
    [check_stats(stat_label) for stat_label in stats_label]


def test_maf_kwargs() -> None:
    """Test MAF with incorrect keyword arguments."""
    with pytest.raises(TypeError):
        bij = MAF(theta, weights=weights)
        bij.sample(4.5)
    with pytest.raises(TypeError):
        MAF(theta, weights=weights, number_networks=4.4)
    with pytest.raises(TypeError):
        MAF(theta, weights=weights, learning_rate="foobar")
    with pytest.raises(TypeError):
        MAF(theta, weights=weights, hidden_layers="foobar")
    with pytest.raises(TypeError):
        MAF(theta, weights=weights, hidden_layers=[4.5, 50])
    with pytest.raises(TypeError):
        MAF(theta, weights=weights)
        bij.train(epochs=4.5)
    with pytest.raises(TypeError):
        MAF(theta, weights=weights)
        bij.train(epochs=100, early_stop="foo")
    with pytest.raises(TypeError):
        MAF(theta, weights=weights)
        bij.train(epochs=100, clustering=5)
    with pytest.raises(TypeError):
        MAF(theta, weights=weights)
        bij.train(epochs=100, cluster_numeber="foo")
    with pytest.raises(TypeError):
        MAF(theta, weights=weights)
        bij.train(epochs=100, cluster_labels=5)


def test_maf_save_load() -> None:
    """Test the save and load functionality of MAF."""
    bij = MAF(theta, weights=weights)
    bij.train(100)
    file = "saved_maf.pkl"
    bij.save(file)
    loaded_bijector = MAF.load(file)
    for i in range(len(bij.mades)):
        assert_equal(
            bij.mades[i].get_weights(), loaded_bijector.mades[i].get_weights()
        )


def test_kde() -> None:
    """Test KDE marginal statistics calculation."""

    def check_stats(label: str) -> None:
        """Check calculated statistics against expected values."""
        if label == "KL Divergence":
            value = samples_kl
            assert_allclose(stats[label], value, rtol=1, atol=1)
        else:
            value = samples_d
            assert_allclose(stats[label], value, rtol=1, atol=1)

    kde = KDE(theta, weights=weights)
    kde.generate_kde()

    stats_label = ["KL Divergence", "BMD"]

    stats = calculate(kde).statistics()
    [check_stats(stat_label) for stat_label in stats_label]


def test_kde_save_load() -> None:
    """Test the save and load functionality of KDE."""
    kde = KDE(theta, weights=weights)
    kde.generate_kde()
    file = "saved_maf.pkl"
    kde.save(file)
    loaded_kde = KDE.load(file)
    assert_equal(kde.kde.covariance, loaded_kde.kde.covariance)


def test_anesthetic() -> None:
    """Test MAF and KDE with anesthetic MCMCSamples object."""
    kde = KDE(mcmc_samples, parameters=names)
    maf = MAF(mcmc_samples, parameters=names)
    cmaf = clusterMAF(mcmc_samples)

    assert_equal(kde.parameters, names)
    assert_equal(maf.parameters, names)

    # not providing parametes here but deriving them from the
    # anesthetic object columns
    assert np.all(cmaf.parameters == np.array(names))
