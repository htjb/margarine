"""Test the various estimators in margarine."""

import jax
import jax.numpy as jnp
from clustered_distributions import TwoMoons
from numpy.testing import assert_allclose

from margarine.estimators.kde import KDE
from margarine.estimators.nice import NICE
from margarine.estimators.realnvp import RealNVP
from margarine.statistics import kldivergence, model_dimensionality
from margarine.utils.utils import approximate_bounds


def D_KL(logL: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Calculate the Kullback-Leibler divergence.

    Args:
        logL (jnp.ndarray): Log-likelihood values
        weights (jnp.ndarray): Corresponding weights

    Returns:
        jnp.ndarray: Kullback-Leibler divergence
    """
    return -jnp.average(logL, weights=weights)


def d_g(logL: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Calculate the BMD statistic.

    Args:
        logL (jnp.ndarray): Log-likelihood values
        weights (jnp.ndarray): Corresponding weights

    Returns:
        jnp.ndarray: BMD statistic
    """
    return 2 * jnp.cov(logL, aweights=weights)


nsamples = 2500
key = jax.random.PRNGKey(0)

tm = TwoMoons()

key, subkey = jax.random.split(key)
original_samples = tm.sample(subkey, nsamples)
weights = jnp.ones(len(original_samples))

logL = tm.log_prob(original_samples)

samples_kl = D_KL(logL, weights)
samples_d = d_g(logL, weights)

bounds = approximate_bounds(original_samples, weights)
prior_samples = jax.random.uniform(
    key, (10000, 2), minval=bounds[0], maxval=bounds[1]
)


def test_nice() -> None:
    """Test NICE estimator."""
    key = jax.random.PRNGKey(42)

    nice_estimator = NICE(
        original_samples,
        weights=weights,
        in_size=2,
        hidden_size=128,
        num_layers=2,
        num_coupling_layers=4,
    )

    # check the forward and inverse transformations
    key, subkey = jax.random.split(key)
    z = jax.random.normal(key, (1000, 2))
    forward_transformed = nice_estimator.forward(z)
    inverse_transformed = nice_estimator.inverse(forward_transformed)
    error = jnp.mean(jnp.abs(inverse_transformed - z))
    assert error < 1e-3

    key, subkey = jax.random.split(key)
    nice_estimator.train(subkey, learning_rate=1e-3, epochs=1000, patience=50)

    key, subkey = jax.random.split(key)
    samples = nice_estimator.sample(subkey, 10000)

    prior_estimator = NICE(
        prior_samples,
        in_size=2,
        hidden_size=128,
        num_layers=2,
        num_coupling_layers=4,
    )
    prior_estimator.train(subkey, learning_rate=1e-3, epochs=1000, patience=50)

    # check the kl divergence and model dimensionality
    kld = kldivergence(nice_estimator, prior_estimator, samples)
    dim = model_dimensionality(nice_estimator, prior_estimator, samples)

    assert_allclose(kld, samples_kl, rtol=1, atol=1)
    assert_allclose(dim, samples_d, rtol=1, atol=1)


def test_realnvp() -> None:
    """Test RealNVP estimator."""
    key = jax.random.PRNGKey(43)

    realnvp_estimator = RealNVP(
        original_samples,
        weights=weights,
        in_size=2,
        hidden_size=128,
        num_coupling_layers=6,
    )

    # check the forward and inverse transformations
    key, subkey = jax.random.split(key)
    z = jax.random.normal(key, (1000, 2))

    forward_transformed = realnvp_estimator.forward(z)
    inverse_transformed = realnvp_estimator.inverse(forward_transformed)
    error = jnp.mean(jnp.abs(inverse_transformed - z))
    assert error < 1e-3

    key, subkey = jax.random.split(key)
    realnvp_estimator.train(
        subkey, learning_rate=1e-3, epochs=1000, patience=50
    )

    key, subkey = jax.random.split(key)
    samples = realnvp_estimator.sample(subkey, 10000)

    prior_estimator = RealNVP(
        prior_samples, in_size=2, hidden_size=128, num_coupling_layers=6
    )
    prior_estimator.train(subkey, learning_rate=1e-3, epochs=1000, patience=50)
    # check the kl divergence and model dimensionality
    kld = kldivergence(realnvp_estimator, prior_estimator, samples)
    dim = model_dimensionality(realnvp_estimator, prior_estimator, samples)
    assert_allclose(kld, samples_kl, rtol=1, atol=1)
    assert_allclose(dim, samples_d, rtol=1, atol=1)


def test_kde() -> None:
    """Test KDE estimator."""
    key = jax.random.PRNGKey(44)
    kde_estimator = KDE(original_samples, weights=weights, bandwidth=0.2)
    kde_estimator.train()
    key, subkey = jax.random.split(key)
    samples = kde_estimator.sample(subkey, 10000)

    prior_estimator = KDE(prior_samples, bandwidth=0.2)
    prior_estimator.train()

    # check the kl divergence and model dimensionality
    kld = kldivergence(kde_estimator, prior_estimator, samples)
    dim = model_dimensionality(kde_estimator, prior_estimator, samples)
    assert_allclose(kld, samples_kl, rtol=1, atol=1)
    assert_allclose(dim, samples_d, rtol=1, atol=1)
