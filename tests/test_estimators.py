"""Test the various estimators in margarine."""

import os

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from numpy.testing import assert_allclose

from margarine.estimators.kde import KDE
from margarine.estimators.nice import NICE
from margarine.estimators.realnvp import RealNVP
from margarine.statistics import kldivergence, model_dimensionality
from margarine.utils.utils import approximate_bounds


def D_KL(logPpi: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Calculate the Kullback-Leibler divergence.

    Args:
        logPpi (jnp.ndarray): Log-likelihood values
        weights (jnp.ndarray): Corresponding weights

    Returns:
        jnp.ndarray: Kullback-Leibler divergence
    """
    return jnp.average(logPpi, weights=weights)


def d_g(logPpi: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Calculate the BMD statistic.

    Args:
        logPpi (jnp.ndarray): Log-likelihood values
        weights (jnp.ndarray): Corresponding weights

    Returns:
        jnp.ndarray: BMD statistic
    """
    return 2 * jnp.cov(logPpi, aweights=weights)


nsamples = 5000
key = jax.random.PRNGKey(0)

original_samples = jax.random.multivariate_normal(
    key,
    mean=jnp.array([0.0, 0.0]),
    cov=jnp.array([[1.0, 0.8], [0.8, 1.0]]),
    shape=(nsamples,),
)
posterior_probs = stats.multivariate_normal.logpdf(
    original_samples,
    mean=jnp.array([0.0, 0.0]),
    cov=jnp.array([[1.0, 0.8], [0.8, 1.0]]),
)
weights = jnp.ones(len(original_samples))

prior_probs = stats.uniform.logpdf(original_samples, loc=-4.0, scale=8.0)
prior_probs = jnp.sum(prior_probs, axis=-1)

logPpi = posterior_probs - prior_probs


samples_kl = D_KL(logPpi, weights)
samples_d = d_g(logPpi, weights)

bounds = approximate_bounds(original_samples, weights)
prior_samples = jax.random.uniform(
    key, (nsamples, 2), minval=bounds[0], maxval=bounds[1]
)


def test_nice() -> None:
    """Test NICE estimator."""
    key = jax.random.PRNGKey(42)

    nice_estimator = NICE(
        original_samples,
        weights=weights,
        in_size=2,
        hidden_size=50,
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

    kld_estimates, bmd_estimates = [], []
    for _ in range(5):
        nice_estimator = NICE(
            original_samples,
            weights=weights,
            in_size=2,
            hidden_size=50,
            num_layers=2,
            num_coupling_layers=6,
            theta_ranges=bounds,
        )

        key, subkey = jax.random.split(key)
        nice_estimator.train(
            subkey,
            learning_rate=1e-3,
            epochs=2000,
            patience=50,
            batch_size=1024,
        )

        key, subkey = jax.random.split(key)
        samples = nice_estimator.sample(subkey, 5000)

        prior_estimator = NICE(
            prior_samples,
            in_size=2,
            hidden_size=50,
            num_layers=2,
            num_coupling_layers=6,
            theta_ranges=bounds,
        )
        key, subkey = jax.random.split(key)
        prior_estimator.train(
            subkey,
            learning_rate=1e-3,
            epochs=2000,
            patience=50,
            batch_size=1024,
        )

        # check the kl divergence and model dimensionality
        kld = kldivergence(nice_estimator, prior_estimator, samples)
        dim = model_dimensionality(nice_estimator, prior_estimator, samples)
        kld_estimates.append(kld)
        bmd_estimates.append(dim)

    kld_estimates = jnp.array(kld_estimates)
    bmd_estimates = jnp.array(bmd_estimates)

    kl_rtol = 3 * jnp.std(kld_estimates) / (jnp.mean(kld_estimates) + 1e-10)
    kl_atol = 3 * jnp.std(kld_estimates)
    kld = jnp.mean(kld_estimates)

    dim_rtol = 3 * jnp.std(bmd_estimates) / (jnp.mean(bmd_estimates) + 1e-10)
    dim_atol = 3 * jnp.std(bmd_estimates)
    dim = jnp.mean(bmd_estimates)

    assert_allclose(kld, samples_kl, rtol=kl_rtol, atol=kl_atol)
    assert_allclose(dim, samples_d, rtol=dim_rtol, atol=dim_atol)

    nice_estimator.save("nice_test")
    loaded_estimator = NICE.load("nice_test")
    key, subkey = jax.random.split(key)
    samples = nice_estimator.sample(subkey, 1000)
    loaded_samples = loaded_estimator.sample(subkey, 1000)
    assert_allclose(samples, loaded_samples, rtol=1e-6, atol=1e-6)
    os.remove("nice_test.marg")


def test_realnvp() -> None:
    """Test RealNVP estimator."""
    key = jax.random.PRNGKey(43)

    realnvp_estimator = RealNVP(
        original_samples,
        weights=weights,
        in_size=2,
        hidden_size=50,
        num_coupling_layers=2,
    )

    # check the forward and inverse transformations
    key, subkey = jax.random.split(key)
    z = jax.random.normal(key, (1000, 2))

    forward_transformed = realnvp_estimator.forward(z)
    inverse_transformed = realnvp_estimator.inverse(forward_transformed)
    error = jnp.mean(jnp.abs(inverse_transformed - z))
    assert error < 1e-3

    kl_estimates, bmd_estimates = [], []
    for _ in range(5):
        realnvp_estimator = RealNVP(
            original_samples,
            weights=weights,
            in_size=2,
            hidden_size=50,
            num_coupling_layers=2,
        )
        key, subkey = jax.random.split(key)
        realnvp_estimator.train(
            subkey,
            learning_rate=1e-4,
            epochs=2000,
            patience=50,
            batch_size=len(original_samples),
        )

        key, subkey = jax.random.split(key)
        samples = realnvp_estimator.sample(subkey, 5000)

        prior_estimator = RealNVP(
            prior_samples, in_size=2, hidden_size=50, num_coupling_layers=2
        )
        key, subkey = jax.random.split(key)
        prior_estimator.train(
            subkey,
            learning_rate=1e-4,
            epochs=2000,
            patience=50,
            batch_size=len(prior_samples),
        )

        # check the kl divergence and model dimensionality
        kld = kldivergence(realnvp_estimator, prior_estimator, samples)
        dim = model_dimensionality(realnvp_estimator, prior_estimator, samples)
        kl_estimates.append(kld)
        bmd_estimates.append(dim)

    kl_estimates = jnp.array(kl_estimates)
    bmd_estimates = jnp.array(bmd_estimates)

    kl_rtol = 3 * jnp.std(kl_estimates) / (jnp.mean(kl_estimates) + 1e-10)
    kl_atol = 3 * jnp.std(kl_estimates)
    kld = jnp.mean(kl_estimates)

    dim_rtol = 3 * jnp.std(bmd_estimates) / (jnp.mean(bmd_estimates) + 1e-10)
    dim_atol = 3 * jnp.std(bmd_estimates)
    dim = jnp.mean(bmd_estimates)
    assert_allclose(kld, samples_kl, rtol=kl_rtol, atol=kl_atol)
    assert_allclose(dim, samples_d, rtol=dim_rtol, atol=dim_atol)

    realnvp_estimator.save("realnvp_test")
    loaded_estimator = RealNVP.load("realnvp_test")
    key, subkey = jax.random.split(key)
    samples = realnvp_estimator.sample(subkey, 1000)
    loaded_samples = loaded_estimator.sample(subkey, 1000)
    assert_allclose(samples, loaded_samples, rtol=1e-6, atol=1e-6)
    os.remove("realnvp_test.marg")


def test_kde() -> None:
    """Test KDE estimator."""
    key = jax.random.PRNGKey(44)
    kl_estimates, bmd_estimates = [], []
    for _ in range(5):
        kde_estimator = KDE(
            original_samples,
            weights=weights,
            theta_ranges=bounds,
            bandwidth=0.08,
        )
        kde_estimator.train()
        key, subkey = jax.random.split(key)
        samples = kde_estimator.sample(subkey, 5000)

        prior_estimator = KDE(
            prior_samples, theta_ranges=bounds, bandwidth=0.08
        )
        prior_estimator.train()

        kld = kldivergence(kde_estimator, prior_estimator, samples)
        dim = model_dimensionality(kde_estimator, prior_estimator, samples)
        kl_estimates.append(kld)
        bmd_estimates.append(dim)

    kl_estimates = jnp.array(kl_estimates)
    bmd_estimates = jnp.array(bmd_estimates)

    kl_rtol = 3 * jnp.std(kl_estimates) / (jnp.mean(kl_estimates) + 1e-10)
    kl_atol = 3 * jnp.std(kl_estimates)
    kld = jnp.mean(kl_estimates)

    dim_rtol = 3 * jnp.std(bmd_estimates) / (jnp.mean(bmd_estimates) + 1e-10)
    dim_atol = 3 * jnp.std(bmd_estimates)
    dim = jnp.mean(bmd_estimates)

    assert_allclose(kld, samples_kl, rtol=kl_rtol, atol=kl_atol)
    assert_allclose(dim, samples_d, rtol=dim_rtol, atol=dim_atol)

    kde_estimator.save("kde_test")
    loaded_estimator = KDE.load("kde_test")
    key, subkey = jax.random.split(key)
    samples = kde_estimator.sample(subkey, 1000)
    loaded_samples = loaded_estimator.sample(subkey, 1000)
    assert_allclose(samples, loaded_samples, rtol=1e-6, atol=1e-6)
    os.remove("kde_test.marg")
