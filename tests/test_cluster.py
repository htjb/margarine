"""Code to test the clustering functionality."""

import os

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from numpy.testing import assert_allclose

from margarine.estimators.clustered import cluster
from margarine.estimators.realnvp import RealNVP
from margarine.statistics import kldivergence, model_dimensionality


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

target_mean_one = jnp.array([-2.0, 0.0])
target_cov_one = jnp.array([[1.0, 0.0], [0.0, 1.0]])
target_mean_two = jnp.array([6.0, 0.0])
target_cov_two = jnp.array([[0.5, 0.0], [0.0, 1.0]])


original_samples = jnp.concatenate(
    [
        jax.random.multivariate_normal(
            key,
            mean=target_mean_one,
            cov=target_cov_one,
            shape=(nsamples // 2,),
        ),
        jax.random.multivariate_normal(
            key,
            mean=target_mean_two,
            cov=target_cov_two,
            shape=(nsamples // 2,),
        ),
    ],
    axis=0,
)


posterior_probs = jnp.concatenate(
    [
        stats.multivariate_normal.logpdf(
            original_samples[: nsamples // 2],
            mean=target_mean_one,
            cov=target_cov_one,
        ),
        stats.multivariate_normal.logpdf(
            original_samples[nsamples // 2 :],
            mean=target_mean_two,
            cov=target_cov_two,
        ),
    ],
    axis=0,
)

weights = jnp.ones(len(original_samples))

prior_probs = stats.uniform.logpdf(original_samples, loc=-6.5, scale=16.5)

prior_probs = jnp.sum(prior_probs, axis=-1)

logPpi = posterior_probs - prior_probs

samples_kl = D_KL(logPpi, weights)
samples_d = d_g(logPpi, weights)

bounds = jnp.array([[-6.5, -6.5], [10.0, 10.0]])
prior_samples = jax.random.uniform(
    key, (nsamples, 2), minval=bounds[0], maxval=bounds[1]
)


def test_clustering() -> None:
    """Test the clustered density estimator."""
    key = jax.random.PRNGKey(42)

    kld_estimates, bmd_estimates = [], []
    for _ in range(5):
        cluster_estimator = cluster(
            original_samples,
            base_estimator=RealNVP,
            weights=weights,
            in_size=2,
            hidden_size=50,
            num_layers=10,
            num_coupling_layers=4,
            clusters=2,
            theta_ranges=bounds,
        )

        key, subkey = jax.random.split(key)
        cluster_estimator.train(
            key=subkey,
            learning_rate=1e-4,
            epochs=2000,
            patience=50,
            batch_size=len(original_samples),
        )

        key, subkey = jax.random.split(key)
        samples = cluster_estimator.sample(subkey, 5000)

        prior_estimator = RealNVP(
            prior_samples,
            in_size=2,
            hidden_size=50,
            num_layers=2,
            num_coupling_layers=2,
            theta_ranges=bounds,
        )
        key, subkey = jax.random.split(key)
        prior_estimator.train(
            subkey,
            learning_rate=1e-4,
            epochs=2000,
            patience=50,
            batch_size=1024,
        )

        # check the kl divergence and model dimensionality
        kld = kldivergence(cluster_estimator, prior_estimator, samples)
        dim = model_dimensionality(cluster_estimator, prior_estimator, samples)
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

    cluster_estimator.save("test-cluster-save")
    loaded_estimator = cluster.load("test-cluster-save")
    key, subkey = jax.random.split(key)
    cluster_samples = cluster_estimator.sample(subkey, 5000)
    loaded_samples = loaded_estimator.sample(subkey, 5000)
    assert_allclose(cluster_samples, loaded_samples, rtol=1e-6, atol=1e-6)
    os.remove("test-cluster-save.clumarg")
