"""Code to test the clustering functionality."""

import os

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from margarine.estimators.clustered import cluster
from margarine.estimators.realnvp import RealNVP
from margarine.statistics import kldivergence, model_dimensionality


def analytic_kl(
    covs: jnp.ndarray, mixture_weights: jnp.ndarray, bounds: jnp.ndarray
) -> jnp.ndarray:
    """KL divergence between a Gaussian mixture and a uniform prior.

    Assumes the components are well separated and that the prior box
    contains essentially all of the mixture's mass.

    Args:
        covs (jnp.ndarray): Covariances of the components
        mixture_weights (jnp.ndarray): Weights of the components
        bounds (jnp.ndarray): Lower and upper bounds of the uniform prior

    Returns:
        jnp.ndarray: Kullback-Leibler divergence
    """
    d = covs.shape[-1]
    log_volume = jnp.sum(jnp.log(bounds[1] - bounds[0]))
    component_entropies = 0.5 * (
        d * jnp.log(2 * jnp.pi * jnp.e) + jnp.linalg.slogdet(covs)[1]
    )
    entropy = jnp.sum(
        mixture_weights * (component_entropies - jnp.log(mixture_weights))
    )
    return log_volume - entropy


def analytic_bmd(
    covs: jnp.ndarray, mixture_weights: jnp.ndarray
) -> jnp.ndarray:
    """BMD between a Gaussian mixture and a uniform prior.

    Assumes the components are well separated. Within component k,
    log P = log w_k - log|Sigma_k| / 2 - chi^2_d / 2 + const, so
    2 * Var[log P] = d + 2 * Var_k[log w_k - log|Sigma_k| / 2].

    Args:
        covs (jnp.ndarray): Covariances of the components
        mixture_weights (jnp.ndarray): Weights of the components

    Returns:
        jnp.ndarray: BMD statistic
    """
    d = covs.shape[-1]
    offsets = jnp.log(mixture_weights) - 0.5 * jnp.linalg.slogdet(covs)[1]
    mean_offset = jnp.sum(mixture_weights * offsets)
    offset_var = jnp.sum(mixture_weights * (offsets - mean_offset) ** 2)
    return d + 2 * offset_var


nsamples = 5000
key = jax.random.PRNGKey(0)

target_mean_one = jnp.array([-2.0, 0.0])
target_cov_one = jnp.array([[1.0, 0.0], [0.0, 1.0]])
target_mean_two = jnp.array([6.0, 0.0])
target_cov_two = jnp.array([[0.5, 0.0], [0.0, 1.0]])
bounds = jnp.array([[-6.5, -6.5], [10.0, 10.0]])

key, key_one, key_two = jax.random.split(key, 3)
original_samples = jnp.concatenate(
    [
        jax.random.multivariate_normal(
            key_one,
            mean=target_mean_one,
            cov=target_cov_one,
            shape=(nsamples // 2,),
        ),
        jax.random.multivariate_normal(
            key_two,
            mean=target_mean_two,
            cov=target_cov_two,
            shape=(nsamples // 2,),
        ),
    ],
    axis=0,
)
weights = jnp.ones(len(original_samples))

key, subkey = jax.random.split(key)
prior_samples = jax.random.uniform(
    subkey, (nsamples, 2), minval=bounds[0], maxval=bounds[1]
)

target_covs = jnp.stack([target_cov_one, target_cov_two])
mixture_weights = jnp.array([0.5, 0.5])
true_kl = analytic_kl(target_covs, mixture_weights, bounds)
true_bmd = analytic_bmd(target_covs, mixture_weights)

# fixed tolerances on the analytic values, calibrated over several seeds
kl_atol = 0.15
bmd_atol = 0.3


def test_clustering() -> None:
    """Test the clustered density estimator."""
    key = jax.random.PRNGKey(42)

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
    assert_allclose(kld, true_kl, rtol=0, atol=kl_atol)
    assert_allclose(dim, true_bmd, rtol=0, atol=bmd_atol)

    cluster_estimator.save("test-cluster-save")
    loaded_estimator = cluster.load("test-cluster-save")
    key, subkey = jax.random.split(key)
    cluster_samples = cluster_estimator.sample(subkey, 5000)
    loaded_samples = loaded_estimator.sample(subkey, 5000)
    assert_allclose(cluster_samples, loaded_samples, rtol=1e-6, atol=1e-6)
    os.remove("test-cluster-save.clumarg")
