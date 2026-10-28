"""Test the various estimators in margarine."""

import os

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from margarine.estimators.kde import KDE
from margarine.estimators.nice import NICE
from margarine.estimators.realnvp import RealNVP
from margarine.statistics import kldivergence, model_dimensionality


def analytic_kl(cov: jnp.ndarray, bounds: jnp.ndarray) -> jnp.ndarray:
    """KL divergence between a Gaussian and a uniform prior.

    Assumes the prior box contains essentially all of the Gaussian's mass.

    Args:
        cov (jnp.ndarray): Covariance of the Gaussian
        bounds (jnp.ndarray): Lower and upper bounds of the uniform prior

    Returns:
        jnp.ndarray: Kullback-Leibler divergence
    """
    d = cov.shape[0]
    log_volume = jnp.sum(jnp.log(bounds[1] - bounds[0]))
    entropy = 0.5 * (
        d * jnp.log(2 * jnp.pi * jnp.e) + jnp.linalg.slogdet(cov)[1]
    )
    return log_volume - entropy


def analytic_bmd(cov: jnp.ndarray) -> jnp.ndarray:
    """BMD between a Gaussian and a uniform prior.

    log P = const - chi^2_d / 2, so 2 * Var[log P] = d.

    Args:
        cov (jnp.ndarray): Covariance of the Gaussian

    Returns:
        jnp.ndarray: BMD statistic
    """
    return jnp.asarray(cov.shape[0], dtype=float)


nsamples = 5000
key = jax.random.PRNGKey(0)

target_mean = jnp.array([0.0, 0.0])
target_cov = jnp.array([[1.0, 0.8], [0.8, 1.0]])
bounds = jnp.array([[-5.0, -5.0], [5.0, 5.0]])

key, subkey = jax.random.split(key)
original_samples = jax.random.multivariate_normal(
    subkey,
    mean=target_mean,
    cov=target_cov,
    shape=(nsamples,),
)
weights = jnp.ones(len(original_samples))

key, subkey = jax.random.split(key)
prior_samples = jax.random.uniform(
    subkey, (nsamples, 2), minval=bounds[0], maxval=bounds[1]
)

true_kl = analytic_kl(target_cov, bounds)
true_bmd = analytic_bmd(target_cov)

# fixed tolerances on the analytic values, calibrated over several seeds
kl_atol = 0.15
bmd_atol = 0.3
# NICE and KDE consistently underestimate the BMD by ~0.15-0.35
smoothing_bmd_atol = 0.45


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
    assert_allclose(kld, true_kl, rtol=0, atol=kl_atol)
    assert_allclose(dim, true_bmd, rtol=0, atol=smoothing_bmd_atol)

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

    realnvp_estimator = RealNVP(
        original_samples,
        weights=weights,
        in_size=2,
        hidden_size=50,
        num_coupling_layers=2,
        theta_ranges=bounds,
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
        prior_samples,
        in_size=2,
        hidden_size=50,
        num_coupling_layers=2,
        theta_ranges=bounds,
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
    assert_allclose(kld, true_kl, rtol=0, atol=kl_atol)
    assert_allclose(dim, true_bmd, rtol=0, atol=bmd_atol)

    realnvp_estimator.save("realnvp_test")
    loaded_estimator = RealNVP.load("realnvp_test")
    key, subkey = jax.random.split(key)
    samples = realnvp_estimator.sample(subkey, 1000)
    loaded_samples = loaded_estimator.sample(subkey, 1000)
    assert_allclose(samples, loaded_samples, rtol=1e-6, atol=1e-6)
    os.remove("realnvp_test.marg")

def test_nice_mlps() -> None:
    """Test to check that the MLPs in a NICE instance are different."""
    theta = jax.random.normal(jax.random.PRNGKey(1), (500, 4))
    flow = NICE(theta, in_size=4, hidden_size=16,
                num_layers=2, num_coupling_layers=2)

    a = flow.mlp

    assert(a[0] != a[1])

def test_realnvp_mlps() -> None:
    """Test to check that the MLPs in a RealNVP instance are different."""
    theta = jax.random.normal(jax.random.PRNGKey(1), (500, 4))
    flow = RealNVP(theta, in_size=4, hidden_size=16,
                num_layers=2, num_coupling_layers=2)

    a = flow.additive_mlp
    b = flow.scaling_mlp

    assert(a[0] != a[1])
    assert(b[0] != b[1])
    assert(a[0] != b[0])
    assert(a[1] != b[1])
    assert(a[1] != b[0])
    assert(a[0] != b[1])

def test_nice_odd_dimensions() -> None:
    """Test NICE estimator with an odd number of dimensions."""
    key = jax.random.PRNGKey(45)

    odd_samples = jax.random.multivariate_normal(
        key,
        mean=jnp.zeros(3),
        cov=jnp.eye(3),
        shape=(nsamples,),
    )

    nice_estimator = NICE(
        odd_samples,
        in_size=3,
        hidden_size=50,
        num_layers=2,
        num_coupling_layers=4,
    )

    # check forward/inverse roundtrip
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, (100, 3))
    forward_transformed = nice_estimator.forward(z)
    inverse_transformed = nice_estimator.inverse(forward_transformed)
    error = jnp.mean(jnp.abs(inverse_transformed - z))
    assert error < 1e-3

    # check log_prob runs without crashing
    log_probs = nice_estimator.log_prob_under_NICE(z)
    assert log_probs.shape == (100,)
    assert jnp.all(jnp.isfinite(log_probs))


def test_realnvp_odd_dimensions() -> None:
    """Test RealNVP estimator with an odd number of dimensions."""
    key = jax.random.PRNGKey(46)

    odd_samples = jax.random.multivariate_normal(
        key,
        mean=jnp.zeros(3),
        cov=jnp.eye(3),
        shape=(nsamples,),
    )

    realnvp_estimator = RealNVP(
        odd_samples,
        in_size=3,
        hidden_size=50,
        num_coupling_layers=4,
    )

    # check forward/inverse roundtrip
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, (100, 3))
    forward_transformed, log_det = realnvp_estimator.forward(
        z, return_log_det=True
    )
    inverse_transformed = realnvp_estimator.inverse(forward_transformed)
    error = jnp.mean(jnp.abs(inverse_transformed - z))
    assert error < 1e-3

    # check log_prob runs without crashing
    log_probs = realnvp_estimator.log_prob_under_RealNVP(z)
    assert log_probs.shape == (100,)
    assert jnp.all(jnp.isfinite(log_probs))


def test_kde() -> None:
    """Test KDE estimator."""
    key = jax.random.PRNGKey(44)
    kde_estimator = KDE(
        original_samples,
        weights=weights,
        theta_ranges=bounds,
        bandwidth=0.08,
    )
    kde_estimator.train()
    key, subkey = jax.random.split(key)
    samples = kde_estimator.sample(subkey, 5000)

    prior_estimator = KDE(prior_samples, theta_ranges=bounds, bandwidth=0.08)
    prior_estimator.train()

    kld = kldivergence(kde_estimator, prior_estimator, samples)
    dim = model_dimensionality(kde_estimator, prior_estimator, samples)
    assert_allclose(kld, true_kl, rtol=0, atol=kl_atol)
    assert_allclose(dim, true_bmd, rtol=0, atol=smoothing_bmd_atol)

    kde_estimator.save("kde_test")
    loaded_estimator = KDE.load("kde_test")
    key, subkey = jax.random.split(key)
    samples = kde_estimator.sample(subkey, 1000)
    loaded_samples = loaded_estimator.sample(subkey, 1000)
    assert_allclose(samples, loaded_samples, rtol=1e-6, atol=1e-6)
    os.remove("kde_test.marg")
