"""Test the NICE and RealNVP estimators with conditioning in margarine."""

import os

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from margarine.estimators.nice import NICE
from margarine.estimators.realnvp import RealNVP
from margarine.statistics import kldivergence, model_dimensionality


def true_log_prob(x: jnp.ndarray, c_val: float) -> jnp.ndarray:
    """Return the true log probability of x given c_val."""
    cov = true_scale(c_val) ** 2 * base_cov
    return jax.scipy.stats.multivariate_normal.logpdf(
        x, true_mean(jnp.array([[c_val]]))[0], cov
    )


nsamples = 10000
key = jax.random.PRNGKey(0)
key, kc, kz = jax.random.split(key, 3)

base_cov = jnp.array([[1.0, 0.5], [0.5, 1.0]])
L = jnp.linalg.cholesky(base_cov)


def true_mean(c: jnp.ndarray) -> jnp.ndarray:  # c: (N, 1) -> (N, 2)
    """Return the true mean of the conditional distribution given c."""
    return jnp.hstack([2 * c, -c])


def true_scale(c: jnp.ndarray) -> jnp.ndarray:
    """Return the true scale of the conditional distribution given c."""
    return 0.5 + 0.25 * c


c = jax.random.uniform(kc, (nsamples, 1), minval=-1, maxval=1)
z = jax.random.normal(kz, (nsamples, 2))
theta = true_mean(c) + true_scale(c) * (z @ L.T)
bounds = jnp.array([[-6.0, -6.0], [6.0, 6.0]])

kl_tol = 0.1  # KL divergence tolerance for the test


def test_conditional_nice() -> None:
    """Test NICE estimator."""
    key = jax.random.PRNGKey(42)

    nice_estimator = NICE(
        theta,
        in_size=2,
        hidden_size=50,
        num_layers=2,
        num_coupling_layers=4,
        c=c,
        theta_ranges=bounds,
    )

    # check the forward and inverse transformations
    key, subkey = jax.random.split(key)
    z = jax.random.normal(key, (nsamples, 2))
    forward_transformed = nice_estimator.forward(z, c)
    inverse_transformed = nice_estimator.inverse(forward_transformed, c)
    error = jnp.mean(jnp.abs(inverse_transformed - z))
    assert error < 1e-3

    nice_estimator = NICE(
        theta,
        in_size=2,
        hidden_size=50,
        num_layers=2,
        num_coupling_layers=6,
        theta_ranges=bounds,
        c=c,
    )

    key, subkey = jax.random.split(key)
    nice_estimator.train(
        subkey,
        learning_rate=1e-3,
        epochs=2000,
        patience=50,
        batch_size=1024,
    )

    for c_val in [-0.5, 0.0, 0.5]:
        c_arr = jnp.array([[c_val]])
        # draw from the true p(x | c_val)
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (5000, 2))
        x = true_mean(c_arr) + true_scale(c_arr) * (z @ L.T)

        kl = jnp.mean(
            true_log_prob(x, c_val)
            - nice_estimator.log_prob(x, jnp.array([c_val]))
        )
        print(f"KL divergence for c={c_val}: {kl}")
        assert jnp.abs(kl) < kl_tol

    nice_estimator.save("nice_test")
    loaded_estimator = NICE.load("nice_test")
    assert loaded_estimator.conditional is True
    key, subkey = jax.random.split(key)
    samples = nice_estimator.sample(subkey, 1000, c=jnp.array([[0.0]]))
    loaded_samples = loaded_estimator.sample(
        subkey, 1000, c=jnp.array([[0.0]])
    )
    assert_allclose(samples, loaded_samples, rtol=1e-6, atol=1e-6)
    os.remove("nice_test.marg")


def test_conditional_realnvp() -> None:
    """Test RealNVP estimator."""
    key = jax.random.PRNGKey(42)

    realnvp_estimator = RealNVP(
        theta,
        in_size=2,
        hidden_size=50,
        num_layers=2,
        num_coupling_layers=4,
        c=c,
        theta_ranges=bounds,
    )

    # check the forward and inverse transformations
    key, subkey = jax.random.split(key)
    z = jax.random.normal(key, (nsamples, 2))
    forward_transformed = realnvp_estimator.forward(z, c)
    inverse_transformed = realnvp_estimator.inverse(forward_transformed, c)
    error = jnp.mean(jnp.abs(inverse_transformed - z))
    assert error < 1e-3

    realnvp_estimator = RealNVP(
        theta,
        in_size=2,
        hidden_size=50,
        num_layers=2,
        num_coupling_layers=6,
        theta_ranges=bounds,
        c=c,
    )

    key, subkey = jax.random.split(key)
    realnvp_estimator.train(
        subkey,
        learning_rate=1e-3,
        epochs=2000,
        patience=50,
        batch_size=1024,
    )

    for c_val in [-0.5, 0.0, 0.5]:
        c_arr = jnp.array([[c_val]])
        # draw from the true p(x | c_val)
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (5000, 2))
        x = true_mean(c_arr) + true_scale(c_arr) * (z @ L.T)

        kl = jnp.mean(
            true_log_prob(x, c_val)
            - realnvp_estimator.log_prob(x, jnp.array([c_val]))
        )
        assert jnp.abs(kl) < kl_tol

    realnvp_estimator.save("realnvp_test")
    loaded_estimator = RealNVP.load("realnvp_test")
    assert loaded_estimator.conditional is True
    key, subkey = jax.random.split(key)
    samples = realnvp_estimator.sample(subkey, 1000, c=jnp.array([[0.0]]))
    loaded_samples = loaded_estimator.sample(
        subkey, 1000, c=jnp.array([[0.0]])
    )
    assert_allclose(samples, loaded_samples, rtol=1e-6, atol=1e-6)
    os.remove("realnvp_test.marg")
