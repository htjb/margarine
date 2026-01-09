"""Test the importance sampling feature."""

import jax
import jax.numpy as jnp
from jax.scipy import stats
from numpy.testing import assert_allclose

from margarine.estimators.realnvp import RealNVP
from margarine.statistics import integrate
from margarine.utils.utils import approximate_bounds

nsamples = 5000
key = jax.random.PRNGKey(0)

target_mean = jnp.array([0.0, 0.0])
target_cov = jnp.array([[1.0, 0.8], [0.8, 1.0]])

original_samples = jax.random.multivariate_normal(
    key,
    mean=target_mean,
    cov=target_cov,
    shape=(nsamples,),
)
likelihood_probs = stats.multivariate_normal.logpdf(
    original_samples,
    mean=target_mean,
    cov=target_cov,
)
weights = jnp.ones(len(original_samples))

prior_probs = stats.uniform.logpdf(original_samples, loc=-4.0, scale=8.0)
prior_probs = jnp.sum(prior_probs, axis=-1)


bounds = approximate_bounds(original_samples, weights)
prior_samples = jax.random.uniform(
    key, (nsamples, 2), minval=bounds[0], maxval=bounds[1]
)


def test_importance_sampling() -> None:
    """Test importance sampling functionality."""
    key = jax.random.PRNGKey(123)

    # Create a RealNVP estimator
    realnvp_estimator = RealNVP(
        original_samples,
        weights=weights,
        in_size=2,
        hidden_size=25,
        num_layers=2,
        num_coupling_layers=2,
        theta_ranges=bounds,
    )

    key, subkey = jax.random.split(key)
    realnvp_estimator.train(
        subkey, learning_rate=1e-4, epochs=2000, patience=50
    )

    prior_estimator = RealNVP(
        prior_samples,
        in_size=2,
        hidden_size=50,
        num_layers=2,
        num_coupling_layers=2,
        theta_ranges=bounds,
    )

    key, subkey = jax.random.split(key)
    prior_estimator.train(subkey, learning_rate=1e-3, epochs=2000, patience=50)

    # Perform importance sampling integration
    integral = integrate(
        realnvp_estimator,
        lambda x: stats.multivariate_normal.logpdf(
            x, mean=target_mean, cov=target_cov
        ),
        prior_estimator,
        sample_size=20000,
    )

    expected_value = jnp.prod(
        1 / (bounds[1] - bounds[0])
    )  # Expected value for the integral in 2D
    # Assert that the computed integral is close to the expected value
    assert_allclose(
        jnp.exp(integral["log_integral"]), expected_value, rtol=0.01, atol=0.01
    )
