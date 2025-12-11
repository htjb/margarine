"""Test the utility functions."""

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from margarine.utils.utils import (
    approximate_bounds,
    forward_transform,
    inverse_transform,
)


def test_transformations_and_bounds() -> None:
    """Test the forward and inverse transformations."""
    key = jax.random.PRNGKey(0)
    num_samples = 1000
    num_dims = 3

    # Generate random samples
    samples = jax.random.uniform(
        key, (num_samples, num_dims), minval=-5.0, maxval=5.0
    )
    weights = jnp.ones(num_samples)

    bounds = approximate_bounds(samples, weights)
    lower_bounds, upper_bounds = bounds
    assert_allclose(lower_bounds, -5.0, rtol=1e-2)
    assert_allclose(upper_bounds, 5.0, rtol=1e-2)

    # Perform forward transformation
    transformed_samples = forward_transform(
        samples, lower_bounds, upper_bounds
    )

    # Perform inverse transformation
    recovered_samples = inverse_transform(
        transformed_samples, lower_bounds, upper_bounds
    )

    # Check that the recovered samples are close to the original samples
    error = jnp.mean(jnp.abs(recovered_samples - samples))
    assert error < 1e-6
