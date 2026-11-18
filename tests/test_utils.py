"""Test the utility functions."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from margarine.utils.utils import (
    approximate_bounds,
    check_conditional_input,
    conditional_forward_transform,
    conditional_inverse_transform,
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


def test_conditional_forward_backward() -> None:
    """Test the forward and inverse transformations for conditional data."""
    key = jax.random.PRNGKey(0)
    num_samples = 1000

    # Generate random samples
    samples = jax.random.uniform(
        key, (num_samples, 1), minval=-5.0, maxval=5.0
    )

    # Perform forward transformation
    transformed_samples, mean, std = conditional_forward_transform(samples)

    # Perform inverse transformation
    recovered_samples = conditional_inverse_transform(
        transformed_samples, mean, std
    )

    # Check that the recovered samples are close to the original samples
    error = jnp.mean(jnp.abs(recovered_samples - samples))
    assert error < 1e-6


def test_conditional_check() -> None:
    """Test the conditional input check function.

    The function should broadcast every accepted shape to (N, d) and
    raise a ValueError for everything else.
    """
    key = jax.random.PRNGKey(0)
    n = 5

    # c=None is not allowed
    with pytest.raises(ValueError):
        check_conditional_input(None, 2, n)

    # a scalar is a single condition for every sample when d == 1
    c = check_conditional_input(jnp.array(0.5), 1, n)
    assert c.shape == (n, 1)
    assert_allclose(c, jnp.full((n, 1), 0.5))

    # (d,) is a single condition broadcast to every sample
    single = jnp.array([1.0, 2.0])
    c = check_conditional_input(single, 2, n)
    assert c.shape == (n, 2)
    assert_allclose(c, jnp.tile(single, (n, 1)))

    # (1, d) is also a single condition broadcast to every sample
    c = check_conditional_input(single[None, :], 2, n)
    assert c.shape == (n, 2)
    assert_allclose(c, jnp.tile(single, (n, 1)))

    # (N,) is one scalar condition per sample when d == 1
    per_sample = jax.random.uniform(key, (n,))
    c = check_conditional_input(per_sample, 1, n)
    assert c.shape == (n, 1)
    assert_allclose(c[:, 0], per_sample)

    # (N, d) is one condition per sample and is returned unchanged
    key, subkey = jax.random.split(key)
    per_sample = jax.random.uniform(subkey, (n, 2))
    c = check_conditional_input(per_sample, 2, n)
    assert c.shape == (n, 2)
    assert_allclose(c, per_sample)

    # wrong number of columns
    with pytest.raises(ValueError):
        check_conditional_input(jax.random.uniform(key, (n, 2)), 1, n)

    # wrong number of rows (neither 1 nor N)
    with pytest.raises(ValueError):
        check_conditional_input(jax.random.uniform(key, (3, 2)), 2, n)

    # 1D input whose length is neither d nor N
    with pytest.raises(ValueError):
        check_conditional_input(jax.random.uniform(key, (3,)), 2, n)

    # (N,) is only one condition per sample when d == 1
    with pytest.raises(ValueError):
        check_conditional_input(jax.random.uniform(key, (n,)), 2, n)

    # more than two dimensions
    with pytest.raises(ValueError):
        check_conditional_input(jax.random.uniform(key, (n, 2, 1)), 2, n)
