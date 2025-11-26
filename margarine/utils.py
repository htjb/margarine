"""Utility functions for margarine package."""

import jax.numpy as jnp


def approximate_bounds(
    theta: jnp.ndarray, weights: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Function to estimate prior bounds from samples.

    Args:
        theta (jnp.ndarray): samples from the target distribution.
        weights (jnp.ndarray): weights for the samples from the
            target distribution.

    Return:
        a (jnp.ndarray): estimate of the upper bound on the prior.
        b (jnp.ndarray): estimate of the lower bound on the prior.
    """
    n = (jnp.sum(weights) ** 2) / (jnp.sum(weights**2))
    sample_max = jnp.max(theta, axis=0)
    sample_min = jnp.min(theta, axis=0)

    a = ((n - 2) * sample_max - sample_min) / (n - 3)
    b = ((n - 2) * sample_min - sample_max) / (n - 2)

    return a, b
