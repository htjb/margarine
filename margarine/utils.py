"""Utility functions for margarine package."""

import jax
import jax.numpy as jnp
from jax.scipy import stats


@jax.jit
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


@jax.jit
def forward_transform(
    x: jnp.ndarray, min_val: float, max_val: float
) -> jnp.ndarray:
    """Forward transform input samples.

    Normalise between 0 and 1 and then transform
    onto samples of standard normal distribution.

    Args:
        x (jnp.ndarray): Samples to be normalised.
        min_val (float): Minimum value for normalization.
        max_val (float): Maximum value for normalization.

    Returns:
        jnp.ndarray: Transformed samples.
    """
    x = stats.uniform.cdf(x, loc=min_val, scale=max_val - min_val)
    x = stats.norm.ppf(x)
    return x


@jax.jit
def inverse_transform(
    x: jnp.ndarray, min_val: float, max_val: float
) -> jnp.ndarray:
    """Inverse transform output samples.

    Inverts the processes in ``forward_transform``.

    Args:
        x (jnp.ndarray): Samples to be inverse normalised.
        min_val (float): Minimum value for normalization.
        max_val (float): Maximum value for normalization.

    Returns:
        jnp.ndarray: Inverse transformed samples.
    """
    x = stats.norm.cdf(x)
    x = stats.uniform.ppf(x, loc=min_val, scale=max_val - min_val)
    return x
