"""Utility functions for margarine package."""

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


@jax.jit
def approximate_bounds(
    theta: jnp.ndarray, weights: jnp.ndarray
) -> jnp.ndarray:
    """Function to estimate prior bounds from samples.

    Sample maximum and minimum are biased estimators of the true
    bounds of the distribution. This function provides an improved
    estimate using the weights of the samples. Comes from the expectation
    value of the maximum and minimum samples
    of a uniform distribution with an effective number of samples
    given by Kish's formula.

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
    b = ((n - 2) * sample_min - sample_max) / (n - 3)

    return jnp.stack([b, a])


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
    x = tfd.Uniform(min_val, max_val).cdf(x)
    x = tfd.Normal(0, 1).quantile(x)
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
    x = tfd.Normal(0, 1).cdf(x)
    x = tfd.Uniform(min_val, max_val).quantile(x)
    return x


def train_test_split(
    a: jnp.ndarray,
    b: jnp.ndarray,
    key: jnp.ndarray | None = None,
    test_size: float = 0.2,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Splitting data into training and testing sets.

    Function is equivalent
    to sklearn.model_selection.train_test_split but a and b
    are jax arrays.

    Args:
        a (jnp.ndarray): First set of data to be split.
        b (jnp.ndarray): Second set of data to be split.
        test_size (float): Proportion of data to be used for testing.
        key (jax.random.KeyArray): JAX random key for shuffling.

    Returns:
        a_train (jnp.ndarray): Training set from a.
        a_test (jnp.ndarray): Testing set from a.
        b_train (jnp.ndarray): Training set from b.
        b_test (jnp.ndarray): Testing set from b.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    n_samples = a.shape[0]
    n_test = int(n_samples * test_size)

    permuted_indices = jax.random.permutation(key, n_samples)
    test_indices = permuted_indices[:n_test]
    train_indices = permuted_indices[n_test:]

    a_train = a[train_indices]
    a_test = a[test_indices]
    b_train = b[train_indices]
    b_test = b[test_indices]

    return a_train, a_test, b_train, b_test
