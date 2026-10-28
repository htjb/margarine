"""Utility functions for margarine package."""

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import initializers
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

default_kernel_init = initializers.lecun_normal()


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


def coupling_permutations(
    key: jnp.ndarray, in_size: int, num_coupling_layers: int
) -> list[jnp.ndarray]:
    """Build permutations that guarantee every dimension gets transformed.

    The coupling layers transform the first in_size // 2 columns
    of their input. Random permutations can leave a dimension out of
    that block in every layer, so it is never transformed. Instead
    we shuffle the dimensions and deal them out in_size // 2 at a time,
    reshuffling once every dimension has been dealt. Each dimension is
    therefore transformed at least once every
    ceil(in_size / (in_size // 2)) layers, while the groupings
    of transformed and conditioning dimensions remain random.

    The flows apply the permutations cumulatively (layer i permutes
    the output of layer i - 1), so the desired ordering of the original
    dimensions for each layer is converted to a permutation relative
    to the ordering left by the previous layer.

    Args:
        key: JAX random key for the shuffles.
        in_size: Number of dimensions.
        num_coupling_layers: Number of coupling layers.

    Returns:
        A list of permutations, one per coupling layer, to be applied
        cumulatively.
    """
    pass_size = in_size // 2

    # current[j] is the original dimension sitting in column j
    current = jnp.arange(in_size)
    deck = []
    permutations = []
    for _ in range(num_coupling_layers):
        transformed = []
        while len(transformed) < pass_size:
            if not deck:
                key, subkey = jax.random.split(key)
                deck = jax.random.permutation(subkey, in_size).tolist()
            # skip dimensions already dealt to this layer, which can
            # happen when a deal spans two shuffles
            dim = next(d for d in deck if d not in transformed)
            deck.remove(dim)
            transformed.append(dim)

        # desired ordering of the original dimensions for this layer
        target = jnp.array(
            transformed + [d for d in range(in_size) if d not in transformed]
        )
        # argsort(current) undoes the previous ordering and indexing
        # with target then applies the desired one
        permutations.append(jnp.argsort(current)[target])
        current = target

    return permutations


def build_mlp(
    in_size: int,
    hidden_size: int,
    num_layers: int,
    pass_size: int,
    nnx_rngs: dict,
    kernel_init: nnx.initializers.Initializer = default_kernel_init,
) -> nnx.Sequential:
    """Build a multi-layer perceptron (MLP) using Flax's nnx module.

    Args:
        in_size: Input size of the MLP.
        hidden_size: Size of the hidden layers.
        num_layers: Number of hidden layers.
        pass_size: Size of the pass-through layer.
        nnx_rngs: RNGs for Flax.
        kernel_init: Optional initializer for the kernel weights.

    Returns:
        An instance of nnx.Sequential representing the MLP.
    """
    layers = nnx.List()
    layers.append(
        nnx.Linear(
            in_size, hidden_size, rngs=nnx_rngs, kernel_init=kernel_init
        )
    )
    layers.append(lambda x: jax.nn.relu(x))
    for _ in range(num_layers):
        layers.append(
            nnx.Linear(
                hidden_size,
                hidden_size,
                rngs=nnx_rngs,
                kernel_init=kernel_init,
            )
        )
        layers.append(lambda x: jax.nn.relu(x))
    layers.append(
        nnx.Linear(
            hidden_size, pass_size, rngs=nnx_rngs, kernel_init=kernel_init
        )
    )
    return nnx.Sequential(*layers)
