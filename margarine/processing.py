"""Processing functions for margarine package."""

import random

import tensorflow as tf
from tensorflow_probability import distributions as tfd


@tf.function(jit_compile=True)
def _forward_transform(
    x: tf.Tensor,
    min: float | int | tf.Tensor = 0,
    max: float | int | tf.tensor = 1,
) -> tf.Tensor:
    r"""Forward tranforms input samples.

    Normalise between 0 and 1 and then tranform
    onto samples of standard normal distribution (i.e. base of
    tfd.TransformedDistribution).

    Args:
        x: (tf.Tensor) Samples to be normalised.
        min: (float | int | tf.Tensor) Passed from the bijectors code.
        max: (float | int | tf.Tensor) Passed from the bijectors code.
    """
    x = tfd.Uniform(min, max).cdf(x)
    x = tfd.Normal(0, 1).quantile(x)
    return x


@tf.function(jit_compile=True)
def _inverse_transform(
    x: tf.Tensor, min: float | int | tf.Tensor, max: float | int | tf.Tensor
) -> tf.Tensor:
    r"""Inverse tranforms output samples.

    Inverts the processes in ``forward_transofrm``.

    Args:
        x: (tf.Tensor) Samples to be inverse normalised.
        min: (float | int | tf.Tensor) Passed from the bijectors code.
        max: (float | int | tf.Tensor) Passed from the bijectors code.
    """
    x = tfd.Normal(0, 1).cdf(x)
    x = tfd.Uniform(min, max).quantile(x)
    return x


def pure_tf_train_test_split(
    a: tf.Tensor, b: tf.Tensor, test_size: float = 0.2
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Splitting data into training and testing sets.

    Function is equivalent
    to sklearn.model_selection.train_test_split but a and b
    are tensorflow tensors.

    Args:
        a: (tf.Tensor) First set of data to be split.
        b: (tf.Tensor) Second set of data to be split.
        test_size: (float) Proportion of data to be used for testing.
    """
    idx = random.sample(range(len(a)), int(len(a) * test_size))

    a_train = tf.gather(
        a, tf.convert_to_tensor(list(set(range(len(a))) - set(idx)))
    )
    b_train = tf.gather(
        b, tf.convert_to_tensor(list(set(range(len(b))) - set(idx)))
    )
    a_test = tf.gather(a, tf.convert_to_tensor(idx))
    b_test = tf.gather(b, tf.convert_to_tensor(idx))

    return a_train, a_test, b_train, b_test
