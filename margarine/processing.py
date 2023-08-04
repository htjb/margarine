from tensorflow_probability import distributions as tfd
import tensorflow as tf
import random


@tf.function(jit_compile=True)
def _forward_transform(x, min=0, max=1):
    r"""

    Tranforms input samples. Normalise between 0 and 1 and then tranform
    onto samples of standard normal distribution (i.e. base of
    tfd.TransformedDistribution).

    **parameters:**

        x: **array**
            | Samples to be normalised.

        min: **array or list**
            | Passed from the bijectors code. (mathematical
                description of their
                values...)

        max: **array or list**
            | Passed from the bijectors code.
                (mathematical description of their
                values...)

    """
    x = tfd.Uniform(min, max).cdf(x)
    x = tfd.Normal(0, 1).quantile(x)
    return x


@tf.function(jit_compile=True)
def _inverse_transform(x, min, max):
    r"""

    Tranforms output samples. Inverts the processes in
    ``forward_transofrm``.

    **parameters:**

        x: **array**
            | Samples to be normalised.

        min: **array or list**
            | Passed from the bijectors code.
                (mathematical description of their
                values...)

        max: **array or list**
            | Passed from the bijectors code.
                (mathematical description of their
                values...)

    """
    x = tfd.Normal(0, 1).cdf(x)
    x = tfd.Uniform(min, max).quantile(x)
    return x


def pure_tf_train_test_split(a, b, test_size=0.2):

    """
    Splitting data into training and testing sets. Function is equivalent
    to sklearn.model_selection.train_test_split but a and b
    are tensorflow tensors.

    **parameters:**

        a: **array**
            | Samples to be split.

        b: **array**
            | Weights to be split.

        test_size: **float**
            | Fraction of data to be used for testing.
    """

    idx = random.sample(range(len(a)), int(len(a)*test_size))

    a_train = tf.gather(a,
                        tf.convert_to_tensor(
                            list(set(range(len(a))) - set(idx))))
    b_train = tf.gather(b,
                        tf.convert_to_tensor(
                            list(set(range(len(b))) - set(idx))))
    a_test = tf.gather(a, tf.convert_to_tensor(idx))
    b_test = tf.gather(b, tf.convert_to_tensor(idx))

    return a_train, a_test, b_train, b_test
