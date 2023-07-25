from tensorflow_probability import distributions as tfd
import tensorflow as tf


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
    x = tf.cast(x, tf.float32)
    x = tfd.Uniform(min, max).cdf(x)
    x = tfd.Normal(0, 1).quantile(x)
    return x


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
    print(min)
    min = tf.cast(min, tf.float32)
    max = tf.cast(max, tf.float32)
    x = tfd.Uniform(min, max).quantile(x)
    return x
