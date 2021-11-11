from tensorflow_probability import distributions as tfd


def _forward_transform(x, min, max):
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
    return tfd.Normal(0, 1).quantile((x - min)/(max-min)).numpy()


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
    return tfd.Normal(0, 1).cdf(x).numpy()*(max-min) + min
