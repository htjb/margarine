from tensorflow_probability import distributions as tfd
import warnings
import numpy as np


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
    x = tfd.Uniform(min, max).cdf(x).numpy()
    x = tfd.Normal(0, 1).quantile(x).numpy()
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
    x = tfd.Normal(0, 1).cdf(x).numpy()
    x = tfd.Uniform(min, max).quantile(x).numpy()
    return x


def reweight(samples, prior, weights, prior_weights, evidence):
    r"""

    Helper function to transform samples, priors and evidences to a space
    where the prior is uniform.

    """

    warnings.warn('Not implemented yet.')

    n = len(samples)
    samples_max = np.max(samples, axis=0)
    samples_min = np.min(samples, axis=0)
    a = ((n-2)*samples_max-samples_min)/(n-3)
    b = ((n-2)*samples_min-samples_max)/(n-3)
    samples_min = b
    samples_max = a

    reweighted_samples = []
    reweighted_prior = []
    new_weights = []
    new_prior_weights = []
    new_evidence = []
    return reweighted_samples, reweighted_prior, new_weights, \
        new_prior_weights, new_evidence
