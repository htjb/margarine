import tensorflow as tf
import numpy as np
from nestedflows2d0.processing import _forward_transform


class bijector_calculations(object):

    r"""

    This class, once initalised with a trained bijector and samples from
    that bijector, can be used to calculate marginal KL divergences and
    bayesian dimensionalities.

    **Paramesters:**

    bij: **instance of Bijector class**
        | This should be a loaded and trained instance of a bijector. Bijectors
            can be loaded like so

            .. code:: python

                from ...bijector import Bijector

                file = '/trained_bijector.pkl'
                bij = Bijector.load(file)

    samples: **np.array**
        | This should be the output of the bijector when called to generate
            a set of samples from the replicated probability distribution. e.g.
            after loading a trained bijector we would pass

            .. code:: python

                u = np.random.uniform(0, 1, size=(10000, 5))
                prior_limits = np.array([[0]*5, [1]*5])
                samples = bij(u, prior_limits)

    """

    def __init__(self, bij, samples):

        self.bij = bij
        self.samples = samples

    def _calc_logL(self):

        r"""

        This is a helper function which is used by klDiv() and
        bayesian_dimensionality() to calculate the difference between the log
        probability of the replica samples (replica posterior) and the log
        probability of the base distribution (prior).

        """

        logprob = self.bij.maf.log_prob(
            _forward_transform(
                self.samples, self.bij.theta_min, self.bij.theta_max))
        base_logprob = self.bij.base.log_prob(
            _forward_transform(
                self.samples, self.bij.theta_min, self.bij.theta_max))

        def mask_tensor(tensor):
            return tf.boolean_mask(tensor, np.isfinite(tensor))

        logprob = mask_tensor(logprob)
        base_logprob = mask_tensor(base_logprob)
        logL = logprob - base_logprob
        return logL

    def klDiv(self):

        r"""

        Calculates the kl divergence between samples from the bijector
        (replica posterior) and the base distribution (prior).

        """
        
        logL = self._calc_logL()
        return tf.reduce_mean(logL)

    def bayesian_dimensionality(self):

        r"""

        Calculates the bayesian dimensionality of the
        samples from the bijector.
        More details on bayesian dimensionality can be found in
        https://arxiv.org/abs/1903.06682.

        """

        logL = self._calc_logL()
        return 2*(tf.reduce_mean(logL**2) - tf.reduce_mean(logL)**2)

    def evidence(self):

        r"""

        Function used to caluclate the marginal evidence.

        """

        logprob = self.bij.maf.log_prob(
            _forward_transform(
                self.samples, self.bij.theta_min, self.bij.theta_max))
        D = self.klDiv()
        return tf.reduce_mean(logprob) - D

class kde_calculations(object):

    r"""

    This class, once initalised with KDE and samples from
    that KDE, can be used to calculate marginal KL divergences and
    bayesian dimensionalities.

    **Paramesters:**

    kde: **instance of KDE class**
        | This should be a loaded instance of a KDE. KDEs
            can be loaded like so

            .. code:: python

                from ... import KDE

                file = '/trained_kde.pkl'
                kde = KDE.load(file)

    samples: **np.array**
        | This should be the output of the KDE when called to generate
            a set of samples from the replicated probability distribution.

    """

    def __init__(self, kde, samples):

        self.kde = kde
        self.samples = samples

    def _calc_logL(self):

        r"""

        This is a helper function which is used by klDiv() and
        bayesian_dimensionality() to calculate the difference between the log
        probability of the replica samples (replica posterior) and the log
        probability of the original distribution (prior).

        """

        # return logL

    def klDiv(self):

        r"""

        Calculates the kl divergence between samples from the KDE
        (replica posterior) and the original distribution (prior).

        """
        # logL = _calc_logL(self.kde, self.samples)
        # return tf.reduce_mean(logL)

    def bayesian_dimensionality(self):

        r"""

        Calculates the bayesian dimensionality of the samples from the KDE.

        """

        # logL = _calc_logL(self.kde, self.samples)
        # return 2*(tf.reduce_mean(logL**2) - tf.reduce_mean(logL)**2)
