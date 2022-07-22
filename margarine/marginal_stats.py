import tensorflow as tf
import numpy as np
from margarine.processing import _forward_transform
from tensorflow_probability import distributions as tfd
import margarine
import warnings
import pandas as pd
from margarine.maf import MAF


class calculate(object):

    r"""

    This class, once initalised with a trained MAF or KDE and samples,
    can be used to calculate marginal KL divergences and
    bayesian dimensionalities.

    **Paramesters:**

        de: **instance of MAF class or KDE class**
            | This should be a loaded and trained instance of a MAF or KDE.
                Bijectors can be loaded like so

                .. code:: python

                    from margarine.maf import MAF
                    from margarine.kde import KDE

                    file = '/trained_maf.pkl'
                    maf = MAF.load(file)

                    file = '/trained_kde.pkl'
                    kde = KDE.load(file)

        samples: **numpy array**
            | This should be the output of the bijector when called to generate
                a set of samples from the replicated probability
                distribution. e.g. after loading a trained MAF we would pass

                .. code:: python

                    u = np.random.uniform(0, 1, size=(10000, 5))
                    prior_limits = np.array([[0]*5, [1]*5])
                    samples = maf(u, prior_limits)

    **Kwargs:**

        prior_samples: **numpy array / default=None**
            | Can be provided if the prior is non-uniform and will be
                used to generate a prior density estimator to calcualte
                prior log-probabilities. If not provided the prior is
                assumed to be uniform.

        prior_weights: **numpy array / default=None**
            | Weights associated with the above prior samples.

    """

    def __init__(self, de, **kwargs):

        self.de = de
        self.theta = self.de.theta
        self.theta_weights = self.de.sample_weights
        self.samples = self.de.sample(len(self.theta))

        self.prior_de = kwargs.pop('prior_de', None)
        self.prior_samples = kwargs.pop('prior_samples', None)
        self.prior_weights = kwargs.pop('prior_weights', None)

        if self.prior_samples is None:
            warnings.warn('If prior samples are not provided the prior is ' +
                'assumed to be uniform and the posterior samples are ' +
                'are assumed to be from the same uniform space.')
        else:
            if self.prior_weights is None:
                warnings.warn('No prior weights have been provided. ' +
                    'Assuming there are none.')

    def statistics(self):

        def mask_arr(arr):
            return arr[np.isfinite(arr)], np.isfinite(arr)

        min = self.de.theta_min
        max = self.de.theta_max

        if isinstance(self.de, margarine.kde.KDE):
            logprob_func = self.de.log_prob
            logprob = logprob_func(self.samples)
            theta_logprob = logprob_func(self.theta)
        elif isinstance(self.de, margarine.maf.MAF):
            logprob_func = self.de.log_prob
            logprob = logprob_func(self.samples)
            theta_logprob = logprob_func(self.theta)

        args = np.argsort(theta_logprob)
        self.theta_weights = self.theta_weights[args]
        theta_logprob = theta_logprob[args]

        deargs = np.argsort(logprob)
        logprob = logprob[deargs]
        wde = [np.sum(self.theta_weights)/len(logprob)]*len(logprob)
        logprob = np.interp(
            np.cumsum(self.theta_weights), np.cumsum(wde), logprob)

        mid_point = np.log((np.exp(logprob) + np.exp(theta_logprob))/2)

        if self.prior_samples is None:
            self.base = tfd.Blockwise(
                [tfd.Uniform(self.de.theta_min[i], self.de.theta_max[i])
                for i in range(self.samples.shape[-1])])
            prior = self.base.sample(len(self.theta)).numpy()

            theta_base_logprob = self.base.log_prob(self.theta).numpy()

            base_logprob = self.base.log_prob(prior).numpy()
            prior_wde = [np.sum(self.theta_weights)/len(base_logprob)]*len(base_logprob)

            base_logprob = base_logprob[deargs]
            base_logprob = np.interp(
                np.cumsum(self.theta_weights), np.cumsum(prior_wde), base_logprob)

            theta_base_logprob = theta_base_logprob[args]
        elif self.prior_de is not None:
            if isinstance(self.prior_de, margarine.kde.KDE):
                self.base = self.prior_de.copy()
                base_logprob_func = self.base.log_prob
                de_prior_samples = self.base.sample(len(self.theta))
                theta_base_logprob = base_logprob_func(self.prior_samples)
                base_logprob = base_logprob_func(self.theta)
            elif isinstance(self.prior_de, margarine.maf.MAF):
                self.base = self.prior_de.copy()
                base_logprob_func = self.base.log_prob
                de_prior_samples = self.base.sample(len(self.theta))
                theta_base_logprob = base_logprob_func(self.prior_samples)
                base_logprob = base_logprob_func(self.theta)
        else:
            if isinstance(self.de, margarine.kde.KDE):
                self.base = margarine.kde.KDE(
                    self.prior_samples, self.prior_weights)
                self.base.generate_kde()
                base_logprob_func = self.base.log_prob
                de_prior_samples = self.base.sample(len(self.theta))
                theta_base_logprob = base_logprob_func(self.prior_samples)
                base_logprob = base_logprob_func(de_prior_samples)
            elif isinstance(self.de, margarine.maf.MAF):
                self.base = MAF(
                    self.prior_samples, self.prior_weights)
                self.base.train(epochs=250)
                base_logprob_func = self.base.log_prob
                de_prior_samples = self.base.sample(len(self.theta))
                theta_base_logprob = base_logprob_func(self.prior_samples)
                base_logprob = base_logprob_func(de_prior_samples)

            base_logprob = base_logprob[deargs]
            base_logprob = np.interp(
                np.cumsum(self.theta_weights), np.cumsum(wde), base_logprob)

            base_args = np.argsort(theta_base_logprob)
            theta_base_logprob = theta_base_logprob[base_args]
            prior_weights = np.cumsum(self.prior_weights[base_args])

            prior_weights = (np.cumsum(self.theta_weights).max()-
                np.cumsum(self.theta_weights).min())*(prior_weights -
                prior_weights.min())/ \
                (prior_weights.max() - prior_weights.min()) + \
                np.cumsum(self.theta_weights).min()

            theta_base_logprob = np.interp(np.cumsum(self.theta_weights),
                prior_weights, theta_base_logprob)

        midbasepoint = np.log((np.exp(base_logprob) + np.exp(theta_base_logprob))/2)

        mid_logL, mask = mask_arr(mid_point - midbasepoint)
        midkl = np.average(mid_logL, weights=self.theta_weights[mask])
        midbd = 2*np.cov(mid_logL, aweights=self.theta_weights[mask])
        de_logL, mask = mask_arr(logprob - base_logprob)
        dekl = np.average(de_logL, weights=self.theta_weights[mask])
        debd = 2*np.cov(de_logL, aweights=self.theta_weights[mask])
        theta_logL, mask = mask_arr(theta_logprob - theta_base_logprob)
        thetakl = np.average(theta_logL, weights=self.theta_weights[mask])
        thetabd = 2*np.cov(theta_logL, aweights=self.theta_weights[mask])

        kl_array = np.sort([midkl, dekl, thetakl])
        bd_array = np.sort([midbd, debd, thetabd])

        results_dict = {'Statistic': ['KL Divergence', 'BMD'],
            'Value': [kl_array[1], bd_array[1]],
            'Lower Bound': [kl_array[0], bd_array[0]],
            'Upper Bound': [kl_array[2], bd_array[2]]}

        results = pd.DataFrame(results_dict)
        results.set_index('Statistic', inplace=True)

        return results
