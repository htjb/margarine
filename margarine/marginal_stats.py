import numpy as np
from tensorflow_probability import distributions as tfd
import pandas as pd
from margarine.maf import MAF
from tqdm import tqdm
from scipy.special import logsumexp


class calculate(object):
    r"""

    This class, once initalised with a trained MAF or KDE and samples,
    can be used to calculate marginal KL divergences and
    bayesian dimensionalities.

    **Paramesters:**

        de: **instance of MAF class or KDE class**
            | This should be a loaded and trained instance of a MAF, clusterMAF
                or KDE.Bijectors can be loaded like so

                .. code:: python

                    from margarine.maf import MAF
                    from margarine.kde import KDE
                    from margarine.clustered import clusterMAF

                    file = '/trained_maf.pkl'
                    maf = MAF.load(file)

                    file = '/trained_kde.pkl'
                    kde = KDE.load(file)

                    file = '/trained_clustered_maf.pkl'
                    clustered_maf = clusterMAF.load(file)

        samples: **numpy array**
            | This should be the output of the bijector when called to generate
                a set of samples from the replicated probability
                distribution. e.g. after loading a trained MAF we would pass

                .. code:: python

                    u = np.random.uniform(0, 1, size=(10000, 5))
                    prior_limits = np.array([[0]*5, [1]*5])
                    samples = maf(u, prior_limits)

    **Kwargs:**

        prior_de: **instance of MAF class, clusterMAF class or KDE class**
            | This should be a loaded and trained instance of a MAF, clusterMAF
                or KDE for the prior.
                If not provided, a uniform prior will be used.

    """

    def __init__(self, de, **kwargs):
        self.de = de

        self.theta = self.de.theta
        self.theta_weights = self.de.sample_weights
        self.theta_min = self.de.theta_min
        self.theta_max = self.de.theta_max

        if isinstance(self.de, MAF):
            self.theta_weights = self.theta_weights.numpy()
            self.theta_min = self.theta_min.numpy()
            self.theta_max = self.theta_max.numpy()

        self.samples = self.de.sample(len(self.theta))

        self.prior_de = kwargs.pop("prior_de", None)

    def statistics(self):
        """
        Calculate marginal bayesian KL divergence and dimensionality with
        approximate errors.
        """

        def mask_arr(arr):
            return arr[np.isfinite(arr)], np.isfinite(arr)

        logprob = self.de.log_prob(self.samples)
        theta_logprob = self.de.log_prob(self.theta)

        if isinstance(self.de, MAF):
            logprob = logprob.numpy()
            theta_logprob = theta_logprob.numpy()

        args = np.argsort(theta_logprob)
        self.theta_weights = self.theta_weights[args]
        theta_logprob = theta_logprob[args]

        deargs = np.argsort(logprob)
        logprob = logprob[deargs]
        wde = [np.sum(self.theta_weights) / len(logprob)] * len(logprob)
        logprob = np.interp(
            np.cumsum(self.theta_weights), np.cumsum(wde), logprob
        )

        mid_point = np.log((np.exp(logprob) + np.exp(theta_logprob)) / 2)

        if self.prior_de is None:
            self.base = tfd.Blockwise(
                [
                    tfd.Uniform(self.theta_min[i], self.theta_max[i])
                    for i in range(self.samples.shape[-1])
                ]
            )
            prior = self.base.sample(len(self.theta)).numpy()

            theta_base_logprob = self.base.log_prob(self.theta).numpy()
            base_logprob = self.base.log_prob(prior).numpy()

            prior_wde = [np.sum(self.theta_weights) / len(base_logprob)] * len(
                base_logprob
            )

            base_logprob = base_logprob[deargs]
            base_logprob = np.interp(
                np.cumsum(self.theta_weights),
                np.cumsum(prior_wde),
                base_logprob,
            )

            theta_base_logprob = theta_base_logprob[args]
        elif self.prior_de is not None:
            self.base = self.prior_de.copy()
            de_prior_samples = self.base.sample(len(self.theta))
            theta_base_logprob = self.base.log_prob(de_prior_samples)
            base_logprob = self.base.log_prob(self.theta)

            if isinstance(self.prior_de, MAF):
                theta_base_logprob = theta_base_logprob.numpy()
                base_logprob = base_logprob.numpy()

            base_logprob = base_logprob[deargs]
            base_logprob = np.interp(
                np.cumsum(self.theta_weights), np.cumsum(wde), base_logprob
            )

            base_args = np.argsort(theta_base_logprob)
            theta_base_logprob = theta_base_logprob[base_args]
            prior_weights = np.cumsum(self.prior_de.sample_weights[base_args])

            prior_weights = (
                np.cumsum(self.theta_weights).max()
                - np.cumsum(self.theta_weights).min()
            ) * (prior_weights - prior_weights.min()) / (
                prior_weights.max() - prior_weights.min()
            ) + np.cumsum(
                self.theta_weights
            ).min()

            theta_base_logprob = np.interp(
                np.cumsum(self.theta_weights),
                prior_weights,
                theta_base_logprob,
            )

        midbasepoint = np.log(
            (np.exp(base_logprob) + np.exp(theta_base_logprob)) / 2
        )

        mid_logL, mask = mask_arr(mid_point - midbasepoint)
        midkl = np.average(mid_logL, weights=self.theta_weights[mask])
        midbd = 2 * np.cov(mid_logL, aweights=self.theta_weights[mask])
        de_logL, mask = mask_arr(logprob - base_logprob)
        dekl = np.average(de_logL, weights=self.theta_weights[mask])
        debd = 2 * np.cov(de_logL, aweights=self.theta_weights[mask])
        theta_logL, mask = mask_arr(theta_logprob - theta_base_logprob)
        thetakl = np.average(theta_logL, weights=self.theta_weights[mask])
        thetabd = 2 * np.cov(theta_logL, aweights=self.theta_weights[mask])

        kl_array = np.sort([midkl, dekl, thetakl])
        bd_array = np.sort([midbd, debd, thetabd])

        results = {
            "KL Divergence": kl_array[1],
            "KL Lower Bound": kl_array[0],
            "KL Upper Bound": kl_array[2],
            "BMD": bd_array[1],
            "BMD Lower Bound": bd_array[0],
            "BMD Upper Bound": bd_array[2],
        }

        return results

    def integrate(
        self,
        loglikelihood,
        prior_pdf,
        batch_size=1000,
        sample_size=10000,
        logzero=-1e30,
    ):
        """
        Importance sampling integration of a likelihood function

        args:
            loglikelihood: **function**
                | A function that takes a numpy array of samples
                    and returns the loglikelihood of each sample.
            prior_pdf: **function**
                | A function that takes a numpy array of samples
                    and returns the prior logpdf of each sample.
            batch_size: **int**
                | The number of samples to draw at each iteration.
            sample_size: **int**
                | The number of samples to draw in total.
            logzero: **float**
                  The definition of zero for the loglikelihood function.

        returns:
        stats: **dict**
            | Dictionary containing useful statistics

        """
        xs = np.empty((sample_size, self.de.theta.shape[-1]))
        fs = np.empty(sample_size)
        gs = np.empty(sample_size)
        pis = np.empty(sample_size)

        n_todo = sample_size
        trials = 0

        with tqdm(total=sample_size) as pbar:
            while n_todo > 0:
                x = self.de.sample(batch_size).numpy()
                f = np.array(list(map(loglikelihood, x)))
                g = self.de.log_prob(x).numpy()
                in_bounds = np.logical_and(f >= logzero, g >= logzero)
                n_accept = x[in_bounds].shape[0]
                if n_accept <= n_todo:
                    xs[
                        sample_size - n_todo : sample_size - n_todo + n_accept
                    ] = x[in_bounds]
                    fs[
                        sample_size - n_todo : sample_size - n_todo + n_accept
                    ] = f[in_bounds]
                    gs[
                        sample_size - n_todo : sample_size - n_todo + n_accept
                    ] = g[in_bounds]
                    pis[
                        sample_size - n_todo : sample_size - n_todo + n_accept
                    ] = prior_pdf(x[in_bounds])
                    trials += batch_size
                else:
                    n_accept = n_todo
                    xs[sample_size - n_todo :] = x[in_bounds][:n_accept]
                    fs[sample_size - n_todo :] = f[in_bounds][:n_accept]
                    gs[sample_size - n_todo :] = g[in_bounds][:n_accept]
                    pis[sample_size - n_todo :] = prior_pdf(x[in_bounds])[
                        :n_accept
                    ]
                    last_index = in_bounds[-1]
                    trials += last_index + 1
                n_todo -= n_accept
                pbar.update(n_accept)
                if trials > 10 * sample_size:
                    raise ValueError(
                        "Too many unsuccessful trials, this typically indicates mismatch between flow and likelihood"
                    )

            weights = np.exp(fs + pis - gs)

            eff = np.sum(weights) ** 2 / np.sum(weights**2) / sample_size
            integral = sample_size / trials * weights.mean()
            log_integral = logsumexp(fs + pis - gs) - np.log(trials)
            # TODO this is the standard error function, we should just work logstderr directly
            stderr = np.sqrt(
                (np.sum(weights**2) / trials - integral**2) / (trials - 1)
            )
            log_stderr = np.log(stderr)

        stats = {
            "x": xs,
            "y": fs,
            "weights": weights,
            "efficiency": eff,
            "trials": trials,
            "log_integral": log_integral,
            "log_stderr": log_stderr,
        }
        return stats
