"""The KDE class used to generate and work with Kernel Density Estimators."""

import pickle
import warnings

import anesthetic
import numpy as np
import tensorflow as tf
from scipy.optimize import root_scalar
from scipy.stats import gaussian_kde, norm
from tensorflow_probability import bijectors as tfb

import margarine
from margarine.processing import _forward_transform, _inverse_transform


class KDE:
    r"""Class used to generate and work with Kernel Density Estimators."""

    def __init__(
        self,
        theta: np.array
        | anesthetic.samples.NestedSample
        | anesthetic.samples.MCMCSamples,
        **kwargs: dict,
    ) -> None:
        r"""Kernel Density Estimation (KDE) class for weighted samples.

        This class generates a KDE from weighted samples,
        generates new samples from
        the KDE, transforms samples from the hypercube to the
        KDE distribution, and provides methods to save and load KDE models.

        Args:
            theta (np.ndarray | NestedSamples | MCMCSamples):
                The samples from the probability
                distribution to learn.
            **kwargs: Additional keyword arguments.

        Keyword Args:
            weights (np.ndarray, optional): The weights associated
                with the samples. If an anesthetic NestedSamples or MCMCSamples
                object is passed, the weights are drawn from it.
                Defaults to np.ones(len(theta)).
            bw_method (str | float | callable, optional): The
                bandwidth method for the KDE. Can be a string
                ('scott', 'silverman'), scalar, or callable.
            theta_max (np.ndarray, optional): The true upper
                limits of the priors used to generate the samples.
            theta_min (np.ndarray, optional): The true lower
                limits of the priors used to generate the samples.
            parameters (list of str, optional): The relevant
                parameters to use. Only needed if theta is an
                anesthetic samples object. If not provided, all
                parameters will be used.

        Attributes:
            kde (scipy.stats.gaussian_kde): The generated KDE object.
                Available after calling `generate_kde()`.
                Initialization and KDE generation are kept
                separate to allow effective model saving and loading.
            theta_max (np.ndarray): The true upper limits of the priors. If not
                supplied as a keyword argument, this is an
                approximate estimate.
            theta_min (np.ndarray): The true lower limits of the priors. If not
                supplied as a keyword argument, this is an
                approximate estimate.

        Example:
            >>> from margarine.kde import KDE
            >>> import numpy as np
            >>> theta = np.loadtxt('path/to/samples.txt')
            >>> weights = np.loadtxt('path/to/weights.txt')
            >>> kde_model = KDE(theta, weights=weights)
            >>> kde_model.generate_kde()
            >>> # Access the KDE via kde_model.kde
        """
        self.parameters = kwargs.pop("parameters", None)

        if isinstance(
            theta,
            anesthetic.samples.NestedSamples | anesthetic.samples.MCMCSamples,
        ):
            weights = theta.get_weights()
            if self.parameters:
                theta = theta[self.parameters].values
            else:
                if isinstance(theta, anesthetic.samples.NestedSamples):
                    self.parameters = theta.columns[:-3].values
                    theta = theta[theta.columns[:-3]].values
                else:
                    self.parameters = theta.columns[:-1].values
                    theta = theta[theta.columns[:-1]].values
        else:
            weights = kwargs.pop("weights", np.ones(len(theta)))

        self.theta = theta
        self.sample_weights = weights

        self.n = (np.sum(weights) ** 2) / (np.sum(weights**2))
        theta_max = np.max(theta, axis=0)
        theta_min = np.min(theta, axis=0)
        a = ((self.n - 2) * theta_max - theta_min) / (self.n - 3)
        b = ((self.n - 2) * theta_min - theta_max) / (self.n - 3)
        self.theta_min = kwargs.pop("theta_min", b)
        self.theta_max = kwargs.pop("theta_max", a)

        self.bw_method = kwargs.pop("bw_method", "silverman")

    def generate_kde(self) -> gaussian_kde:
        r"""Generates a weighted KDE."""
        theta = tf.convert_to_tensor(self.theta, dtype=tf.float32)
        theta_min = tf.convert_to_tensor(self.theta_min, dtype=tf.float32)
        theta_max = tf.convert_to_tensor(self.theta_max, dtype=tf.float32)

        phi = _forward_transform(theta, theta_min, theta_max).numpy()
        mask = np.isfinite(phi).all(axis=-1)
        phi = phi[mask, :]
        weights_phi = self.sample_weights[mask]
        weights_phi /= weights_phi.sum()

        self.kde = gaussian_kde(
            phi.T, weights=weights_phi, bw_method=self.bw_method
        )

        return self.kde

    def __call__(self, u: np.array) -> np.array:
        r"""Transform samples from the unit hypercube to samples on the KDE.

        Args:
            u (np.array): Samples from the unit hypercube
                to be transformed.

        Returns:
            np.array: The transformed samples.
        """
        # generate useful parameters for __call__ function to transform
        # hypercube into samples on the KDE.
        S = self.kde.covariance
        mu = self.kde.dataset.T
        steps, s = [], []
        for i in range(mu.shape[-1]):
            step = S[i, :i] @ np.linalg.inv(S[:i, :i])
            steps.append(step)
            s.append((S[i, i] - step @ S[:i, i]) ** 0.5)

        # transform samples from unit hypercube to kde
        transformed_samples = []
        for j in range(len(u)):
            x = u[j]
            y = np.zeros_like(x)
            for i in range(len(x)):
                m = mu[:, i] + steps[i] @ (y[:i] - mu[:, :i]).T
                y[i] = root_scalar(
                    lambda f: (
                        norm().cdf((f - m) / s[i]) * self.kde.weights
                    ).sum()
                    - x[i],
                    bracket=(mu[:, i].min() * 2, mu[:, i].max() * 2),
                    method="bisect",
                ).root
            transformed_samples.append(
                _inverse_transform(
                    tf.convert_to_tensor(y, dtype=tf.float32),
                    tf.convert_to_tensor(self.theta_min, dtype=tf.float32),
                    tf.convert_to_tensor(self.theta_max, dtype=tf.float32),
                )
            )
        transformed_samples = np.array(transformed_samples)
        return transformed_samples

    def sample(self, length: int = 1000) -> tf.Tensor:
        r"""Direct sampling from the KDE.

        Function can be used to generate samples from the KDE. It is much
        faster than the __call__ function but does not transform samples
        from the hypercube onto the KDE. It is however useful if we
        want to generate a large number of samples that can then be
        used to calulate the marginal statistics.

        Args:
            length (int): The number of samples to generate from the KDE.

        Returns:
            tf.Tensor: Samples generated from the KDE.
        """
        x = self.kde.resample(length).T
        return _inverse_transform(
            tf.convert_to_tensor(x, dtype="float32"),
            tf.convert_to_tensor(self.theta_min, dtype=tf.float32),
            tf.convert_to_tensor(self.theta_max, dtype=tf.float32),
        )

    def log_prob(self, params: np.array) -> np.array:
        """Caluclate the log-probability for a given set of parameters.

        While the density estimator has its own built in log probability
        function, a correction has to be applied for the transformation of
        variables that is used to improve accuracy when learning. The
        correction is implemented here.

        Args:
            params (np.array): The set of samples for which to calculate
                the log probability.

        Returns:
            np.array: The log-probability for the given samples.
        """
        mins = self.theta_min.astype(np.float32)
        maxs = self.theta_max.astype(np.float32)

        transformed_x = _forward_transform(
            tf.convert_to_tensor(params, dtype=tf.float32),
            tf.convert_to_tensor(mins, dtype=tf.float32),
            tf.convert_to_tensor(maxs, dtype=tf.float32),
        ).numpy()

        transform_chain = tfb.Chain(
            [
                tfb.Invert(tfb.NormalCDF()),
                tfb.Scale(1 / (maxs - mins)),
                tfb.Shift(-mins),
            ]
        )

        def norm_jac(y: np.array) -> np.array:
            """Calculate the normalising jacobian for the transformation."""
            return transform_chain.inverse_log_det_jacobian(
                y, event_ndims=0
            ).numpy()

        correction = norm_jac(transformed_x)
        if params.ndim == 1:
            logprob = (
                self.kde.logpdf(transformed_x.T) - np.sum(correction)
            ).astype(np.float64)
        else:
            logprob = (
                self.kde.logpdf(transformed_x.T) - np.sum(correction, axis=1)
            ).astype(np.float64)

        return logprob

    def log_like(
        self,
        params: np.ndarray,
        logevidence: float,
        prior: np.ndarray | None = None,
        prior_weights: np.array | None = None,
    ) -> np.ndarray:
        r"""Return the log-likelihood for a given set of parameters.

        It requires the logevidence from the original nested sampling run
        in order to do this and in the case that the prior is non-uniform
        prior samples should be provided.

        Args:
            params (np.ndarray): The set of samples for which to
                calculate the log probability.
            logevidence (float): Should be the log-evidence from
                the full nested samplingrun with nuisance parameters.

            prior (np.ndarray | None): An array of prior samples
                corresponding to the prior. Default assumption is that
                the prior is uniform which is required if you want to
                combine likelihoods from different experiments/data sets.
                In this case samples and prior samples should be
                reweighted prior to any training.
            prior_weights (np.array | None): The weights associated
                with the prior samples.

        Returns:
            np.ndarray: The log-likelihood for the given samples.
        """
        if prior is None:
            warnings.warn("Assuming prior is uniform!")
            prior_logprob = np.log(
                np.prod(
                    [
                        1 / (self.theta_max[i] - self.theta_min[i])
                        for i in range(len(self.theta_min))
                    ]
                )
            )
        else:
            self.prior = margarine.kde.KDE(prior, prior_weights)
            self.prior.generate_kde()
            prior_logprob_func = self.prior.log_prob
            prior_logprob = prior_logprob_func(params)

        posterior_logprob = self.log_prob(params)

        loglike = posterior_logprob + logevidence - prior_logprob

        return loglike

    def save(self, filename: str) -> None:
        r"""Save an instance of the KDE class and assosiated generated KDE.

        Args:
            filename (str): Path to save the KDE to.
        """
        with open(filename, "wb") as f:
            pickle.dump([self.theta, self.sample_weights, self.kde], f)

    @classmethod
    def load(cls, filename: str) -> "KDE":
        r"""Load a saved KDE from a file.

        Example:
            >>> from margarine.kde import KDE
            >>> file = 'path/to/pickled/bijector.pkl'
            >>> kde_model = KDE.load(file)

        Args:
            filename (str): Path to the saved KDE file.

        Returns:
            KDE: The loaded KDE object.
        """
        with open(filename, "rb") as f:
            theta, sample_weights, kde = pickle.load(f)

        kde_class = cls(theta, weights=sample_weights)
        kde_class.kde = kde
        kde_class(np.random.uniform(0, 1, size=(2, theta.shape[-1])))

        return kde_class
