import numpy as np
from scipy.stats import gaussian_kde, norm
from margarine.processing import _forward_transform, _inverse_transform
from scipy.optimize import root_scalar
import tensorflow as tf
import pickle
import warnings
from tensorflow_probability import bijectors as tfb
import margarine
import anesthetic


class KDE(object):

    r"""
    This class is used to generate a KDE given a weighted set of samples,
    generate samples from that KDE, transform samples on the
    hypercube into samples on the KDE and save and load the KDE model.

    **Parameters:**

        theta: **numpy array or anesthetic.samples**
            | The samples from the probability distribution that we require the
                MAF to learn. This can either be a numpy array or an anesthetic
                NestedSamples or MCMCSamples object.

    **kwargs:**

        weights: **numpy array / default=np.ones(len(theta))**
            | The weights associated with the samples above. If an anesthetic
                NestedSamples or MCMCSamples object is passed the code
                draws the weights from this.

        bw_method: **str, scalar or callable**
            | The bandwidth for the KDE.

        theta_max: **numpy array**
            | The true upper limits of the priors used to generate the samples
                that we want the MAF to learn.

        theta_min: **numpy array**
            | As above but the true lower limits of the priors.

        parameters: **list of strings**
            | A list of the relevant parameters to train on. Only needed
                if theta is an anestehetic samples object. If not provided,
                all parameters will be used.
    **Attributes:**

    A list of some key attributes accessible to the user.

        kde: **Instance of scipy.stats.gaussian_kde**
            | Once the class has been initalised with a set of samples and
                their corresponding weights we can generate the kde using
                the following code

                .. code:: python

                    from bayesstats.kde import KDE
                    import numpy as np

                    theta = np.loadtxt('path/to/samples.txt')
                    weights = np.loadtxt('path/to/weights.txt')

                    KDE_class = KDE(theta, weights)
                    KDE_class.generate_kde()

                This is analogous to training a Normalising Flow (Bijector
                class). Once the KDE is generated it can be accessed via
                `KDE_class.kde`. Initialisation of the class and generation
                of the KDE are kept seperate to allow models to be saved and
                loaded effectively.

        theta_max: **numpy array**
            | The true upper limits of the priors used to generate the samples
                that we want the MAF to learn. If theta_max is not
                supplied as a kwarg, then this is is an approximate
                estimate.

        theta_min: **numpy array**
            | As above but for the true lower limits of the priors.
                If theta_max is not supplied as a kwarg, then this
                is an approximate estimate.

    """

    def __init__(self, theta, **kwargs):

        self.parameters = kwargs.pop('parameters', None)

        if isinstance(theta, 
                      (anesthetic.samples.NestedSamples, 
                       anesthetic.samples.MCMCSamples)):
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
            weights = kwargs.pop('weights', np.ones(len(theta)))

        self.theta = theta
        self.sample_weights = weights

        self.n = (np.sum(weights)**2)/(np.sum(weights**2))
        theta_max = np.max(theta, axis=0)
        theta_min = np.min(theta, axis=0)
        a = ((self.n-2)*theta_max-theta_min)/(self.n-3)
        b = ((self.n-2)*theta_min-theta_max)/(self.n-3)
        self.theta_min = kwargs.pop('theta_min', b)
        self.theta_max = kwargs.pop('theta_max', a)

        self.bw_method = kwargs.pop('bw_method', 'silverman')

    def generate_kde(self):

        r"""
        Function noramlises the input data into a standard normal parameter
        space and then generates a weighted KDE.
        """
        theta = tf.convert_to_tensor(self.theta, dtype=tf.float32)
        theta_min = tf.convert_to_tensor(self.theta_min, dtype=tf.float32)
        theta_max = tf.convert_to_tensor(self.theta_max, dtype=tf.float32)

        phi = _forward_transform(theta, theta_min, theta_max).numpy()
        mask = np.isfinite(phi).all(axis=-1)
        phi = phi[mask, :]
        weights_phi = self.sample_weights[mask]
        weights_phi /= weights_phi.sum()

        self.kde = gaussian_kde(
            phi.T, weights=weights_phi, bw_method=self.bw_method)

        return self.kde

    def __call__(self, u):

        r"""

        This function is used when calling the kde class to transform
        samples from the unit hypercube to samples on the kde.

        **Parameters:**

            u: **numpy array**
                | Samples on the uniform hypercube.

        """

        # generate useful parameters for __call__ function to transform
        # hypercube into samples on the KDE.
        S = self.kde.covariance
        mu = self.kde.dataset.T
        steps, s = [], []
        for i in range(mu.shape[-1]):
            step = S[i, :i] @ np.linalg.inv(S[:i, :i])
            steps.append(step)
            s.append((S[i, i] - step @ S[:i, i])**0.5)

        # transform samples from unit hypercube to kde
        transformed_samples = []
        for j in range(len(u)):
            x = u[j]
            y = np.zeros_like(x)
            for i in range(len(x)):
                m = mu[:, i] + steps[i] @ (y[:i] - mu[:, :i]).T
                y[i] = root_scalar(
                    lambda f:
                    (norm().cdf((f-m)/s[i])*self.kde.weights).sum()-x[i],
                    bracket=(mu[:, i].min()*2, mu[:, i].max()*2),
                    method='bisect').root
            transformed_samples.append(
                _inverse_transform(
                    tf.convert_to_tensor(y, dtype=tf.float32),
                    tf.convert_to_tensor(self.theta_min, dtype=tf.float32),
                    tf.convert_to_tensor(self.theta_max, dtype=tf.float32)))
        transformed_samples = np.array(transformed_samples)
        return transformed_samples

    def sample(self, length=1000):

        r"""

        Function can be used to generate samples from the KDE. It is much
        faster than the __call__ function but does not transform samples
        from the hypercube onto the KDE. It is however useful if we
        want to generate a large number of samples that can then be
        used to calulate the marginal statistics.

        **Kwargs:**

            length: **int / default=1000**
                | This should be an integer and is used to determine how many
                    samples are generated when calling the bijector.

        """
        x = self.kde.resample(length).T
        return _inverse_transform(
            tf.convert_to_tensor(x, dtype='float32'),
            tf.convert_to_tensor(self.theta_min, dtype=tf.float32),
            tf.convert_to_tensor(self.theta_max, dtype=tf.float32))

    def log_prob(self, params):

        """
        Function to caluclate the log-probability for a given KDE and
        set of parameters.

        While the density estimator has its own built in log probability
        function, a correction has to be applied for the transformation of
        variables that is used to improve accuracy when learning. The
        correction is implemented here.

        **Parameters:**

            params: **numpy array**
                | The set of samples for which to calculate the log
                    probability.

        """
        mins = self.theta_min.astype(np.float32)
        maxs = self.theta_max.astype(np.float32)

        transformed_x = _forward_transform(
            tf.convert_to_tensor(params, dtype=tf.float32),
            tf.convert_to_tensor(mins, dtype=tf.float32),
            tf.convert_to_tensor(maxs, dtype=tf.float32)).numpy()

        transform_chain = tfb.Chain([
            tfb.Invert(tfb.NormalCDF()),
            tfb.Scale(1/(maxs - mins)), tfb.Shift(-mins)])

        def norm_jac(y):
            return transform_chain.inverse_log_det_jacobian(
                y, event_ndims=0).numpy()

        correction = norm_jac(transformed_x)
        if params.ndim == 1:
            logprob = (self.kde.logpdf(transformed_x.T) -
                       np.sum(correction)).astype(np.float64)
        else:
            logprob = (self.kde.logpdf(transformed_x.T) -
                       np.sum(correction, axis=1)).astype(np.float64)

        return logprob

    def log_like(self, params, logevidence, prior=None, prior_weights=None):

        r"""
        This function should return the log-likelihood for a given set of
        parameters.

        It requires the logevidence from the original nested sampling run
        in order to do this and in the case that the prior is non-uniform
        prior samples should be provided.

        **Parameters:**

            params: **numpy array**
                | The set of samples for which to calculate the log
                    probability.

            logevidence: **float**
                | Should be the log-evidence from the full nested sampling
                    run with nuisance parameters.

            prior: **numpy array/default=None**
                | An array of prior samples corresponding to the prior. Default
                    assumption is that the prior is uniform which is
                    required if you want to combine likelihoods from different
                    experiments/data sets. In this case samples and prior
                    samples should be reweighted prior to any training.

        """

        if prior is None:
            warnings.warn('Assuming prior is uniform!')
            prior_logprob = np.log(np.prod(
                                   [1/(self.theta_max[i] - self.theta_min[i])
                                    for i in range(len(self.theta_min))]))
        else:
            self.prior = margarine.kde.KDE(prior, prior_weights)
            self.prior.generate_kde()
            prior_logprob_func = self.prior.log_prob
            prior_logprob = prior_logprob_func(params)

        posterior_logprob = self.log_prob(params)

        loglike = posterior_logprob + logevidence - prior_logprob

        return loglike

    def save(self, filename):

        r"""
        Function can be used to save an initalised version of the KDE class
        and it's assosiated generated KDE.

        **Parameters:**

            filename: **string**
                | Path in which to save the pickled KDE.
        """
        with open(filename, 'wb') as f:
            pickle.dump([self.theta, self.sample_weights, self.kde], f)

    @classmethod
    def load(cls, filename):

        r"""

        This function can be used to load a saved KDE. For example

        .. code:: python

            from margarine.kde import KDE

            file = 'path/to/pickled/bijector.pkl'
            KDE_class = KDE.load(file)

        **Parameters:**

            filename: **string**
                | Path to the saved KDE.

        """

        with open(filename, 'rb') as f:
            theta, sample_weights, kde = pickle.load(f)

        kde_class = cls(theta, weights=sample_weights)
        kde_class.kde = kde
        kde_class(np.random.uniform(0, 1, size=(2, theta.shape[-1])))

        return kde_class
