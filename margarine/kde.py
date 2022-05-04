import numpy as np
from scipy.stats import gaussian_kde, norm
from margarine.processing import _forward_transform, _inverse_transform
from scipy.optimize import root_scalar
import pickle


class KDE(object):

    r"""
    This class is used to generate a KDE given a weighted set of samples,
    generate samples from that KDE, transform samples on the
    hypercube into samples on the KDE and save and load the KDE model.

    **Parameters:**

        theta: **numpy array**
            | The samples from the probability distribution that we require the
                bijector to learn.

        weights: **numpy array**
            | The weights associated with the samples above.

    **kwargs:**

        bw_method: **str, scalar or callable**
            | The bandwidth for the KDE.

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
            | This is an approximate estimate of the true upper limits of the
                priors used to generate the samples that we want the
                bijector to learn (for more info see the ... paper).

        theta_min: **numpy array**
            | As above but an estimate of the true lower limits of the priors.

    """

    def __init__(self, theta, weights, **kwargs):

        self.theta = theta
        self.weights = weights

        self.n = (np.sum(weights)**2)/(np.sum(weights**2))
        theta_max = np.max(theta, axis=0)
        theta_min = np.min(theta, axis=0)
        a = ((self.n-2)*theta_max-theta_min)/(self.n-3)
        b = ((self.n-2)*theta_min-theta_max)/(self.n-3)
        self.theta_min = b
        self.theta_max = a

        self.bw_method = kwargs.pop('bw_method', 'silverman')

    def generate_kde(self):

        r"""
        Function noramlises the input data into a standard normal parameter
        space and then generates a weighted KDE.
        """

        phi = _forward_transform(self.theta, self.theta_min, self.theta_max)
        mask = np.isfinite(phi).all(axis=-1)
        phi = phi[mask, :]
        weights_phi = self.weights[mask]
        weights_phi /= weights_phi.sum()

        self.kde = gaussian_kde(
            phi.T, weights=self.weights, bw_method=self.bw_method)

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
                _inverse_transform(y, self.theta_min, self.theta_max))
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
        return _inverse_transform(x, self.theta_min, self.theta_max)

    def save(self, filename):

        r"""
        Function can be used to save an initalised version of the KDE class
        and it's assosiated generated KDE.

        **Parameters:**

            filename: **string**
                | Path in which to save the pickled KDE.
        """
        with open(filename, 'wb') as f:
            pickle.dump([self.theta, self.weights, self.kde], f)

    @classmethod
    def load(cls, filename):

        r"""

        This function can be used to load a saved KDE. For example

        .. code:: python

            from bayesstats.kde import KDE

            file = 'path/to/pickled/bijector.pkl'
            KDE_class = KDE.load(file)

        **Parameters:**

            filename: **string**
                | Path to the saved KDE.

        """

        with open(filename, 'rb') as f:
            theta, sample_weights, kde = pickle.load(f)

        kde_class = cls(theta, sample_weights)
        kde_class.kde = kde
        kde_class(np.random.uniform(0, 1, size=(2, theta.shape[-1])))

        return kde_class
