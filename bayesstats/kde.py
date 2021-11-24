import numpy as np
from scipy.stats import gaussian_kde, norm
from bayesstats.processing import _forward_transform, _inverse_transform
from scipy.optimize import root_scalar
import pickle

class KDE(object):

    def __init__(self, theta, weights):
        self.theta = theta
        self.weights = weights

        self.n = (np.sum(weights)**2)/(np.sum(weights**2))
        theta_max = np.max(theta, axis=0)
        theta_min = np.min(theta, axis=0)
        a = ((self.n-2)*theta_max-theta_min)/(self.n-3)
        b = ((self.n-2)*theta_min-theta_max)/(self.n-3)
        self.theta_min = b
        self.theta_max = a

    def generate_kde(self):
        phi = _forward_transform(self.theta, self.theta_min, self.theta_max)
        mask = np.isfinite(phi).all(axis=-1)
        phi = phi[mask, :]
        weights_phi = self.weights[mask]
        weights_phi /= weights_phi.sum()

        self.kde = gaussian_kde(
            phi.T, weights=self.weights, bw_method='silverman')

        return self.kde

    def __call__(self, u):
        # generate useful parameters for __call__ function to transform
        # hypercube into samples on the KDE.
        S = self.kde.covariance
        mu = self.kde.dataset.T
        steps, s = [], []
        for i in range(mu.shape[-1]):
            step = S[i, :i] @ np.linalg.inv(S[:i,:i])
            steps.append(step)
            s.append((S[i,i] - step @ S[:i,i])**0.5)

        # transform samples from unit hypercube to kde
        transformed_samples = []
        for j in range(len(u)):
            x=u[j]
            y = np.zeros_like(x)
            for i in range(len(x)):
                m = mu[:,i] + steps[i] @ (y[:i] - mu[:,:i]).T
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
        x = self.kde.resample(length).T
        return _inverse_transform(x, self.theta_min, self.theta_max)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump([self.theta, self.weights, self.kde], f)

    @classmethod
    def load(cls, filename):

        with open(filename, 'rb') as f:
            theta, sample_weights, kde = pickle.load(f)

        kde_class = cls(theta, sample_weights)
        kde_class.kde = kde
        kde_class(np.random.uniform(0, 1, size=(2, theta.shape[-1])))

        return kde_class
