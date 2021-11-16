import numpy as np
from scipy.stats import gaussian_kde
from bayesstats.processing import _forward_transform, _inverse_transform
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

        self.kde = gaussian_kde(phi.T, weights=self.weights, bw_method='silverman')
        return self.kde

    def __call__(self, length=1000):
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
        kde_class(len(theta))

        return kde_class
