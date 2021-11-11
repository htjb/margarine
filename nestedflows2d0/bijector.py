import numpy as np
from scipy.stats import norm, uniform, cauchy, gennorm
import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
from tensorflow_probability import (bijectors as tfb, distributions as tfd)
from nestedflows2d0.processing import forward_transform, inverse_transform
import pickle
import pprint
import matplotlib.pyplot as plt

def load(filename):
    return Bijector.load(filename)

class Bijector(object):
    def __init__(self, theta_min, theta_max, weights, **kwargs):
        self.n = (np.sum(weights)**2)/(np.sum(weights**2))
        self.sample_weights = weights
        a = ((self.n-2)*theta_max-theta_min)/(self.n-3)
        b = ((self.n-2)*theta_min-theta_max)/(self.n-3)
        self.theta_min = a
        self.theta_max = b

        self.loc = 0
        self.scale = 1

        self.number_networks = kwargs.pop('number_networks', 6)
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.hidden_layers = kwargs.pop('hidden_layers', [50, 50])

        self.mades = [tfb.AutoregressiveNetwork(params=2,
                      hidden_units=self.hidden_layers, activation='tanh',
                      input_order='random')
                      for _ in range(self.number_networks)]

        self.bij = tfb.Chain([
            tfb.MaskedAutoregressiveFlow(made) for made in self.mades])

        self.base = tfd.Blockwise(
            [tfd.Normal(loc=self.loc, scale=self.scale)
             for _ in range(len(theta_min))])
        self.maf = tfd.TransformedDistribution(self.base, bijector=self.bij)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate)

        self.early_stop = kwargs.pop('early_stop', False)

    def _train_step(self, x, w):
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(w*self.maf.log_prob(x))
            gradients = tape.gradient(loss, self.maf.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.maf.trainable_variables))
            return loss

    def train(self, theta, weights, epochs=100):
        """ Train a neural network """
        phi = forward_transform(theta, self.theta_min, self.theta_max)

        mask = np.isfinite(phi).all(axis=-1)
        phi = phi[mask,:]
        weights_phi = weights[mask]
        weights_phi /= weights_phi.sum()

        phi = phi.astype('float32')
        self.phi = phi.copy()
        weights_phi = weights_phi.astype('float32')

        self.loss_history = []
        for i in range(epochs):
            loss=self._train_step(phi, weights_phi).numpy()
            self.loss_history.append(loss)
            print('Epoch: ' + str(i) + ' Loss: ' + str(loss))
            if self.early_stop:
                if len(self.loss_history) > 10:
                    delta = (self.loss_history[-1]-np.mean(self.loss_history[-11:-1])) \
                        /np.mean(self.loss_history[-11:-1])
                    if np.abs(delta) < 1e-6:
                        print('Early Stopped: (Loss[-1] - Mean(Loss[-11:-1]))/Mean(Loss[-11:-1]) < 1e-6')
                        print(np.mean(self.loss_history[-11:-1]), self.loss_history[-1])
                        print((self.loss_history[-1]- np.mean(self.loss_history[-11:-1]))
                            /np.mean(self.loss_history[-11:-1]))
                        break

    def __call__(self, u, prior_limits):
        x = forward_transform(u, prior_limits[0], prior_limits[1])
        x = self.bij(x.astype(np.float32)).numpy()
        x = inverse_transform(x, self.theta_min, self.theta_max)
        mask = np.isfinite(x).all(axis=-1)
        return x[mask, :]

    def save(self, filename):
        nn_weights = [made.get_weights() for made in self.mades]
        with open(filename,'wb') as f:
            pickle.dump([self.theta_min, self.theta_max, nn_weights, self.sample_weights], f)

    @classmethod
    def load(cls, filename):
        with open(filename,'rb') as f:
            theta_min, theta_max, nn_weights, sample_weights = pickle.load(f)

        bijector = cls(theta_min, theta_max, sample_weights)
        bijector(np.random.rand(len(theta_min)), np.array([[0]*5, [1]*5]))
        for made, nn_weights in zip(bijector.mades, nn_weights):
            made.set_weights(nn_weights)

        return bijector
