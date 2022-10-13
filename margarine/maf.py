import numpy as np
import tensorflow as tf
from tensorflow_probability import (bijectors as tfb, distributions as tfd)
from margarine.processing import _forward_transform, _inverse_transform
import pickle
import warnings
import margarine


class MAF(object):

    r"""

    This class is used to train, load and call instances of a bijector
    built from a series of autoregressive neural networks.

    **Parameters:**

        theta: **numpy array**
            | The samples from the probability distribution that we require the
                MAF to learn.

        weights: **numpy array**
            | The weights associated with the samples above.

    **kwargs:**

        number_networks: **int / default = 6**
            | The bijector is built by chaining a series of
                autoregressive neural
                networks together and this parameter is used to determine
                how many networks there are in the chain.

        learning_rate: **float / default = 1e-3**
            | The learning rate determines the 'step size' of the optimization
                algorithm used to train the MAF. Its value can effect the
                quality of emulation.

        hidden_layers: **list / default = [50, 50]**
            | The number of layers and number of nodes in each hidden layer for
                each neural network. The default is two hidden layers with
                50 nodes each and each network in the chain has the same hidden
                layer structure.

        theta_max: **numpy array**
            | The true upper limits of the priors used to generate the samples
                that we want the MAF to learn.

        theta_min: **numpy array**
            | As above but the true lower limits of the priors.


    **Attributes:**

    A list of some key attributes accessible to the user.

        maf: **Instance of tfd.TransformedDistribution**
            | By loading a trained instance of this class and accessing the
                ``maf`` (masked autoregressive flow) attribute, which is an
                instance of
                ``tensorflow_probability.distributions.TransformedDistribution``,
                the used can sample the trained MAF. e.g.

                .. code:: python

                    from ...maf import MAF

                    file = '/trained_maf.pkl'
                    bij = MAF.load(file)

                    samples = bij.maf.sample(1000)

                It can also be used to calculate log probabilities via

                .. code:: python

                    log_prob = bij.log_prob(samples)

                For more information on the attributes associated with
                ``tensorflow_probability.distributions.TransformedDistribution``
                see the tensorflow documentation.

        theta_max: **numpy array**
            | The true upper limits of the priors used to generate the samples
                that we want the MAF to learn. If theta_max is not supplied as a
                kwarg, then this is is an approximate estimate (for more info see
                the ... paper).

        theta_min: **numpy array**
            | As above but for the true lower limits of the priors. If theta_max is
                not supplied as a kwarg, then this is is an approximate estimate.

        loss_history: **list**
            | This list contains the value of the loss function at each epoch
                during training.

    """

    def __init__(self, theta, weights, **kwargs):
        self.n = (np.sum(weights)**2)/(np.sum(weights**2))
        self.sample_weights = weights

        theta_max = np.max(theta, axis=0)
        theta_min = np.min(theta, axis=0)
        a = ((self.n-2)*theta_max-theta_min)/(self.n-3)
        b = ((self.n-2)*theta_min-theta_max)/(self.n-3)
        self.theta_min = kwargs.pop('theta_min', b)
        self.theta_max = kwargs.pop('theta_max', a)

        self.theta = theta

        self.number_networks = kwargs.pop('number_networks', 6)
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.hidden_layers = kwargs.pop('hidden_layers', [50, 50])

        if type(self.number_networks) is not int:
            raise TypeError("'number_networks' must be an integer.")
        if type(self.learning_rate) not in [int, float]:
            raise TypeError("'learning_rate', must be a float.")
        if type(self.hidden_layers) is not list:
            raise TypeError("'hidden_layers' must be a list of integers.")
        else:
            for i in range(len(self.hidden_layers)):
                if type(self.hidden_layers[i]) is not int:
                    raise TypeError(
                        "One or more valus in 'hidden_layers'" +
                        "is not an integer.")

        self.mades = [tfb.AutoregressiveNetwork(params=2,
                      hidden_units=self.hidden_layers, activation='tanh',
                      input_order='random')
                      for _ in range(self.number_networks)]

        self.bij = tfb.Chain([
            tfb.MaskedAutoregressiveFlow(made) for made in self.mades])

        self.base = tfd.Blockwise(
            [tfd.Normal(loc=0, scale=1)
             for _ in range(len(theta_min))])
        self.maf = tfd.TransformedDistribution(self.base, bijector=self.bij)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate)

    def _train_step(self, x, w, loss_type, prior_phi=None, prior_weights=None):

        r"""
        This function is used to calculate the loss value at each epoch and
        adjust the weights and biases of the neural networks via the
        optimizer algorithm.
        """

        with tf.GradientTape() as tape:
            if loss_type == 'sum':
                    loss = -tf.reduce_sum(w*self.maf.log_prob(x))
            elif loss_type == 'mean':
                    loss = -tf.reduce_mean(w*self.maf.log_prob(x))
            gradients = tape.gradient(loss, self.maf.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients,
                    self.maf.trainable_variables))
            return loss

    def train(self, epochs=100, early_stop=False, loss_type='sum'):
        r"""

        This function is called to train the MAF once it has been
        initialised. For example

        .. code:: python

            from ...maf import MAF

            bij = MAF(theta, weights)
            bij.train()

        **Kwargs:**

            epochs: **int / default = 100**
                | The number of iterations to train the neural networks for.

            early_stop: **boolean / default = False**
                | Determines whether or not to implement an early stopping
                    algorithm or
                    train for the set number of epochs. If set to True then the
                    algorithm will stop training when the
                    fractional difference
                    between the current loss
                    and the average loss value over the
                    preceeding 10 epochs is < 1e-6.

        """
        if type(epochs) is not int:
            raise TypeError("'epochs' is not an integer.")
        if type(early_stop) is not bool:
            raise TypeError("'early_stop' must be a boolean.")

        phi = _forward_transform(self.theta, self.theta_min, self.theta_max)

        mask = np.isfinite(phi).all(axis=-1)
        phi = phi[mask, :]
        weights_phi = self.sample_weights[mask]
        weights_phi /= weights_phi.sum()

        phi = phi.astype('float32')
        self.phi = phi.copy()
        weights_phi = weights_phi.astype('float32')

        self.loss_history = []
        for i in range(epochs):
            loss = self._train_step(phi, weights_phi, loss_type).numpy()
            self.loss_history.append(loss)
            if early_stop:
                if len(self.loss_history) > 10:
                    delta = (
                        self.loss_history[-1] -
                        np.mean(self.loss_history[-11:-1])) \
                        / np.mean(self.loss_history[-11:-1])
                    if np.abs(delta) < 1e-6:
                        print('Early Stopped:' +
                              ' (Loss[-1] - Mean(Loss[-11:-1]))' +
                              '/Mean(Loss[-11:-1]) < 1e-6')
                        break

    def __call__(self, u):

        r"""

        This function is used when calling the MAF class to transform
        samples from the unit hypercube to samples on the MAF.

        **Parameters:**

            u: **numpy array**
                | Samples on the uniform hypercube.

        """

        x = _forward_transform(u)
        x = self.bij(x.astype(np.float32)).numpy()
        x = _inverse_transform(x, self.theta_min, self.theta_max)
        mask = np.isfinite(x).all(axis=-1)
        return x[mask, ...]

    def sample(self, length=1000):

        r"""

        This function is used to generate samples on the MAF via the
        MAF __call__ function.

        **Kwargs:**

            length: **int / default=1000**
                | This should be an integer and is used to determine how many
                    samples are generated when calling the MAF.

        """
        if type(length) is not int:
            raise TypeError("'length' must be an integer.")

        u = np.random.uniform(0, 1, size=(length, self.theta.shape[-1]))
        return self(u)

    def log_prob(self, params):

        """
        Function to caluclate the log-probability for a given MAF and
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

        transformed_x = _forward_transform(params, mins, maxs)

        transform_chain = tfb.Chain([
            tfb.Invert(tfb.NormalCDF()),
            tfb.Scale(1/(maxs - mins)), tfb.Shift(-mins)])

        def norm_jac(y):
            return transform_chain.inverse_log_det_jacobian(
                y, event_ndims=0).numpy()

        correction = norm_jac(transformed_x)
        logprob = (self.maf.log_prob(transformed_x).numpy() -
                   np.sum(correction, axis=-1)).astype(np.float64)

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
            prior_logprob = np.prod([1/(self.theta_max[i] - self.theta_min[i])
                                    for i in range(len(self.theta_min))])
        else:
            self.prior = margarine.maf.MAF(prior, prior_weights)
            self.prior.train()
            prior_logprob_func = self.prior.log_prob
            prior_logprob = prior_logprob_func(params)

        posterior_logprob = self.log_prob(params)

        loglike = posterior_logprob + logevidence - prior_logprob

        return loglike

    def save(self, filename):
        r"""

        This function can be used to save an instance of a trained MAF as
        a pickled class so that it can be loaded and used in differnt scripts.

        **Parameters:**

            filename: **string**
                | Path in which to save the pickled MAF.

        """
        nn_weights = [made.get_weights() for made in self.mades]
        with open(filename, 'wb') as f:
            pickle.dump([self.theta,
                         nn_weights,
                         self.sample_weights,
                         self.number_networks,
                         self.hidden_layers,
                         self.learning_rate], f)

    @classmethod
    def load(cls, filename):
        r"""

        This function can be used to load a saved MAF. For example

        .. code:: python

            from ...maf import MAF

            file = 'path/to/pickled/MAF.pkl'
            bij = MAF.load(file)

        **Parameters:**

            filename: **string**
                | Path to the saved MAF.

        """

        with open(filename, 'rb') as f:
            theta, nn_weights, \
                sample_weights, \
                number_networks, \
                hidden_layers, \
                learning_rate = pickle.load(f)

        bijector = cls(
            theta, sample_weights, number_networks=number_networks,
            learning_rate=learning_rate, hidden_layers=hidden_layers)
        bijector(np.random.uniform(0, 1, size=(len(theta), theta.shape[-1])))
        for made, nn_weights in zip(bijector.mades, nn_weights):
            made.set_weights(nn_weights)

        return bijector
