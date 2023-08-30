import tensorflow as tf
from tensorflow import keras
from tensorflow_probability import (bijectors as tfb, distributions as tfd)
from margarine.processing import (_forward_transform, _inverse_transform,
                                  pure_tf_train_test_split)
import numpy as np
import tqdm
import warnings
import pickle
import anesthetic


class MAF:

    r"""

    This class is used to train, load and call instances of a bijector
    built from a series of autoregressive neural networks.

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
            
        parameters: **list of strings**
            | A list of the relevant parameters to train on. Only needed
                if theta is an anestehetic samples object. If not provided,
                all parameters will be used.
            
    **Attributes:**

    A list of some key attributes accessible to the user.

        theta_max: **numpy array**
            | The true upper limits of the priors used to generate the
                samples that we want the MAF to learn. If theta_max is not
                supplied as a kwarg, then this is is an approximate estimate.

        theta_min: **numpy array**
            | As above but for the true lower limits of the priors. If
                theta_max is not supplied as a kwarg, then this is is an
                approximate estimate.

        loss_history: **list**
            | This list contains the value of the loss function at each epoch
                during training.

    """

    def __init__(self, theta, **kwargs):
        self.number_networks = kwargs.pop('number_networks', 6)
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.hidden_layers = kwargs.pop('hidden_layers', [50, 50])
        self.parameters = kwargs.pop('parameters', None)

        # Avoids unintended side effects outside the class
        if not isinstance(theta, tf.Tensor):
            theta = theta.copy()
        else:
            theta = tf.identity(theta)

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

        self.theta = tf.convert_to_tensor(theta, dtype=tf.float32)
        if not isinstance(weights, tf.Tensor):
            weights = tf.convert_to_tensor(weights.copy(), dtype=tf.float32)
        else:
            weights = tf.identity(weights)
        if weights.dtype != tf.float32:
            weights = tf.cast(weights, tf.float32)
        self.sample_weights = weights

        mask = np.isfinite(theta).all(axis=-1)
        self.theta = tf.boolean_mask(self.theta, mask, axis=0)
        self.sample_weights = tf.boolean_mask(
                                              self.sample_weights,
                                              mask, axis=0)

        self.n = tf.math.reduce_sum(self.sample_weights)**2 / \
            tf.math.reduce_sum(self.sample_weights**2)

        theta_max = tf.math.reduce_max(self.theta, axis=0)
        theta_min = tf.math.reduce_min(self.theta, axis=0)
        a = ((self.n-2)*theta_max-theta_min)/(self.n-3)
        b = ((self.n-2)*theta_min-theta_max)/(self.n-3)
        self.theta_min = kwargs.pop('theta_min', b)
        self.theta_max = kwargs.pop('theta_max', a)

        # Convert to tensors if not already
        if not isinstance(self.theta_min, tf.Tensor):
            self.theta_min = tf.convert_to_tensor(self.theta_min.copy(), dtype=tf.float32)
        if not isinstance(self.theta_max, tf.Tensor):
            self.theta_max = tf.convert_to_tensor(self.theta_max.copy(), dtype=tf.float32)

        # Discard samples outside the prior range, provide a warning if any
        # are discarded
        mask = tf.math.reduce_all((self.theta >= self.theta_min) & 
                                  (self.theta <= self.theta_max), axis=-1)
        if not tf.math.reduce_all(mask):
            warnings.warn('Some samples are outside the user specified prior range '
                          'and will be discarded! The specified range is likely smaller ' 
                          'than the range covered by the samples.')
            self.theta = tf.boolean_mask(self.theta, mask, axis=0)
            self.sample_weights = tf.boolean_mask(self.sample_weights, mask, axis=0)

        if type(self.number_networks) is not int:
            raise TypeError("'number_networks' must be an integer.")
        if not isinstance(self.learning_rate,
                          (int, float,
                           keras.optimizers.schedules.LearningRateSchedule)):
            raise TypeError("'learning_rate', " +
                            "must be an integer, float or keras scheduler.")
        if type(self.hidden_layers) is not list:
            raise TypeError("'hidden_layers' must be a list of integers.")
        else:
            for i in range(len(self.hidden_layers)):
                if type(self.hidden_layers[i]) is not int:
                    raise TypeError(
                        "One or more values in 'hidden_layers'" +
                        "is not an integer.")

        self.optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=self.learning_rate)

        self.gen_mades()

    def gen_mades(self):

        """Generating the masked autoregressive flow."""

        self.mades = [tfb.AutoregressiveNetwork(params=2,
                      hidden_units=self.hidden_layers, activation='tanh',
                      input_order='random')
                      for _ in range(self.number_networks)]

        self.bij = tfb.Chain([
            tfb.MaskedAutoregressiveFlow(made) for made in self.mades])

        self.base = tfd.Blockwise(
            [tfd.Normal(loc=0, scale=1)
             for _ in range(len(self.theta_min))])

        self.maf = tfd.TransformedDistribution(self.base, bijector=self.bij)

        return self.bij, self.maf

    def train(self, epochs=100, early_stop=False, loss_type='sum'):

        r"""

        This function is called to train the MAF once it has been
        initialised. It calls the `_training()` function.

        .. code:: python

            from margarine.maf import MAF

            bij = MAF(theta, weights=weights)
            bij.train()

        **Kwargs:**

            epochs: **int / default = 100**
                | The number of iterations to train the neural networks for.

            early_stop: **boolean / default = False**
                | Determines whether or not to implement an early stopping
                    algorithm or
                    train for the set number of epochs. If set to True then the
                    algorithm will stop training when test loss has not
                    improved for 2% of the requested epochs. At this point
                    margarine will roll back to the best model and return this
                    to the user.

            loss_type: **string / default = 'sum'**
                | Determines whether to use the sum or mean of the weighted
                    log probabilities to calculate the loss function.


        """

        if type(epochs) is not int:
            raise TypeError("'epochs' is not an integer.")
        if type(early_stop) is not bool:
            raise TypeError("'early_stop' must be a boolean.")

        self.epochs = epochs
        self.early_stop = early_stop
        self.loss_type = loss_type

        self.maf = self._training(self.theta,
                                  self.sample_weights, self.maf,
                                  self.theta_min, self.theta_max)

    def _training(self, theta, sample_weights, maf,
                  theta_min, theta_max):

        """Training the masked autoregressive flow."""

        phi = _forward_transform(theta, theta_min, theta_max)
        weights_phi = sample_weights/tf.reduce_sum(sample_weights)

        phi_train, phi_test, weights_phi_train, weights_phi_test = \
            pure_tf_train_test_split(phi, weights_phi, test_size=0.2)

        self.loss_history = []
        self.test_loss_history = []
        c = 0
        for i in tqdm.tqdm(range(self.epochs)):
            loss = self._train_step(phi_train,
                                    weights_phi_train,
                                    self.loss_type, maf)
            self.loss_history.append(loss)

            self.test_loss_history.append(self._test_step(phi_test,
                                          weights_phi_test,
                                          self.loss_type, maf))

            if self.early_stop:
                c += 1
                if i == 0:
                    minimum_loss = self.test_loss_history[-1]
                    minimum_epoch = i
                    minimum_model = None
                else:
                    if self.test_loss_history[-1] < minimum_loss:
                        minimum_loss = self.test_loss_history[-1]
                        minimum_epoch = i
                        minimum_model = maf.copy()
                        c = 0
                if minimum_model:
                    if c == round((self.epochs/100)*2):
                        print('Early stopped. Epochs used = ' + str(i) +
                              '. Minimum at epoch = ' + str(minimum_epoch))
                        return minimum_model
        return maf

    @tf.function(jit_compile=True)
    def _test_step(self, x, w, loss_type, maf):

        r"""
        This function is used to calculate the test loss value at each epoch
        for early stopping.
        """

        if loss_type == 'sum':
            loss = -tf.reduce_sum(w*maf.log_prob(x))
        elif loss_type == 'mean':
            loss = -tf.reduce_mean(w*maf.log_prob(x))
        return loss

    @tf.function(jit_compile=True)
    def _train_step(self, x, w, loss_type, maf):

        r"""
        This function is used to calculate the loss value at each epoch and
        adjust the weights and biases of the neural networks via the
        optimizer algorithm.
        """

        with tf.GradientTape() as tape:
            if loss_type == 'sum':
                loss = -tf.reduce_sum(w*maf.log_prob(x))
            elif loss_type == 'mean':
                loss = -tf.reduce_mean(w*maf.log_prob(x))
        gradients = tape.gradient(loss, maf.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients,
                maf.trainable_variables))
        return loss

    @tf.function(jit_compile=True)
    def __call__(self, u):
        r"""

        This function is used when calling the MAF class to transform
        samples from the unit hypercube to samples on the MAF.

        **Parameters:**

            u: **numpy array**
                | Samples on the uniform hypercube.

        """

        if u.dtype != tf.float32:
            u = tf.cast(u, tf.float32)

        x = _forward_transform(u)
        x = self.bij(x)
        x = _inverse_transform(x, self.theta_min, self.theta_max)

        return x

    @tf.function(jit_compile=True)
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

        u = tf.random.uniform((length, len(self.theta_min)))
        return self(u)

    @tf.function(jit_compile=True, reduce_retracing=True)
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

        # Enforce float32 dtype
        if params.dtype != tf.float32:
            params = tf.cast(params, tf.float32)

        def calc_log_prob(mins, maxs, maf):

            """Function to calculate log-probability for a given MAF."""

            def norm_jac(y):
                return transform_chain.inverse_log_det_jacobian(
                    y, event_ndims=0)

            transformed_x = _forward_transform(params, mins, maxs)

            transform_chain = tfb.Chain([
                tfb.Invert(tfb.NormalCDF()),
                tfb.Scale(1/(maxs - mins)), tfb.Shift(-mins)])

            correction = norm_jac(transformed_x)
            logprob = (maf.log_prob(transformed_x) -
                       tf.reduce_sum(correction, axis=-1))
            return logprob

        logprob = calc_log_prob(self.theta_min, self.theta_max, self.maf)

        return logprob

    def log_like(self, params, logevidence, prior_de=None):

        r"""
        This function should return the log-likelihood for a given set of
        parameters.

        It requires the logevidence from the original nested sampling run
        in order to do this and in the case that the prior is non-uniform
        a trained prior density estimator should be provided.

        **Parameters:**

            params: **numpy array**
                | The set of samples for which to calculate the log
                    probability.

            logevidence: **float**
                | Should be the log-evidence from the full nested sampling
                    run with nuisance parameters.

            prior_de: **margarine.maf.MAF / default=None**
                | If the prior is non-uniform then a trained prior density
                    estimator should be provided. Otherwise the prior
                    is assumed to be uniform and the prior probability
                    is calculated analytically from the minimum and maximum
                    values of the parameters.

        """

        if prior_de is None:
            warnings.warn('Assuming prior is uniform!')
            prior_logprob = tf.math.log(
                                        tf.math.reduce_prod(
                                            [1/(self.theta_max[i] -
                                                self.theta_min[i])
                                                for i in range(
                                                    len(self.theta_min))]))
        else:
            prior_logprob = self.prior_de.log_prob(params)

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
                        self.learning_rate,
                        self.theta_min,
                        self.theta_max], f)

    @classmethod
    def load(cls, filename):
        r"""

        This function can be used to load a saved MAF. For example

        .. code:: python

            from margarine.maf import MAF

            file = 'path/to/pickled/MAF.pkl'
            bij = MAF.load(file)

        **Parameters:**

            filename: **string**
                | Path to the saved MAF.

        """

        with open(filename, 'rb') as f:
            data = pickle.load(f)
            theta, nn_weights, \
                sample_weights, \
                number_networks, \
                hidden_layers, \
                learning_rate, theta_min, theta_max = data

        bijector = cls(
            theta, weights=sample_weights, number_networks=number_networks,
            learning_rate=learning_rate, hidden_layers=hidden_layers,
            theta_min=theta_min, theta_max=theta_max)
        bijector(
            np.random.uniform(0, 1, size=(len(theta), theta.shape[-1])))
        for made, nn_weights in zip(bijector.mades, nn_weights):
            made.set_weights(nn_weights)

        return bijector
