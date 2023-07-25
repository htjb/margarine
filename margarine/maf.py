import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_probability import (bijectors as tfb, distributions as tfd)
from margarine.processing import _forward_transform, _inverse_transform
from sklearn.cluster import KMeans
import pickle
import warnings
import margarine
from scipy.special import logsumexp
from sklearn.model_selection import train_test_split


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

        clustering: **Bool / default = False**
            | Whether or not to perform clustering as in
                https://arxiv.org/abs/2305.02930.

        cluster_labels: **list / default = None**
            | If clustering has been performed externally to margarine you can
                provide a list of labels for the samples theta. The labels
                should be integers from 0 to k corresponding to the cluster
                that each sample is in. Clustering is turned on if cluster
                labels are supplied.

        cluster_number: **int / default = None**
            | If clustering has been performed externally to margarine you
                need to provide the number of clusters, k, alongside the
                cluster labels.


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

    def __init__(self, theta, weights, **kwargs):
        self.number_networks = kwargs.pop('number_networks', 6)
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.hidden_layers = kwargs.pop('hidden_layers', [50, 50])
        self.clustering = kwargs.pop('clustering', False)
        self.cluster_labels = kwargs.pop('cluster_labels', None)
        self.cluster_number = kwargs.pop('cluster_number', None)

        if self.cluster_number is not None:
            if self.cluster_labels is None:
                raise ValueError("'cluster_labels' should be provided if " +
                                 "'cluster_number' is specified.")
        else:
            if self.cluster_labels is not None:
                raise ValueError("'cluster_number' should be provided if " +
                                 "'cluster_labels' is specified.")

        if self.cluster_labels is not None and self.cluster_number is not None:
            self.clustering = True

        self.n = (np.sum(weights)**2)/(np.sum(weights**2))
        self.sample_weights = weights

        theta_max = np.max(theta, axis=0)
        theta_min = np.min(theta, axis=0)
        a = ((self.n-2)*theta_max-theta_min)/(self.n-3)
        b = ((self.n-2)*theta_min-theta_max)/(self.n-3)
        self.theta_min = kwargs.pop('theta_min', b)
        self.theta_max = kwargs.pop('theta_max', a)

        self.theta = theta

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
                        "One or more valus in 'hidden_layers'" +
                        "is not an integer.")

        self.optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=self.learning_rate)

        if self.clustering is False:
            self.gen_mades()
            self.cluster_number = None
        if self.clustering is True:
            if self.cluster_number is not None:
                if self.cluster_number%1!=0:
                    raise TypeError("'cluster_number' must be a whole number.")
            if self.cluster_labels is not None:
                if not isinstance(self.cluster_labels, (np.ndarray, list)):
                    raise TypeError("'cluster_labels' must be an array " +
                                    "or a list.")
            self.clustering_call()

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

    def clustering_call(self):

        """Generating a piecewise masked autoregressive
        flow with clustering."""

        if self.cluster_number is None:
            from sklearn.metrics import silhouette_score
            ks = np.arange(2, 21)
            losses = []
            for k in ks:
                kmeans = KMeans(k, random_state=0)
                labels = kmeans.fit(self.theta).predict(self.theta)
                losses.append(-silhouette_score(self.theta, labels))
            losses = np.array(losses)
            minimum_index = np.argmin(losses)
            self.cluster_number = ks[minimum_index]

            kmeans = KMeans(self.cluster_number, random_state=0)
            self.cluster_labels = kmeans.fit(self.theta).predict(self.theta)
        
        if self.cluster_number == 20:
            warnings.warn("The number of clusters is 20. This is the maximum "+
                            "number of clusters that can be used. If you " +
                            "require more clusters, please specify the " +
                            "'cluster_number' kwarg. margarine will continue "+
                            "with 20 clusters.")
        
        if np.array(list(self.cluster_labels)).dtype == 'float':
            # convert cluster labels to integers
            self.cluster_labels = self.cluster_labels.astype(int)
        # count the number of times a cluster label appears in cluster_labels
        self.cluster_count = np.bincount(self.cluster_labels)
        # While loop to make sure clusters are not too small
        while self.cluster_count.min() < 100:
            warnings.warn("One or more clusters are too small " +
                          "(n_cluster < 100). " +
                          "Reducing the number of clusters by 1.")
            minimum_index -= 1
            self.cluster_number = ks[minimum_index]
            kmeans = KMeans(self.cluster_number, random_state=0)
            self.cluster_labels = kmeans.fit(self.theta).predict(self.theta)
            self.cluster_count = np.bincount(self.cluster_labels)
            if self.cluster_number == 2:
                # break if two clusters
                warnings.warn("The number of clusters is 2. This is the " +
                            "minimum number of clusters that can be used. " +
                            "Some clusters may be too small and the " +
                            "train/test split may fail." +
                            "Try running without clusting. ")
                break

        self.n, split_theta, self.new_theta_max = [], [], []
        self.new_theta_min, split_sample_weights = [], []
        self.bij, self.maf, self.mades = [], [], []
        for i in range(self.cluster_number):
            theta = self.theta[self.cluster_labels == i]
            split_sample_weights.append(
                self.sample_weights[self.cluster_labels == i])

            self.n.append(
                (np.sum(self.sample_weights[self.cluster_labels == i])**2) /
                (np.sum(self.sample_weights[self.cluster_labels == i]**2)))

            if not any(isinstance(tm, list) for tm in self.theta_max):
                new_theta_max = np.max(theta, axis=0)
                new_theta_min = np.min(theta, axis=0)
                a = ((self.n[-1]-2)*new_theta_max-new_theta_min)/(self.n[-1]-3)
                b = ((self.n[-1]-2)*new_theta_min-new_theta_max)/(self.n[-1]-3)
                self.new_theta_min.append(b)
                self.new_theta_max.append(a)

            split_theta.append(theta)

            self.mades.append(
                [tfb.AutoregressiveNetwork(
                            params=2,
                            hidden_units=self.hidden_layers, activation='tanh',
                            input_order='random')
                    for _ in range(self.number_networks)])

            self.bij.append(tfb.Chain([
                tfb.MaskedAutoregressiveFlow(made)
                for made in self.mades[-1]]))

            self.base = tfd.Blockwise(
                [tfd.Normal(loc=0, scale=1)
                    for _ in range(self.theta.shape[-1])])

            self.maf.append(tfd.TransformedDistribution(
                self.base, bijector=self.bij[-1]))
        self.theta = split_theta
        self.sample_weights = split_sample_weights
        if self.new_theta_max != []:
            self.theta_max = self.new_theta_max
            self.theta_min = self.new_theta_min

    def train(self, epochs=100, early_stop=False, loss_type='sum'):
        r"""

        This function is called to train the MAF once it has been
        initialised. It calls `_training()` for each MAF whether there
        is only one or whether clustering is turned on. For example

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
                    algorithm will stop training when test loss has not
                    improved for 2% of the requested epochs. At this point
                    margarine will roll back to the best model and return this
                    to the user.

        """
        if type(epochs) is not int:
            raise TypeError("'epochs' is not an integer.")
        if type(early_stop) is not bool:
            raise TypeError("'early_stop' must be a boolean.")

        self.epochs = epochs
        self.early_stop = early_stop
        self.loss_type = loss_type

        if self.cluster_number is not None:
            for i in range(len(self.theta)):
                self.maf[i] = self._training(self.theta[i],
                                             self.sample_weights[i],
                                             self.maf[i], self.theta_min[i],
                                             self.theta_max[i])
        else:
            self.maf = self._training(self.theta,
                                      self.sample_weights, self.maf,
                                      self.theta_min, self.theta_max)

    def _training(self, theta, sample_weights, maf,
                  theta_min, theta_max):

        """Function to perform the training of each MAF."""

        phi = _forward_transform(theta, theta_min, theta_max).numpy()

        mask = np.isfinite(phi).all(axis=-1)
        phi = phi[mask, :]
        weights_phi = sample_weights[mask]
        weights_phi /= weights_phi.sum()

        phi = phi.astype('float32')
        self.phi = phi.copy()
        weights_phi = weights_phi.astype('float32')

        phi_train, phi_test, weights_phi_train, weights_phi_test = \
            train_test_split(phi, weights_phi, test_size=0.2)

        self.loss_history = []
        self.test_loss_history = []
        c = 0
        for i in range(self.epochs):
            loss = self._train_step(phi_train,
                                    weights_phi_train,
                                    self.loss_type, maf).numpy()
            self.loss_history.append(loss)

            if self.loss_type == 'sum':
                loss_test = -tf.reduce_sum(
                    weights_phi_test*maf.log_prob(phi_test))
            elif self.loss_type == 'mean':
                loss_test = -tf.reduce_mean(
                    weights_phi_test*maf.log_prob(phi_test))

            self.test_loss_history.append(loss_test)

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

    def __call__(self, u):

        r"""

        This function is used when calling the MAF class to transform
        samples from the unit hypercube to samples on the MAF.

        **Parameters:**

            u: **numpy array**
                | Samples on the uniform hypercube.

        """

        @tf.function(jit_compile=True)
        def call_bij(bij, u, min=self.theta_min, max=self.theta_max):
            x = _forward_transform(u)
            x = bij(x)
            x = _inverse_transform(x, min, max)
            return x


        if self.cluster_number is not None:
            len_thetas = [len(self.theta[i]) for i in range(len(self.theta))]
            probabilities = [len_thetas[i]/np.sum(len_thetas)
                             for i in range(len(self.theta))]
            options = np.arange(0, self.cluster_number)
            choice = np.random.choice(options,
                                      p=probabilities, size=len(u))

            totals = [len(choice[choice == options[i]])
                      for i in range(len(options))]
            totals = np.hstack([0, np.cumsum(totals)])

            values = []
            for i in range(len(options)):
                x = call_bij(self.bij[i], u, min=self.theta_min[i], max=self.theta_max[i]).numpy()
                values.append(x)

            x = np.concatenate(values)
        else:
            x = call_bij(self.bij, u).numpy()

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

        if self.cluster_number is not None:
            u = np.random.uniform(0, 1, size=(length, self.theta[0].shape[-1]))
        else:
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

        def calc_log_prob(mins, maxs, maf):

            """Function to calculate log-probability for a given MAF."""

            def norm_jac(y):
                return transform_chain.inverse_log_det_jacobian(
                    y, event_ndims=0).numpy()

            transformed_x = _forward_transform(params, mins, maxs).numpy()

            transform_chain = tfb.Chain([
                tfb.Invert(tfb.NormalCDF()),
                tfb.Scale(1/(maxs - mins)), tfb.Shift(-mins)])

            correction = norm_jac(transformed_x)
            logprob = (maf.log_prob(transformed_x).numpy() -
                       np.sum(correction, axis=-1)).astype(np.float64)
            return logprob

        if self.clustering is True:
            logprob = []
            for i in range(len(self.theta_min)):
                mins = self.theta_min[i].astype(np.float32)
                maxs = self.theta_max[i].astype(np.float32)
                probs = calc_log_prob(mins, maxs, self.maf[i])
                for j in range(len(probs)):
                    if np.isnan(probs[j]):
                        probs[j] = np.log(1e-300)
                logprob.append(probs)
            logprob = np.array(logprob)
            logprob = logsumexp(logprob, axis=0)
        else:
            mins = self.theta_min.astype(np.float32)
            maxs = self.theta_max.astype(np.float32)
            logprob = calc_log_prob(mins, maxs, self.maf)

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

            prior_weights: **numpy array/default=None**
                | Weights to go with the prior samples.

        """

        if prior is None:
            warnings.warn('Assuming prior is uniform!')
            prior_logprob = np.log(
                np.prod([1/(self.theta_max[i] - self.theta_min[i])
                         for i in range(len(self.theta_min))]))
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

        This function can be used to save an instance of a trained MAF
        or piecewise MAF as
        a pickled class so that it can be loaded and used in differnt scripts.

        **Parameters:**

            filename: **string**
                | Path in which to save the pickled MAF.

        """
        if self.cluster_number is None:
            nn_weights = [made.get_weights() for made in self.mades]
        else:
            nn_weights = {}
            for i in range(self.cluster_number):
                made = self.mades[i]
                w = [m.get_weights() for m in made]
                nn_weights[i] = w
        with open(filename, 'wb') as f:
            if self.clustering is True:
                pickle.dump([self.theta,
                            nn_weights,
                            self.sample_weights,
                            self.number_networks,
                            self.hidden_layers,
                            self.learning_rate,
                            self.theta_min,
                            self.theta_max,
                            self.cluster_number,
                            self.cluster_labels], f)
            else:
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

            from ...maf import MAF

            file = 'path/to/pickled/MAF.pkl'
            bij = MAF.load(file)

        **Parameters:**

            filename: **string**
                | Path to the saved MAF.

        """

        with open(filename, 'rb') as f:
            data = pickle.load(f)
            try:
                theta, nn_weights, \
                    sample_weights, \
                    number_networks, \
                    hidden_layers, \
                    learning_rate, theta_min, theta_max = data
                cluster_number = None
            except Exception:
                theta, nn_weights, \
                    sample_weights, \
                    number_networks, \
                    hidden_layers, \
                    learning_rate, theta_min, theta_max, \
                    cluster_number, \
                    labels = data

        if cluster_number is None:
            bijector = cls(
                theta, sample_weights, number_networks=number_networks,
                learning_rate=learning_rate, hidden_layers=hidden_layers,
                theta_min=theta_min, theta_max=theta_max)
            bijector(
                np.random.uniform(0, 1, size=(len(theta), theta.shape[-1])))
            for made, nn_weights in zip(bijector.mades, nn_weights):
                made.set_weights(nn_weights)
        else:
            # have to sort labels because theta is reordered when
            # it gets concatenated
            labels.sort()
            theta = np.concatenate(theta)
            sample_weights = np.concatenate(sample_weights)
            bijector = cls(
                theta, sample_weights, number_networks=number_networks,
                learning_rate=learning_rate, hidden_layers=hidden_layers,
                clustering=True,
                cluster_number=cluster_number, cluster_labels=labels,
                theta_min=theta_min, theta_max=theta_max)
            bijector(
                np.random.uniform(0, 1, size=(len(theta), theta.shape[-1])))
            for i in range(cluster_number):
                mades = bijector.mades[i]
                weights = nn_weights[i]
                for m, w in zip(mades, weights):
                    m.set_weights(w)

        return bijector
