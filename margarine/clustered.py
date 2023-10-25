from scipy.special import logsumexp
from sklearn.cluster import KMeans
from tensorflow import keras
from margarine.maf import MAF
import anesthetic
import numpy as np
import tensorflow as tf
import warnings
import pickle


class clusterMAF():

    r"""

    This class is used to train, load and call a piecewise normalising flow
    built from a set of masked autoregressive flows. The class
    is essentially a wrapper around the MAF class with some additional
    clustering functionality. This class has all the same
    functionality as the MAF class and can be used in the same way.

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

        cluster_labels: **list / default = None**
            | If clustering has been performed externally to margarine you can
                provide a list of labels for the samples theta. The labels
                should be integers from 0 to k corresponding to the cluster
                that each sample is in.

        cluster_number: **int / default = None**
            | If clustering has been performed externally to margarine you
                need to provide the number of clusters, k, alongside the
                cluster labels.

        parameters: **list of strings**
            | A list of the relevant parameters to train on. Only needed
                if theta is an anestehetic samples object. If not provided,
                all parameters will be used.

    """

    def __init__(self, theta, **kwargs):

        # Unpack kwargs
        self.number_networks = kwargs.pop('number_networks', 6)
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.hidden_layers = kwargs.pop('hidden_layers', [50, 50])
        self.cluster_labels = kwargs.pop('cluster_labels', None)
        self.cluster_number = kwargs.pop('cluster_number', None)
        self.parameters = kwargs.pop('parameters', None)

        # Avoid unintended side effects by copying theta
        theta = theta.copy()

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
        self.sample_weights = weights.copy()

        # needed for marginal stats
        self.n = np.sum(self.sample_weights)**2 / \
            np.sum(self.sample_weights**2)
        theta_max = np.max(self.theta, axis=0)
        theta_min = np.min(self.theta, axis=0)
        a = ((self.n-2)*theta_max-theta_min)/(self.n-3)
        b = ((self.n-2)*theta_min-theta_max)/(self.n-3)
        self.theta_min = kwargs.pop('theta_min', b)
        self.theta_max = kwargs.pop('theta_max', a)

        # Convert min and max to float 32 if needed
        if not isinstance(self.theta_min, tf.Tensor):
            self.theta_min = tf.convert_to_tensor(self.theta_min.copy(), dtype=tf.float32)
        if not isinstance(self.theta_max, tf.Tensor):
            self.theta_max = tf.convert_to_tensor(self.theta_max.copy(), dtype=tf.float32)

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

        mask = np.isfinite(theta).all(axis=-1)
        self.theta = self.theta[mask]
        self.sample_weights = self.sample_weights[mask]

        if self.cluster_number is not None:
            if self.cluster_labels is None:
                raise ValueError("'cluster_labels' should be provided if " +
                                 "'cluster_number' is specified.")
        else:
            if self.cluster_labels is not None:
                raise ValueError("'cluster_number' should be provided if " +
                                 "'cluster_labels' is specified.")

        if self.cluster_number is not None:
            if self.cluster_number % 1 != 0:
                raise TypeError("'cluster_number' must be a whole number.")
        if self.cluster_labels is not None:
            if not isinstance(self.cluster_labels, (np.ndarray, list)):
                raise TypeError("'cluster_labels' must be an array " +
                                "or a list.")

        if self.cluster_number is None:
            # this code currently performs the clustering with
            # kmeans but we could have anything here.
            from sklearn.metrics import silhouette_score
            ks = np.arange(2, 21)
            losses = []
            for k in ks:
                kmeans = KMeans(k, random_state=0, n_init='auto')
                labels = kmeans.fit(self.theta).predict(self.theta)
                losses.append(-silhouette_score(self.theta, labels))
            losses = np.array(losses)
            minimum_index = np.argmin(losses)
            self.cluster_number = ks[minimum_index]

            kmeans = KMeans(self.cluster_number, random_state=0, n_init='auto')
            self.cluster_labels = kmeans.fit(self.theta).predict(self.theta)
            self.custom_cluster = False
        else:
            self.custom_cluster = True

        if self.cluster_number == 20:
            warnings.warn("The number of clusters is 20. " +
                          "This is the maximum " +
                          "number of clusters that can be used. If you " +
                          "require more clusters, please specify the " +
                          "'cluster_number' kwarg. " +
                          "margarine will continue " +
                          "with 20 clusters.")

        if np.array(list(self.cluster_labels)).dtype == 'float':
            # convert cluster labels to integers
            self.cluster_labels = self.cluster_labels.astype(int)
        # count the number of times a cluster label appears in cluster_labels
        self.cluster_count = np.bincount(self.cluster_labels)

        if self.custom_cluster:
            if self.cluster_count.min() < 100:
                warnings.warn("One or more clusters are too small " +
                          "(n_cluster < 100). " +
                          "Since cluster_number was supplied margarine" +
                          "will continue but may crash.")
        else:
            # While loop to make sure clusters are not too small
            while self.cluster_count.min() < 100:
                warnings.warn("One or more clusters are too small " +
                            "(n_cluster < 100). " +
                            "Reducing the number of clusters by 1.")
                minimum_index -= 1
                self.cluster_number = ks[minimum_index]
                kmeans = KMeans(self.cluster_number, random_state=0, n_init='auto')
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

        split_theta = []
        split_sample_weights = []
        for i in range(self.cluster_number):
            split_theta.append(self.theta[self.cluster_labels == i])
            split_sample_weights.append(
                self.sample_weights[self.cluster_labels == i])

        self.split_theta = split_theta
        self.split_sample_weights = split_sample_weights

        self.flow = []
        for i in range(len(split_theta)):
            self.flow.append(MAF(split_theta[i], 
                                 weights=split_sample_weights[i],
                                 number_networks=self.number_networks,
                                 learning_rate=self.learning_rate,
                                 hidden_layers=self.hidden_layers,
                                 theta_min=self.theta_min,
                                 theta_max=self.theta_max))

    def train(self, epochs=100, early_stop=False, loss_type='sum'):

        r"""
        This function is called to train the clusterMAF once it has been
        initialised. It calls the `train()` function for each flow.

        .. code:: python

            from margarine.clustered import clusterMAF

            bij = clusterMAF(theta, weights=weights)
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

        for i in range(len(self.flow)):
            self.flow[i].train(epochs=epochs, early_stop=early_stop,
                               loss_type=loss_type)

    def log_prob(self, params):

        """
        Function to caluclate the log-probability for a given clusterMAF and
        set of parameters.

        While each density estimator has its own built in log probability
        function, a correction has to be applied for the transformation of
        variables that is used to improve accuracy when learning and we
        have to sum probabilities over the series of flows. The
        correction and the sum are implemented here.

        **Parameters:**

            params: **numpy array**
                | The set of samples for which to calculate the log
                    probability.

        """

        # currently working with numpy not tensorflow
        # nan repalacement with 0 difficult in tensorflow

        flow_weights = [np.sum(weights) for weights in
                        self.split_sample_weights]
        flow_weights = np.array(flow_weights)
        flow_weights = flow_weights / np.sum(flow_weights)

        logprob = []
        for flow, weight in zip(self.flow, flow_weights):
            flow_prob = flow.log_prob(
                tf.convert_to_tensor(params, dtype=tf.float32)).numpy()
            probs = flow_prob + np.log(weight)
            logprob.append(probs)
        logprob = np.array(logprob)
        logprob = logsumexp(logprob, axis=0)

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

            n = (np.sum(self.sample_weights)**2) / \
                (np.sum(self.sample_weights**2))

            theta_max = np.max(self.theta, axis=0)
            theta_min = np.min(self.theta, axis=0)
            a = ((n-2)*theta_max-theta_min)/(n-3)
            b = ((n-2)*theta_min-theta_max)/(n-3)
            prior_logprob = np.log(
                np.prod([1/(a - b)
                         for i in range(len(b))]))
        else:
            prior_logprob = prior_de.log_prob(params).numpy()

        posterior_logprob = self.log_prob(params)

        loglike = posterior_logprob + prior_logprob - logevidence

        return loglike

    #@tf.function(jit_compile=True)
    def __call__(self, u, seed=1420):

        r"""

        This function is used when calling the clusterMAF class to transform
        samples from the unit hypercube to samples on the clusterMAF.

        **Parameters:**

            u: **numpy array**
                | Samples on the uniform hypercube.
            
            seed: **int / default=1420**
                | Set the seed for the cluster choice.

        """

        flow_weights = [np.sum(weights) for weights in
                        self.split_sample_weights]
        flow_weights = np.array(flow_weights)
        probabilities = flow_weights / np.sum(flow_weights)
        options = np.arange(0, self.cluster_number)

        np.random.seed(int(round(u[0][0]*1000)))
        choice = np.random.choice(options,
                                  p=probabilities, size=len(u))
        np.random.seed(None)

        totals = [len(choice[choice == options[i]])
                  for i in range(len(options))]
        totals = np.hstack([0, np.cumsum(totals)])

        values = []
        for i in range(len(options)):
            x = self.flow[i](u[totals[i]:totals[i+1]])
            values.append(x)

        x = tf.concat(values, axis=0)
        return x

    def sample(self, length=1000):

        r"""

        This function is used to generate samples on the clusterMAF via the
        clusterMAF __call__ function.

        **Kwargs:**

            length: **int / default=1000**
                | This should be an integer and is used to determine how many
                    samples are generated when calling the clusterMAF.

        """

        u = np.random.uniform(size=(length, self.theta.shape[-1]))
        return self(u)

    def save(self, filename):

        r"""

        This function can be used to save an instance of
        a trained clusterMAF as
        a pickled class so that it can be loaded and used in differnt scripts.

        **Parameters:**

            filename: **string**
                | Path in which to save the pickled MAF.

        """

        nn_weights = {}
        for i in range(self.cluster_number):
            made = self.flow[i].mades
            w = [m.get_weights() for m in made]
            nn_weights[i] = w

        with open(filename, 'wb') as f:
            pickle.dump([self.theta,
                        nn_weights,
                        self.sample_weights,
                        self.number_networks,
                        self.hidden_layers,
                        self.learning_rate,
                        self.cluster_number,
                        self.cluster_labels], f)

    @classmethod
    def load(cls, filename):

        r"""

        This function can be used to load a saved MAF. For example

        .. code:: python

            from margarine.clustered import clusterMAF

            file = 'path/to/pickled/MAF.pkl'
            bij = clusterMAF.load(file)

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
                learning_rate, \
                cluster_number, \
                labels = data

        maf = cls(theta, weights=sample_weights,
                  number_networks=number_networks,
                  hidden_layers=hidden_layers,
                  learning_rate=learning_rate,
                  cluster_number=cluster_number,
                  cluster_labels=labels)
        maf(np.random.uniform(size=(1, theta.shape[-1])))
        for j in range(len(maf.flow)):
            for made, nnw in zip(maf.flow[j].mades, nn_weights[j]):
                made.set_weights(nnw)

        return maf
