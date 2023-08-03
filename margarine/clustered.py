from scipy.special import logsumexp
from sklearn.cluster import KMeans
import margarine
from tensorflow import keras
from margarine.maf import MAF
import numpy as np
import tensorflow as tf
import warnings
import pickle


class clusterMAF():

    def __init__(self, theta, weights, **kwargs):
        self.number_networks = kwargs.pop('number_networks', 6)
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.hidden_layers = kwargs.pop('hidden_layers', [50, 50])
        self.cluster_labels = kwargs.pop('cluster_labels', None)
        self.cluster_number = kwargs.pop('cluster_number', None)
        self.theta = theta
        self.sample_weights = weights

        # needed for marginal stats
        self.n = np.sum(self.sample_weights)**2/ \
            np.sum(self.sample_weights**2)
        theta_max = np.max(self.theta, axis=0)
        theta_min = np.min(self.theta, axis=0)
        a = ((self.n-2)*theta_max-theta_min)/(self.n-3)
        b = ((self.n-2)*theta_min-theta_max)/(self.n-3)
        self.theta_min = kwargs.pop('theta_min', b)
        self.theta_max = kwargs.pop('theta_max', a)

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
            if self.cluster_number%1!=0:
                raise TypeError("'cluster_number' must be a whole number.")
        if self.cluster_labels is not None:
            if not isinstance(self.cluster_labels, (np.ndarray, list)):
                raise TypeError("'cluster_labels' must be an array " +
                                "or a list.")

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
            # need to ba able to pass number networks etc here
            self.flow.append(MAF(split_theta[i], split_sample_weights[i],
                                 number_networks=self.number_networks,
                                 learning_rate=self.learning_rate,
                                 hidden_layers=self.hidden_layers))
        
    def train(self, epochs=100, early_stop=False, loss_type='sum'):

        for i in range(len(self.flow)):
            self.flow[i].train(epochs=epochs, early_stop=early_stop, 
                               loss_type=loss_type)
    
    def log_prob(self, params):
        
        # currently working with numpy not tensorflow
        # nan repalacement with 0 difficult in tensorflow
        logprob = []
        for flow in self.flow:
            probs = flow.log_prob(params).numpy()
            for j in range(len(probs)):
                if np.isnan(probs[j]):
                    probs[j] = np.log(1e-300)
            logprob.append(probs)
        logprob = np.array(logprob)
        print(logprob.shape)
        logprob = logsumexp(logprob, axis=0)

        return logprob
    
    def log_like(self, params, logevidence, prior=None):
        
        if prior is None:
            warnings.warn('Assuming prior is uniform!')

            n = (np.sum(self.sample_weights)**2)/(np.sum(self.sample_weights**2))

            theta_max = np.max(self.theta, axis=0)
            theta_min = np.min(self.theta, axis=0)
            a = ((n-2)*theta_max-theta_min)/(n-3)
            b = ((n-2)*theta_min-theta_max)/(n-3)
            prior_logprob = np.log(
                np.prod([1/(a - b)
                         for i in range(len(b))]))
        else:
            prior_logprob = prior.log_prob(params).numpy()

        posterior_logprob = self.log_prob(params)

        loglike = posterior_logprob + prior_logprob - logevidence

        return loglike
    
    @tf.function(jit_compile=True)
    def __call__(self, u):

        len_thetas = [len(self.split_theta[i]) 
                      for i in range(len(self.split_theta))]
        probabilities = [len_thetas[i]/np.sum(len_thetas)
                            for i in range(len(self.split_theta))]
        options = np.arange(0, self.cluster_number)
        choice = np.random.choice(options,
                                    p=probabilities, size=len(u))

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
        u = np.random.uniform(size=(length, self.theta.shape[-1]))
        return self(u)
    
    def save(self, filename):
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

            from ...maf import MAF

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
                learning_rate, \
                cluster_number, \
                labels = data

        maf = cls(theta, sample_weights,
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