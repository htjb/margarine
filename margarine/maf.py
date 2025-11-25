"""This module contains the MAF class used to build, train and call."""

import pickle
import warnings

import anesthetic
import numpy as np
import tensorflow as tf
import tqdm
from anesthetic.samples import Samples
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from margarine.processing import (
    _forward_transform,
    _inverse_transform,
    pure_tf_train_test_split,
)


class MAF:
    r"""The MAF class.

    This class is used to train, load and call instances of a bijector
    built from a series of autoregressive neural networks.
    """

    def __init__(
        self,
        theta: tf.Tensor | np.ndarray | Samples,
        **kwargs: dict,
    ) -> None:
        """Initialize the MAF class.

        Args:
            theta (tf.Tensor | np.ndarray | Samples): The
                samples from the
                probability distribution that the MAF should learn.
                Can be a numpy array or an
                anesthetic NestedSamples or MCMCSamples object.
            **kwargs: Additional keyword arguments.

        Keyword Args:
            weights (np.ndarray, optional): The weights associated with the
                samples. If an anesthetic NestedSamples or MCMCSamples object
                is passed, the weights are drawn from it.
                Defaults to np.ones(len(theta)).
            number_networks (int, optional): The number of
                autoregressive neural networks to chain together
                in the bijector. Defaults to 6.
            learning_rate (float, optional): The step size of the optimization
                algorithm used to train the MAF. Can affect the
                quality of emulation.
                Defaults to 1e-3.
            hidden_layers (list, optional): The number of layers
                and nodes in each hidden layer for each neural network.
                Each network in the chain has the same hidden layer structure.
                Defaults to [50, 50].
            activation_func (str, optional): The activation function keyword
                recognizable by TensorFlow. Defaults to 'tanh'.
            theta_max (np.ndarray, optional): The true upper
                limits of the priors used to generate the samples.
            theta_min (np.ndarray, optional): The true lower limits
                of the priors used to generate the samples.
            parameters (list of str, optional): The relevant
                parameters to train on. Only needed if theta is an
                anesthetic samples object. If not provided,
                all parameters will be used.

        Attributes:
            theta_max (np.ndarray): The true upper limits of the priors. If not
                supplied as a keyword argument, this is an approximate
                estimate.
            theta_min (np.ndarray): The true lower limits of the priors. If not
                supplied as a keyword argument, this is an approximate
                estimate.
            loss_history (list): The value of the loss function
                at each epoch during training.
        """
        self.number_networks = kwargs.pop("number_networks", 6)
        self.learning_rate = kwargs.pop("learning_rate", 1e-3)
        self.hidden_layers = kwargs.pop("hidden_layers", [50, 50])
        self.parameters = kwargs.pop("parameters", None)
        self.activation_func = kwargs.pop("activation_func", "tanh")

        # Avoids unintended side effects outside the class
        if not isinstance(theta, tf.Tensor):
            theta = theta.copy()
        else:
            theta = tf.identity(theta)

        if isinstance(
            theta,
            anesthetic.samples.NestedSamples | anesthetic.samples.MCMCSamples,
        ):
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
            weights = kwargs.pop("weights", np.ones(len(theta)))

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
            self.sample_weights, mask, axis=0
        )

        self.n = tf.math.reduce_sum(
            self.sample_weights
        ) ** 2 / tf.math.reduce_sum(self.sample_weights**2)

        theta_max = tf.math.reduce_max(self.theta, axis=0)
        theta_min = tf.math.reduce_min(self.theta, axis=0)
        a = ((self.n - 2) * theta_max - theta_min) / (self.n - 3)
        b = ((self.n - 2) * theta_min - theta_max) / (self.n - 3)
        self.theta_min = kwargs.pop("theta_min", b)
        self.theta_max = kwargs.pop("theta_max", a)

        # Convert to tensors if not already
        if not isinstance(self.theta_min, tf.Tensor):
            self.theta_min = tf.convert_to_tensor(
                self.theta_min.copy(), dtype=tf.float32
            )
        if not isinstance(self.theta_max, tf.Tensor):
            self.theta_max = tf.convert_to_tensor(
                self.theta_max.copy(), dtype=tf.float32
            )

        # Discard samples outside the prior range, provide a warning if any
        # are discarded
        mask = tf.math.reduce_all(
            (self.theta >= self.theta_min) & (self.theta <= self.theta_max),
            axis=-1,
        )
        if not tf.math.reduce_all(mask):
            warnings.warn(
                "Some samples are outside the user specified prior range "
                "and will be discarded! The specified range is likely smaller "
                "than the range covered by the samples."
            )
            self.theta = tf.boolean_mask(self.theta, mask, axis=0)
            self.sample_weights = tf.boolean_mask(
                self.sample_weights, mask, axis=0
            )

        if type(self.number_networks) is not int:
            raise TypeError("'number_networks' must be an integer.")
        if not isinstance(
            self.learning_rate,
            int | float | tf.keras.optimizers.schedules.LearningRateSchedule,
        ):
            raise TypeError(
                "'learning_rate', "
                + "must be an integer, float or tf.keras scheduler."
            )
        if type(self.hidden_layers) is not list:
            raise TypeError("'hidden_layers' must be a list of integers.")
        else:
            for i in range(len(self.hidden_layers)):
                if type(self.hidden_layers[i]) is not int:
                    raise TypeError(
                        "One or more values in 'hidden_layers'"
                        + "is not an integer."
                    )

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )

        self.gen_mades()

    def gen_mades(self) -> tuple[tfb.Bijector, tfd.TransformedDistribution]:
        """Generating the masked autoregressive flow."""
        self.mades = [
            tfb.AutoregressiveNetwork(
                params=2,
                hidden_units=self.hidden_layers,
                activation=self.activation_func,
                input_order="random",
            )
            for _ in range(self.number_networks)
        ]

        self.bij = tfb.Chain(
            [tfb.MaskedAutoregressiveFlow(made) for made in self.mades]
        )

        self.base = tfd.Blockwise(
            [tfd.Normal(loc=0, scale=1) for _ in range(len(self.theta_min))]
        )

        self.maf = tfd.TransformedDistribution(self.base, bijector=self.bij)

        return self.bij, self.maf

    def train(
        self,
        epochs: int = 100,
        patience: int | None = None,
        early_stop: bool = False,
        loss_type: str = "sum",
    ) -> None:
        r"""Train the MAF once it has been initialized.

        This method calls the internal `_training()` function
        to perform the training process.

        Example:
            >>> from margarine.maf import MAF
            >>> bij = MAF(theta, weights=weights)
            >>> bij.train()

        Keyword Args:
            epochs (int, optional): The number of iterations to
                train the neural networks for. Defaults to 100.
            patience (int | None, optional): The number of epochs with
                no improvement on the test loss before early
                stopping is triggered.
                Defaults to 2% of total requested epochs.
            early_stop (bool, optional): Whether to implement
                early stopping or train for the set number of epochs.
                If True, training stops when test loss
                has not improved for 2% of the requested epochs,
                and the model rolls back to the best performing version.
                Defaults to False.
            loss_type (str, optional): Whether to use the 'sum' or
                'mean' of the weighted log probabilities to
                calculate the loss function.
                Defaults to 'sum'.
        """
        if type(epochs) is not int:
            raise TypeError("'epochs' is not an integer.")
        if type(early_stop) is not bool:
            raise TypeError("'early_stop' must be a boolean.")
        if patience is not None and type(patience) is not int:
            raise TypeError("'patience' must be an integer or None.")

        self.epochs = epochs
        self.early_stop = early_stop
        self.loss_type = loss_type

        if patience is None:
            self.patience = round((self.epochs / 100) * 2)
        else:
            self.patience = patience

        self.maf = self._training(
            self.theta,
            self.patience,
            self.sample_weights,
            self.maf,
            self.theta_min,
            self.theta_max,
        )

    def _training(
        self,
        theta: tf.Tensor,
        patience: int,
        sample_weights: tf.Tensor,
        maf: tfd.TransformedDistribution,
        theta_min: float | tf.Tensor,
        theta_max: float | tf.Tensor,
    ) -> tfd.TransformedDistribution:
        """Training the masked autoregressive flow.

        This function performs the training of the MAF by splitting
        the data into training and testing sets, then iteratively
        updating the weights of the neural networks based on the
        calculated loss.

        Args:
            theta (tf.Tensor): The samples to train on.
            patience (int): The number of epochs with no improvement
                on the test loss before early stopping is triggered.
            sample_weights (tf.Tensor): The weights associated with the
                samples.
            maf (tfd.TransformedDistribution): The MAF to be trained.
            theta_min (float | tf.Tensor): The minimum values of the priors.
            theta_max (float | tf.Tensor): The maximum values of the priors.

        Returns:
            tfd.TransformedDistribution: The trained MAF.
        """
        phi = _forward_transform(theta, theta_min, theta_max)
        weights_phi = sample_weights / tf.reduce_sum(sample_weights)

        phi_train, phi_test, weights_phi_train, weights_phi_test = (
            pure_tf_train_test_split(phi, weights_phi, test_size=0.2)
        )

        self.loss_history = []
        self.test_loss_history = []
        c = 0
        for i in tqdm.tqdm(range(self.epochs)):
            loss = self._train_step(
                phi_train, weights_phi_train, self.loss_type, maf
            )
            self.loss_history.append(loss)

            self.test_loss_history.append(
                self._test_step(
                    phi_test, weights_phi_test, self.loss_type, maf
                )
            )

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
                    if c == patience:
                        print(
                            "Early stopped. Epochs used = "
                            + str(i)
                            + ". Minimum at epoch = "
                            + str(minimum_epoch)
                        )
                        return minimum_model
        return maf

    @tf.function(jit_compile=True)
    def _test_step(
        self,
        x: tf.Tensor,
        w: tf.Tensor,
        loss_type: str,
        maf: tfd.TransformedDistribution,
    ) -> tf.Tensor:
        r"""Calculate the test loss.

        This function is used to calculate the test loss value at each epoch
        for early stopping.

        Args:
            x (tf.Tensor): The test samples.
            w (tf.Tensor): The weights associated with the test samples.
            loss_type (str): The type of loss to calculate ('sum' or 'mean').
            maf (tfd.TransformedDistribution): The MAF being trained.

        Returns:
            tf.Tensor: The calculated loss value.
        """
        if loss_type == "sum":
            loss = -tf.reduce_sum(w * maf.log_prob(x))
        elif loss_type == "mean":
            loss = -tf.reduce_mean(w * maf.log_prob(x))
        return loss

    @tf.function(jit_compile=True)
    def _train_step(
        self,
        x: tf.Tensor,
        w: tf.Tensor,
        loss_type: str,
        maf: tfd.TransformedDistribution,
    ) -> tf.Tensor:
        r"""Calculate the training loss and apply gradients.

        This function is used to calculate the loss value at each epoch and
        adjust the weights and biases of the neural networks via the
        optimizer algorithm.

        Args:
            x (tf.Tensor): The training samples.
            w (tf.Tensor): The weights associated with the training samples.
            loss_type (str): The type of loss to calculate ('sum' or 'mean').
            maf (tfd.TransformedDistribution): The MAF being trained.

        Returns:
            tf.Tensor: The calculated loss value.
        """
        with tf.GradientTape() as tape:
            if loss_type == "sum":
                loss = -tf.reduce_sum(w * maf.log_prob(x))
            elif loss_type == "mean":
                loss = -tf.reduce_mean(w * maf.log_prob(x))
        gradients = tape.gradient(loss, maf.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, maf.trainable_variables))
        return loss

    @tf.function(jit_compile=True)
    def __call__(self, u: tf.Tensor | np.ndarray) -> tf.Tensor:
        r"""Transform samples from the unit hypercube to samples on the MAF.

        Args:
            u (tf.Tensor | np.ndarray): Samples from the unit hypercube
                to be transformed.

        Returns:
            tf.Tensor: The transformed samples.
        """
        if u.dtype != tf.float32:
            u = tf.cast(u, tf.float32)

        x = _forward_transform(u)
        x = self.bij(x)
        x = _inverse_transform(x, self.theta_min, self.theta_max)

        return x

    @tf.function(jit_compile=True)
    def sample(self, length: int = 1000) -> tf.Tensor:
        r"""Generate samples from the MAF.

        Args:
            length (int, optional): The number of samples to generate.
                Defaults to 1000.

        Returns:
            tf.Tensor: The generated samples.
        """
        if type(length) is not int:
            raise TypeError("'length' must be an integer.")

        u = tf.random.uniform((length, len(self.theta_min)))
        return self(u)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def log_prob(self, params: tf.Tensor | np.ndarray) -> tf.Tensor:
        """Caluclate the log-probability for a given set of parameters.

        While the density estimator has its own built in log probability
        function, a correction has to be applied for the transformation of
        variables that is used to improve accuracy when learning. The
        correction is implemented here.

        Args:
            params (tf.Tensor | np.ndarray): The set of samples for which to
                calculate the log probability.

        Returns:
            tf.Tensor: The log-probabilities of the provided samples.

        """
        # Enforce float32 dtype
        if params.dtype != tf.float32:
            params = tf.cast(params, tf.float32)

        def calc_log_prob(
            mins: float | int | tf.Tensor,
            maxs: float | int | tf.Tensor,
            maf: tfd.TransformedDistribution,
        ) -> tf.Tensor:
            """Function to calculate log-probability for a given MAF.

            Args:
                mins (float | int | tf.Tensor): Minimum prior values.
                maxs (float | int | tf.Tensor): Maximum prior values.
                maf (tfd.TransformedDistribution): The MAF to calculate
                    the log-probability for.

            Returns:
                tf.Tensor: The log-probabilities of the provided samples.
            """

            def norm_jac(y: tf.Tensor) -> tf.Tensor:
                return transform_chain.inverse_log_det_jacobian(
                    y, event_ndims=0
                )

            transformed_x = _forward_transform(params, mins, maxs)

            transform_chain = tfb.Chain(
                [
                    tfb.Invert(tfb.NormalCDF()),
                    tfb.Scale(1 / (maxs - mins)),
                    tfb.Shift(-mins),
                ]
            )

            correction = norm_jac(transformed_x)
            logprob = maf.log_prob(transformed_x) - tf.reduce_sum(
                correction, axis=-1
            )
            return logprob

        logprob = calc_log_prob(self.theta_min, self.theta_max, self.maf)

        return logprob

    def log_like(
        self,
        params: tf.Tensor | np.ndarray,
        logevidence: float,
        prior_de: tfd.TransformedDistribution = None,
    ) -> tf.Tensor:
        r"""Return the log-likelihood for a given set of parameters.

        It requires the logevidence from the original nested sampling run
        in order to do this and in the case that the prior is non-uniform
        a trained prior density estimator should be provided.

        Args:
            params (tf.Tensor | np.ndarray): The set of samples for which to
                calculate the log-likelihood.
            logevidence (float): The log-evidence from the original
                nested sampling run.
            prior_de (tfd.TransformedDistribution, optional): A trained
                density estimator for the prior. If the prior is uniform,
                this can be left as None. Defaults to None.

        Returns:
            tf.Tensor: The log-likelihoods of the provided samples.
        """
        if prior_de is None:
            warnings.warn("Assuming prior is uniform!")
            prior_logprob = tf.math.log(
                tf.math.reduce_prod(
                    [
                        1 / (self.theta_max[i] - self.theta_min[i])
                        for i in range(len(self.theta_min))
                    ]
                )
            )
        else:
            prior_logprob = self.prior_de.log_prob(params)

        posterior_logprob = self.log_prob(params)

        loglike = posterior_logprob + logevidence - prior_logprob

        return loglike

    def save(self, filename: str) -> None:
        r"""Save an instance of a trained MAF.

        Args:
            filename (str): Path to save the MAF to.
        """
        nn_weights = [made.get_weights() for made in self.mades]
        with open(filename, "wb") as f:
            pickle.dump(
                [
                    self.theta,
                    nn_weights,
                    self.sample_weights,
                    self.number_networks,
                    self.hidden_layers,
                    self.learning_rate,
                    self.theta_min,
                    self.theta_max,
                ],
                f,
            )

    @classmethod
    def load(cls, filename: str) -> "MAF":
        r"""Load a saved MAF from a file.

        Example:
            >>> from margarine.maf import MAF
            >>> file = 'path/to/pickled/MAF.pkl'
            >>> bij = MAF.load(file)

        Args:
            filename (str): Path to the saved MAF file.

        Returns:
            MAF: The loaded MAF object.
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)
            (
                theta,
                nn_weights,
                sample_weights,
                number_networks,
                hidden_layers,
                learning_rate,
                theta_min,
                theta_max,
            ) = data

        bijector = cls(
            theta,
            weights=sample_weights,
            number_networks=number_networks,
            learning_rate=learning_rate,
            hidden_layers=hidden_layers,
            theta_min=theta_min,
            theta_max=theta_max,
        )
        bijector(np.random.uniform(0, 1, size=(len(theta), theta.shape[-1])))
        for made, nn_weights in zip(bijector.mades, nn_weights):
            made.set_weights(nn_weights)

        return bijector
