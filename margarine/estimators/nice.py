"""Implementation of the NICE estimator."""

import jax
import jax.numpy as jnp
import optax
import tqdm
from flax import nnx
from tensorflow_probability.substrates import jax as tfp

from margarine.base.baseflow import BaseDensityEstimator
from margarine.utils.utils import (
    approximate_bounds,
    forward_transform,
    inverse_transform,
    train_test_split,
)

tfb = tfp.bijectors
tfd = tfp.distributions


class NICE(BaseDensityEstimator, nnx.Module):
    """Implementation of the NICE architecture for density estimation.

    Details in https://arxiv.org/abs/1410.8516.
    """

    def __init__(
        self,
        theta: jnp.ndarray,
        weights: jnp.ndarray | None = None,
        theta_ranges: jnp.ndarray | None = None,
        in_size: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_coupling_layers: int = 4,
        nnx_rngs: dict | None = None,
    ) -> None:
        """Initialize the NICE estimator.

        Args:
            theta: Parameters of the density estimator.
            weights: Optional weights for the parameters.
            theta_ranges: Optional ranges for the parameters.
            in_size: Input size.
            hidden_size: Sizes of hidden layers.
            num_layers: Number of layers in each coupling network.
            num_coupling_layers: Number of coupling layers.
            nnx_rngs: Optional RNGs for Flax.
        """
        super().__init__()
        self.theta = nnx.data(theta)
        self.weights = nnx.data(weights)
        self.theta_ranges = nnx.data(theta_ranges)

        if self.weights is None:
            self.weights = jnp.ones(len(self.theta))
        if theta_ranges is None:
            self.theta_ranges = approximate_bounds(self.theta, self.weights)

        if nnx_rngs is None:
            nnx_rngs = nnx.Rngs(0)

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.nlayers = num_layers
        self.num_coupling_layers = num_coupling_layers

        self.pass_size = in_size // 2
        self.net_in_size = in_size - self.pass_size

        layers = nnx.List()
        layers.append(
            nnx.Linear(self.net_in_size, self.hidden_size, rngs=nnx_rngs)
        )
        for _ in range(self.nlayers):
            layers.append(
                nnx.Linear(self.hidden_size, self.hidden_size, rngs=nnx_rngs)
            )
            layers.append(lambda x: jax.nn.relu(x))

        layers.append(
            nnx.Linear(self.hidden_size, self.net_in_size, rngs=nnx_rngs)
        )
        self.mlp = nnx.List(
            [nnx.Sequential(*layers) for _ in range(self.num_coupling_layers)]
        )

        self.S = nnx.Param(jnp.zeros(in_size))

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """NICE forward pass.

        This is the forward pass of the NICE coupling layer.

        Args:
            x: Input samples.
        """
        for i in range(self.num_coupling_layers):
            if i % 2 == 0:
                x = jnp.flip(x, axis=-1)
            xa, xb = x[:, : self.pass_size], x[:, self.pass_size :]
            m = self.mlp[i](xb)
            xa = xa + m
            x = jnp.concat([xa, xb], axis=1)  # Concatenate correctly

        x = jnp.exp(self.S) * x

        if self.num_coupling_layers % 2 == 1:
            x = jnp.flip(x, axis=-1)

        return x

    def inverse(self, x: jnp.ndarray) -> jnp.ndarray:
        """NICE inverse pass.

        Args:
            x: Input samples.
        """
        if self.num_coupling_layers % 2 == 1:
            x = jnp.flip(x, axis=-1)

        x = x / jnp.exp(self.S)  # Undo scaling first

        for i in reversed(
            range(self.num_coupling_layers)
        ):  # Reverse the coupling order
            xa, xb = x[:, : self.pass_size], x[:, self.pass_size :]
            m = self.mlp[i](xb)
            xa = xa - m  # Subtract instead of add
            x = jnp.concat([xa, xb], axis=1)  # Concatenate correctly
            if i % 2 == 0:
                x = jnp.flip(x, axis=-1)
        return x

    def train(
        self,
        key: jnp.ndarray,
        learning_rate: float = 1e-4,
        epochs: int = 1000,
        patience: int = 50,
    ) -> None:
        """Train the NICE model.

        Args:
            key: JAX random key for data splitting.
            learning_rate: Learning rate for the optimizer.
            epochs: Number of training epochs.
            patience: Patience for early stopping.
        """

        @jax.jit
        def loss_function(
            model: BaseDensityEstimator,
            targets: jnp.ndarray,
            weights: jnp.ndarray,
        ) -> jnp.ndarray:
            """Loss function for training NICE model."""
            return -jnp.mean(weights * model.log_prob_under_NICE(targets))

        phi = forward_transform(
            self.theta, self.theta_ranges[0], self.theta_ranges[1]
        )
        weights = self.weights / jnp.sum(self.weights)

        key, subkey = jax.random.split(key)
        # need to split the data into training and validation sets
        (
            self.train_phi,
            self.test_phi,
            self.train_weights,
            self.test_weights,
        ) = train_test_split(
            phi,
            weights,
            subkey,
            test_size=0.2,
        )
        key, subkey = jax.random.split(key)
        self.test_phi, self.val_phi, self.test_weights, self.val_weights = (
            train_test_split(
                self.test_phi,
                self.test_weights,
                subkey,
                test_size=0.5,
            )
        )

        tx = optax.adam(learning_rate)
        optimizer = nnx.Optimizer(self, tx, wrt=nnx.Param)

        pbar = tqdm.tqdm(range(epochs), desc="Training")

        best_loss = jnp.inf
        best_model = nnx.state(self, nnx.Param)
        c = 0

        tl, vl = [], []
        for _ in pbar:
            loss = loss_function(self, self.train_phi, self.train_weights)
            tl.append(loss)
            grad = nnx.grad(loss_function)(
                self, self.train_phi, self.train_weights
            )
            optimizer.update(self, grad)

            val_loss = loss_function(self, self.val_phi, self.val_weights)
            vl.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = nnx.state(self, nnx.Param)
                best_epoch = _
                c = 0
            else:
                c += 1
                if c >= patience:
                    print(
                        f"Early stopping at epoch {_}, best "
                        + f"epoch was {best_epoch} with val loss {best_loss}"
                    )
                    break

            pbar.set_postfix(
                loss=f"{loss:.3f}",
                val_loss=f"{val_loss:.3f}",
                best_loss=f"{best_loss:.3f}",
            )

        if best_model is not None:
            nnx.update(self, best_model)

    def sample(self, key: jnp.ndarray, num_samples: int) -> jnp.ndarray:
        """Sample from the trained NICE model.

        Args:
            key: JAX random key.
            num_samples: Number of samples to draw.

        Returns:
            Samples drawn from the NICE model.
        """
        u = jax.random.uniform(key, shape=(num_samples, self.theta.shape[1]))
        samples = self(u)
        return samples

    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        """Transform samples from the unit hypercube.

        Args:
            u (jnp.ndarray): Samples from the unit hypercube.

        Returns:
            Log probabilities of the input data.
        """
        # from unit hypercube to standard normal space
        x = forward_transform(u, 0, 1)

        # from standard normal to target space
        x = self.inverse(x)

        # from gaussianized target space to original target space
        x = inverse_transform(x, self.theta_ranges[0], self.theta_ranges[1])
        return x

    def log_prob_under_NICE(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the log probability under the NICE model.

        Args:
            x (jnp.ndarray): Input data.

        Returns:
            Log probabilities of the input data.
        """
        # calcualte the actual log prob under the NICE model
        # assuming a standard normal base distribution
        z = self.forward(x)
        log_pz = -0.5 * jnp.sum(z**2 + jnp.log(2 * jnp.pi), axis=1)
        log_det_J = jnp.sum(self.S)
        log_prob_under_nice = log_pz + log_det_J
        return log_prob_under_nice

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the log probability of the input data.

        Args:
            x (jnp.ndarray): Input data.

        Returns:
            Log probabilities of the input data.
        """
        transformed_x = forward_transform(
            x, self.theta_ranges[0], self.theta_ranges[1]
        )
        transform_chain = tfb.Chain(
            [
                tfb.Invert(tfb.NormalCDF()),
                tfb.Scale(1 / (self.theta_ranges[0] - self.theta_ranges[1])),
                tfb.Shift(-self.theta_ranges[1]),
            ]
        )

        def norm_jac(y: jnp.ndarray) -> jnp.ndarray:
            """Calculate the normalising jacobian for the transformation."""
            return transform_chain.inverse_log_det_jacobian(y, event_ndims=0)

        correction = norm_jac(transformed_x).sum(axis=1)

        return self.log_prob_under_NICE(transformed_x) + correction

    def log_like(
        self,
        x: jnp.ndarray,
        logevidence: float,
        prior_density: jnp.ndarray | BaseDensityEstimator,
    ) -> jnp.ndarray:
        """Compute the marginal log-likelihood of given samples.

        Args:
            x: Samples for which to compute the log-likelihood.
            logevidence: Log-evidence term.
            prior_density: Prior density or density estimator.

        Returns:
            Log likelihoods of the input data.
        """
        if prior_density is isinstance(prior_density, BaseDensityEstimator):
            prior_density = prior_density.log_prob(x)

        return self.log_prob(x) + logevidence - prior_density

    def save(self, filename: str) -> None:
        """Save the trained NICE model to a file.

        Args:
            filename: Path to the file where the model will be saved.
        """
        # Placeholder for save logic
        pass

    @classmethod
    def load(cls, filename: str) -> "NICE":
        """Load a trained NICE model from a file.

        Args:
            filename: Path to the file from which the model will be loaded.

        Returns:
            Loaded NICE model.
        """
        # Placeholder for load logic
        pass
