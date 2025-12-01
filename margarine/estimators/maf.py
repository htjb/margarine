"""Implementation of Masked Autoregressive Flow (MAF) estimator."""

import jax
import jax.numpy as jnp
import optax
import tqdm
from flax import nnx
from tensorflow_probability.substrates import jax as tfp

from margarine.base.baseflow import BaseDensityEstimator
from margarine.utils.custom_layers import MaskedLinear
from margarine.utils.utils import (
    approximate_bounds,
    forward_transform,
    inverse_transform,
    train_test_split,
)

tfd = tfp.distributions
tfb = tfp.bijectors


class MAF(BaseDensityEstimator, nnx.Module):
    """Masked Autoregressive Flow (MAF) density estimator."""

    def __init__(
        self,
        theta: jnp.ndarray,
        weights: jnp.ndarray | None = None,
        theta_ranges: jnp.ndarray | None = None,
        in_size: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_made_networks: int = 5,
        nnx_rngs: dict | None = None,
        key: jnp.ndarray = jax.random.PRNGKey(0),
        permutations: str = "random",
    ) -> None:
        """Initialize MAF model."""
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
        self.num_layers = num_layers
        self.num_made_networks = num_made_networks

        self.mades = nnx.List()
        self.permutations = nnx.data([])
        for i in range(num_made_networks):
            key, subkey = jax.random.split(key)
            if permutations == "random":
                input_order = jax.random.permutation(
                    subkey, jnp.arange(self.in_size)
                )
            elif permutations == "reverse":
                if i % 2 == 0:
                    input_order = jnp.arange(self.in_size)[::-1]
                else:
                    input_order = jnp.arange(self.in_size)
            self.permutations.append(input_order)
            masks = nnx.data(self.create_masks(input_order))
            made = []
            made.append(
                MaskedLinear(
                    self.in_size,
                    self.hidden_size,
                    mask=masks.metadata["nnx_value"][0],
                    rngs=nnx_rngs,
                )
            )
            made.append(lambda x: jax.nn.gelu(x))
            for j in range(1, self.num_layers):
                made.append(
                    MaskedLinear(
                        self.hidden_size,
                        self.hidden_size,
                        mask=masks.metadata["nnx_value"][j],
                        rngs=nnx_rngs,
                    )
                )
                made.append(lambda x: jax.nn.gelu(x))
            made.append(
                MaskedLinear(
                    self.hidden_size,
                    self.in_size * 2,
                    mask=masks.metadata["nnx_value"][-1],
                    rngs=nnx_rngs,
                )
            )
            self.mades.append(nnx.Sequential(*made))

    def create_masks(self, input_order: jnp.ndarray) -> list:
        """Create masks for MADE networks in MAF.

        Args:
            input_order: Order of input dimensions.

        Returns:
            jnp.ndarray: Masks for MADE networks.
        """
        # create the masks
        masks = []

        # First hidden layer masks
        # degrees is the maximum input each hidden unit can depend on
        # so a hidden node with degree 0 doesn't see any inputs
        # but a hidden node with degree 1 can see input 0 etc
        degrees = (jnp.arange(self.hidden_size) % (self.in_size - 1)) + 1
        mask = input_order[None, :] < degrees[:, None]
        masks.append(mask)

        # Subsequent hidden layer masks
        for i in range(1, self.num_layers):
            prev_degrees = degrees
            degrees = jnp.arange(self.hidden_size) % (self.in_size - 1) + 1
            mask = prev_degrees[None, :] <= degrees[:, None]
            masks.append(mask)

        # Output layer mask - separate for shift and log_scale
        # Create degrees for outputs (shift and scale for each dimension)
        output_degrees = jnp.repeat(
            jnp.arange(self.in_size), 2
        )  # Repeat each degree twice

        # Connect output units to hidden units with appropriate dependencies
        mask = degrees[None, :] <= output_degrees[:, None]
        masks.append(mask)

        return masks

    def forward(
        self, x: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass through the MAF model.

        Args:
            x: Input data.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Transformed data,
            shifts, and log scales.
        """
        shifts, log_scales = [], []

        for i in range(self.num_made_networks):
            # MADE processes in natural order
            out = self.mades[i](x)
            shift, log_scale = out[:, : self.in_size], out[:, self.in_size :]
            log_scale = jnp.clip(log_scale, -5, 3)

            shifts.append(shift)
            log_scales.append(log_scale)

            # Transform
            x = (x - shift) * jnp.exp(-log_scale)

            # Permute for next layer
            x = x[:, self.permutations[i]]

        return x, jnp.stack(shifts), jnp.stack(log_scales)

    def inverse(self, z: jnp.ndarray) -> jnp.ndarray:
        """Inverse pass through the MAF model.

        Args:
            z: Input data.

        Returns:
            jnp.ndarray: Inversely transformed data.
        """
        for i in reversed(range(self.num_made_networks)):
            # Inverse permutation from previous layer
            inverse_order = jnp.argsort(self.permutations[i])
            z = z[:, inverse_order]

            # Generate sequentially
            x = jnp.zeros_like(z)
            for d in range(self.in_size):
                out = self.mades[i](x)
                shift, log_scale = (
                    out[:, : self.in_size],
                    out[:, self.in_size :],
                )
                log_scale = jnp.clip(log_scale, -5, 3)

                x = x.at[:, d].set(
                    z[:, d] * jnp.exp(log_scale[:, d]) + shift[:, d]
                )

        return x

    def log_prob_under_MAF(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute log-probability under the MAF model.

        Args:
            x: Input data.

        Returns:
            jnp.ndarray: Log-probabilities of the samples.
        """
        z, shifts, log_scales = self.forward(x)

        # Log det jacobian from all layers
        log_det_jacobian = -jnp.sum(log_scales, axis=(0, 2))

        # Base distribution log prob (standard normal)
        base_log_prob = -0.5 * jnp.sum(
            jnp.square(z), axis=-1
        ) - 0.5 * self.in_size * jnp.log(2 * jnp.pi)

        return base_log_prob + log_det_jacobian

    def train(
        self,
        key: jnp.ndarray,
        learning_rate: float = 1e-4,
        epochs: int = 1000,
        patience: int = 50,
    ) -> None:
        """Train the MAF model.

        Args:
            key: JAX random key for data splitting.
            learning_rate: Learning rate for the optimizer.
            epochs: Number of training epochs.
            patience: Patience for early stopping.
        """

        @jax.jit
        def loss_function(
            model: "MAF",
            targets: jnp.ndarray,
            weights: jnp.ndarray,
        ) -> jnp.ndarray:
            """Loss function for training MAF model.

            Args:
                model (NICE): NICE model.
                targets (jnp.ndarray): Samples from the target distribution
                    that have been passed through the forward_transform.
                weights (jnp.ndarray): Weights for the samples.

            Returns:
                jnp.ndarray: Computed loss.
            """
            return -jnp.mean(weights * model.log_prob_under_MAF(targets))

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
                loss=f"{loss:.3e}",
                val_loss=f"{val_loss:.3e}",
                best_loss=f"{best_loss:.3e}",
            )

        if best_model is not None:
            nnx.update(self, best_model)

    def sample(self, key: jnp.ndarray, num_samples: int) -> jnp.ndarray:
        """Generate samples from the MAF model.

        Args:
            key: JAX random key for sampling.
            num_samples: Number of samples to generate.

        Returns:
            jnp.ndarray: Generated samples as a JAX array.
        """
        u = jax.random.uniform(key, shape=(num_samples, self.theta.shape[1]))
        return self(u)

    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the MAF model at given points.

        Args:
            u: Samples from the unit hypercube.

        Returns:
            jnp.ndarray: samples from the MAF model.
        """
        # from unit hypercube to standard normal space
        x = forward_transform(u, 0, 1)

        # from standard normal to target space
        x = self.inverse(x)

        # from gaussianized target space to original target space
        x = inverse_transform(x, self.theta_ranges[0], self.theta_ranges[1])
        return x

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the log-probability of given samples.

        Args:
            x: Samples for which to compute the log-probability.

        Returns:
            jnp.ndarray: Log-probabilities of the samples.
        """
        # Implementation of log-probability computation goes here
        pass

    def log_like(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the log-likelihood of given samples.

        Args:
            x: Samples for which to compute the log-likelihood.

        Returns:
            jnp.ndarray: Log-likelihoods of the samples.
        """
        # Implementation of log-likelihood computation goes here
        pass

    def save(self, filepath: str) -> None:
        """Save the MAF model to a file.

        Args:
            filepath: Path to the file where the model will be saved.
        """
        # Implementation of saving logic goes here
        pass

    @classmethod
    def load(cls, filepath: str) -> "MAF":
        """Load a MAF model from a file.

        Args:
            filepath: Path to the file from which the model will be loaded.

        Returns:
            MAF: Loaded MAF model instance.
        """
        # Implementation of loading logic goes here
        pass
