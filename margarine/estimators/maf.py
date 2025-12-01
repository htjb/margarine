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
            input_order = jax.random.permutation(key, jnp.arange(self.in_size))
            self.permutations.append(input_order)
            masks = nnx.data(self.create_masks(input_order))
            made = nnx.List()
            made.append(
                MaskedLinear(
                    self.in_size,
                    self.hidden_size,
                    mask=masks.metadata["nnx_value"][0],
                    rngs=nnx_rngs,
                )
            )
            for j in range(1, self.num_layers):
                made.append(
                    MaskedLinear(
                        self.hidden_size,
                        self.hidden_size,
                        mask=masks.metadata["nnx_value"][j],
                        rngs=nnx_rngs,
                    )
                )
            made.append(
                MaskedLinear(
                    self.hidden_size,
                    self.in_size * 2,
                    mask=masks.metadata["nnx_value"][-1],
                    rngs=nnx_rngs,
                )
            )
            self.mades.append(made)

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
            x = x[:, self.permutations[i]]
            out = self.mades[i](x)
            shift, log_scale = out[:, 0, :], out[:, 1, :]
            shifts.append(shift)
            log_scales.append(log_scale)
            x = jnp.exp(-log_scale) * (x - shift)
        return x, jnp.stack(shifts), jnp.stack(log_scales)

    def inverse(self, x: jnp.ndarray) -> jnp.ndarray:
        """Inverse pass through the MAF model.

        Args:
            x: Input data.

        Returns:
            jnp.ndarray: Inversely transformed data.
        """
        # Go backwards through the flow
        for i in reversed(range(self.num_made_networks)):
            # Pass through MADE (x is already in correct order from
            # previous inverse perm)
            out = self.mades[i](x)
            shift, log_scale = (
                out[:, 0, :],
                out[:, 1, :],
            )  # Check your output shape!

            # Inverse affine transformation
            x = x * jnp.exp(log_scale) + shift

            # Inverse permutation
            inverse_order = jnp.argsort(self.permutations[i])
            x = x[:, inverse_order]

        return x

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
        # Implementation of the MAF transformation goes here
        pass

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
