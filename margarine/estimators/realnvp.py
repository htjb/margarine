"""Implementation of the RealNVP estimator."""

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


class RealNVP(BaseDensityEstimator, nnx.Module):
    """Implementation of the RealNVP architecture for density estimation.

    Details in https://arxiv.org/abs/1605.08803.
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
        key: jnp.ndarray = jax.random.PRNGKey(0),
    ) -> None:
        """Initialize the RealNVP estimator.

        Args:
            theta: Parameters of the density estimator.
            weights: Optional weights for the parameters.
            theta_ranges: Optional ranges for the parameters.
            in_size: Input size.
            hidden_size: Size of hidden layers.
            num_layers: Number of layers in each coupling network.
            num_coupling_layers: Number of coupling layers.
            nnx_rngs: Optional RNGs for Flax.
            key: JAX random key for permutations.
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

        # like glorot normal but with 0.1 stddev
        kernel_init = nnx.initializers.variance_scaling(
            0.1, "fan_avg", "truncated_normal"
        )

        additive_layers = nnx.List()
        additive_layers.append(
            nnx.Linear(
                self.net_in_size,
                self.hidden_size,
                kernel_init=kernel_init,
                rngs=nnx_rngs,
            )
        )
        additive_layers.append(lambda x: jax.nn.relu(x))
        for _ in range(self.nlayers):
            additive_layers.append(
                nnx.Linear(
                    self.hidden_size,
                    self.hidden_size,
                    kernel_init=kernel_init,
                    rngs=nnx_rngs,
                )
            )
            additive_layers.append(lambda x: jax.nn.relu(x))
        additive_layers.append(
            nnx.Linear(
                self.hidden_size,
                self.net_in_size,
                kernel_init=kernel_init,
                rngs=nnx_rngs,
            )
        )
        self.additive_mlp = nnx.List(
            [
                nnx.Sequential(*additive_layers)
                for _ in range(self.num_coupling_layers)
            ]
        )

        scaling_layers = nnx.List()
        scaling_layers.append(
            nnx.Linear(
                self.net_in_size,
                self.hidden_size,
                kernel_init=kernel_init,
                rngs=nnx_rngs,
            )
        )
        scaling_layers.append(lambda x: jax.nn.relu(x))
        for _ in range(self.nlayers):
            scaling_layers.append(
                nnx.Linear(
                    self.hidden_size,
                    self.hidden_size,
                    kernel_init=kernel_init,
                    rngs=nnx_rngs,
                )
            )
            scaling_layers.append(lambda x: jax.nn.relu(x))
        scaling_layers.append(
            nnx.Linear(
                self.hidden_size,
                self.net_in_size,
                kernel_init=kernel_init,
                rngs=nnx_rngs,
            )
        )

        self.scaling_mlp = nnx.List(
            [
                nnx.Sequential(*scaling_layers)
                for _ in range(self.num_coupling_layers)
            ]
        )

        self.permutations = []
        for _ in range(self.num_coupling_layers):
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, in_size)
            self.permutations.append(perm)
        self.permutations = nnx.data(self.permutations)
        self.inverse_permutations = nnx.data(
            [jnp.argsort(p) for p in self.permutations]
        )

    def forward(
        self, x: jnp.ndarray, return_log_det: bool = False
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass of the RealNVP coupling layer.

        Args:
            x (jnp.ndarray): Input data.
            return_log_det (bool): Whether to return the log-determinant
                of the Jacobian.

        Returns:
            jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray] :
                Transformed data, and optionally the
                log-determinant of the Jacobian.
        """
        log_det = 0.0
        for i in range(self.num_coupling_layers):
            x = x[:, self.permutations[i]]
            xa, xb = x[:, : self.pass_size], x[:, self.pass_size :]
            m = self.additive_mlp[i](xb)
            s = self.scaling_mlp[i](xb)
            xa = xa * jnp.exp(s) + m
            x = jnp.concat([xa, xb], axis=1)  # Concatenate correctly
            log_det += jnp.sum(s, axis=1)  # Sum over the scaling parameters

        if return_log_det:
            return x, log_det
        else:
            return x

    def inverse(self, x: jnp.ndarray) -> jnp.ndarray:
        """Inverse pass of the RealNVP coupling layer.

        From the base distribution to the target distribution.

        Args:
            x (jnp.ndarray): Samples from the base distribution.

        Returns:
            jnp.ndarray: Transformed samples in the target distribution.
        """
        for i in reversed(
            range(self.num_coupling_layers)
        ):  # Reverse the coupling order
            xa, xb = x[:, : self.pass_size], x[:, self.pass_size :]
            m = self.additive_mlp[i](xb)
            s = self.scaling_mlp[i](xb)
            xa = (xa - m) / jnp.exp(s)
            x = jnp.concat([xa, xb], axis=1)  # Concatenate correctly
            x = x[:, self.inverse_permutations[i]]

        return x

    def train(
        self,
        key: jnp.ndarray,
        learning_rate: float = 1e-4,
        epochs: int = 1000,
        patience: int = 50,
    ) -> None:
        """Train the RealNVP model.

        Args:
            key: JAX random key for data splitting.
            learning_rate: Learning rate for the optimizer.
            epochs: Number of training epochs.
            patience: Patience for early stopping.
        """

        @jax.jit
        def loss_function(
            model: "RealNVP",
            targets: jnp.ndarray,
            weights: jnp.ndarray,
        ) -> jnp.ndarray:
            """Loss function for training RealNVP model."""
            return -jnp.mean(weights * model.log_prob_under_RealNVP(targets))

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
        """Sample from the trained RealNVP model.

        Args:
            key: JAX random key.
            num_samples: Number of samples to draw.

        Returns:
            Samples drawn from the RealNVP model.
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

    @jax.jit
    def log_prob_under_RealNVP(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the log probability under the RealNVP model.

        Args:
            x (jnp.ndarray): Input data.

        Returns:
            Log probabilities of the input data.
        """
        # calcualte the actual log prob under the RealNVP model
        # assuming a standard normal base distribution
        z, log_det_J = self.forward(x, return_log_det=True)
        log_pz = -0.5 * jnp.sum(z**2 + jnp.log(2 * jnp.pi), axis=1)
        log_prob_under_realnvp = log_pz + log_det_J
        return log_prob_under_realnvp

    @jax.jit
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

        return self.log_prob_under_RealNVP(transformed_x) + correction

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
        """Save the trained RealNVP model to a file.

        Args:
            filename: Path to the file where the model will be saved.
        """
        # Placeholder for save logic
        pass

    @classmethod
    def load(cls, filename: str) -> "RealNVP":
        """Load a trained RealNVP model from a file.

        Args:
            filename: Path to the file from which the model will be loaded.

        Returns:
            Loaded RealNVP model.
        """
        # Placeholder for load logic
        pass
