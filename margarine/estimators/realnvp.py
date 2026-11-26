"""Implementation of the RealNVP estimator."""

import logging
import shutil
import warnings
from pathlib import Path
from zipfile import ZipFile

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as orbax
import tqdm
import yaml
from flax import nnx
from tensorflow_probability.substrates import jax as tfp

from margarine import _version
from margarine.base.baseflow import BaseDensityEstimator
from margarine.utils.utils import (
    approximate_bounds,
    build_mlp,
    check_conditional_input,
    conditional_forward_transform,
    coupling_permutations,
    forward_transform,
    inverse_transform,
    prepare_conditional_input,
    train_test_split,
)

# surpress some orbax warnings
warnings.filterwarnings("ignore", category=UserWarning, module="orbax")
logging.getLogger("absl").setLevel(logging.ERROR)


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
        permutations_key: jnp.ndarray = jax.random.PRNGKey(0),
        c: jnp.ndarray | None = None,
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
            permutations_key: JAX random key for permutations.
            c: Optional conditional parameters.
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

        if self.in_size != self.theta.shape[1]:
            raise ValueError(
                "Input size must match the number of parameters in theta."
            )
        if self.in_size < 2:
            raise ValueError("Input size must be at least 2 for RealNVP.")

        self.hidden_size = hidden_size
        self.nlayers = num_layers
        self.num_coupling_layers = num_coupling_layers

        self.pass_size = in_size // 2
        self.net_in_size = in_size - self.pass_size

        if c is not None:
            if c.ndim == 1:
                # one conditional parameter per sample
                c = c[:, None]
            if c.shape[0] != len(self.theta):
                raise ValueError(
                    "Conditional parameters must have the same number of "
                    "samples as theta."
                )
            self.c = nnx.data(c)
            self.conditional = True
            self.conditional_shape = self.c.shape[1]
            self.net_in_size += self.conditional_shape
        else:
            self.conditional = False

        # like glorot normal but with 0.1 stddev
        kernel_init = nnx.initializers.variance_scaling(
            0.1, "fan_avg", "truncated_normal"
        )

        self.additive_mlp = nnx.List(
            [
                build_mlp(
                    self.net_in_size,
                    self.hidden_size,
                    self.nlayers,
                    self.pass_size,
                    nnx_rngs,
                    kernel_init=kernel_init,
                )
                for _ in range(self.num_coupling_layers)
            ]
        )

        self.scaling_mlp = nnx.List(
            [
                build_mlp(
                    self.net_in_size,
                    self.hidden_size,
                    self.nlayers,
                    self.pass_size,
                    nnx_rngs,
                    kernel_init=kernel_init,
                )
                for _ in range(self.num_coupling_layers)
            ]
        )

        self.permutations_key = permutations_key
        self.permutations = nnx.data(
            coupling_permutations(
                self.permutations_key, in_size, self.num_coupling_layers
            )
        )
        self.inverse_permutations = nnx.data(
            [jnp.argsort(p) for p in self.permutations]
        )

    def forward(
        self,
        x: jnp.ndarray,
        c: jnp.ndarray | None = None,
        return_log_det: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass of the RealNVP coupling layer.

        Args:
            x (jnp.ndarray): Input data.
            return_log_det (bool): Whether to return the log-determinant
                of the Jacobian.
            c (jnp.ndarray | None): Conditional parameters for the samples.

        Returns:
            jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray] :
                Transformed data, and optionally the
                log-determinant of the Jacobian.
        """
        log_det = 0.0
        for i in range(self.num_coupling_layers):
            x = x[:, self.permutations[i]]
            xa, xb = x[:, : self.pass_size], x[:, self.pass_size :]
            m = self.additive_mlp[i](
                jnp.concatenate([xb, c], axis=1) if self.conditional else xb
            )
            s = self.scaling_mlp[i](
                jnp.concatenate([xb, c], axis=1) if self.conditional else xb
            )
            xa = xa * jnp.exp(s) + m
            x = jnp.concat([xa, xb], axis=1)  # Concatenate correctly
            log_det += jnp.sum(s, axis=1)  # Sum over the scaling parameters

        if return_log_det:
            return x, log_det
        else:
            return x

    def inverse(
        self, x: jnp.ndarray, c: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Inverse pass of the RealNVP coupling layer.

        From the base distribution to the target distribution.

        Args:
            x (jnp.ndarray): Samples from the base distribution.
            c (jnp.ndarray | None): Conditional parameters for the samples.

        Returns:
            jnp.ndarray: Transformed samples in the target distribution.
        """
        for i in reversed(
            range(self.num_coupling_layers)
        ):  # Reverse the coupling order
            xa, xb = x[:, : self.pass_size], x[:, self.pass_size :]
            m = self.additive_mlp[i](
                jnp.concatenate([xb, c], axis=1) if self.conditional else xb
            )
            s = self.scaling_mlp[i](
                jnp.concatenate([xb, c], axis=1) if self.conditional else xb
            )
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
        batch_size: int = 256,
    ) -> None:
        """Train the RealNVP model.

        Args:
            key: JAX random key for data splitting.
            learning_rate: Learning rate for the optimizer.
            epochs: Number of training epochs.
            patience: Patience for early stopping.
            batch_size: Size of training batches.
        """

        @jax.jit
        def loss_function(
            model: RealNVP,
            targets: jnp.ndarray,
            weights: jnp.ndarray,
            c: jnp.ndarray | None = None,
        ) -> jnp.ndarray:
            """Loss function for training RealNVP model.

            Args:
                model (RealNVP): The RealNVP model.
                targets (jnp.ndarray): Target samples.
                weights (jnp.ndarray): Weights for the samples.
                c (jnp.ndarray | None): Conditional parameters for the samples.

            Returns:
                jnp.ndarray: Computed loss.
            """
            return -jnp.mean(
                weights * model.log_prob_under_RealNVP(targets, c)
            )

        # @nnx.jit
        def train_step(
            model: RealNVP,
            optimizer: nnx.Optimizer,
            train_phi: jnp.ndarray,
            train_weights: jnp.ndarray,
            train_c: jnp.ndarray | None,
        ) -> None:
            """Single training step for NICE model.

            Args:
                model (NICE): The NICE model to be trained.
                optimizer (nnx.Optimizer): The optimizer to
                    update model parameters.
                train_phi (jnp.ndarray): Training samples.
                train_weights (jnp.ndarray): Weights for the training samples.
                train_c (jnp.ndarray | None): Conditional
                    parameters for the training samples.
            """
            grad = nnx.grad(loss_function)(
                model, train_phi, train_weights, train_c
            )
            optimizer.update(model, grad)

        phi = forward_transform(
            self.theta, self.theta_ranges[0], self.theta_ranges[1]
        )
        weights = self.weights / jnp.sum(self.weights)

        if self.conditional:
            normalized_c, mean, std = conditional_forward_transform(self.c)
            self.normalized_c = nnx.data(normalized_c)
            self.conditional_mean = nnx.data(mean)
            self.conditional_std = nnx.data(std)

        key, subkey = jax.random.split(key)
        # need to split the data into training and validation sets
        (
            self.train_phi,
            self.test_phi,
            self.train_weights,
            self.test_weights,
            self.train_c,
            self.test_c,
        ) = train_test_split(
            phi,
            weights,
            self.normalized_c if self.conditional else None,
            subkey,
            test_size=0.2,
        )
        key, subkey = jax.random.split(key)
        (
            self.test_phi,
            self.val_phi,
            self.test_weights,
            self.val_weights,
            self.test_c,
            self.val_c,
        ) = train_test_split(
            self.test_phi,
            self.test_weights,
            self.test_c,
            subkey,
            test_size=0.5,
        )

        tx = optax.adam(learning_rate)
        optimizer = nnx.Optimizer(self, tx, wrt=nnx.Param)

        pbar = tqdm.tqdm(range(epochs), desc="Training")

        best_loss = jnp.inf
        best_model = nnx.state(self, nnx.Param)
        c = 0

        data_size = len(self.train_phi)

        tl, vl = [], []
        for _ in pbar:
            data_permutations = jax.random.permutation(
                subkey, jnp.arange(data_size)
            )
            accumulated_train_loss = 0.0
            for i in range(0, len(self.train_phi), batch_size):
                batch_indices = data_permutations[i : i + batch_size]
                batch_phi = self.train_phi[batch_indices]
                if self.conditional:
                    batch_c = self.train_c[batch_indices]
                batch_weights = self.train_weights[batch_indices]
                loss = loss_function(
                    self,
                    batch_phi,
                    batch_weights,
                    batch_c if self.conditional else None,
                )
                train_step(
                    self,
                    optimizer,
                    batch_phi,
                    batch_weights,
                    batch_c if self.conditional else None,
                )
                accumulated_train_loss += loss * len(batch_phi)
            loss = accumulated_train_loss / len(self.train_phi)
            tl.append(loss)

            val_loss = loss_function(
                self,
                self.val_phi,
                self.val_weights,
                self.val_c if self.conditional else None,
            )
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

            key, subkey = jax.random.split(subkey)

        self.train_loss = jnp.array(tl)
        self.val_loss = jnp.array(vl)

        if best_model is not None:
            nnx.update(self, best_model)

    def sample(
        self,
        key: jnp.ndarray,
        num_samples: int,
        c: jnp.ndarray | float | None = None,
    ) -> jnp.ndarray:
        """Sample from the trained RealNVP model.

        Args:
            key (jnp.ndarray): JAX random key.
            num_samples (int): Number of samples to draw.
            c (jnp.ndarray | float | None): Conditional
                parameters for the samples.

        Returns:
            Samples drawn from the RealNVP model.
        """
        u = jax.random.uniform(key, shape=(num_samples, self.theta.shape[1]))
        samples = self(u, c)
        return samples

    def __call__(
        self, u: jnp.ndarray, c: jnp.ndarray | float | None = None
    ) -> jnp.ndarray:
        """Transform samples from the unit hypercube.

        Args:
            u (jnp.ndarray): Samples from the unit hypercube.
            c (jnp.ndarray | float | None): Conditional
            parameters for the samples.
                Can be a single sample or a batch of samples.

        Returns:
            Transformed samples.
        """
        # from unit hypercube to standard normal space
        x = forward_transform(u, 0, 1)

        c = prepare_conditional_input(self, c, x.shape[0])

        # from standard normal to target space
        x = self.inverse(x, c)

        # from gaussianized target space to original target space
        x = inverse_transform(x, self.theta_ranges[0], self.theta_ranges[1])
        return x

    @jax.jit
    def log_prob_under_RealNVP(
        self, x: jnp.ndarray, c: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Compute the log probability under the RealNVP model.

        Args:
            x (jnp.ndarray): Input data.
            c (jnp.ndarray | None): Conditional data.
                Can be a single sample or a batch of samples.

        Returns:
            Log probabilities of the input data.
        """
        if self.conditional:
            c = check_conditional_input(c, self.conditional_shape, x.shape[0])
        # calcualte the actual log prob under the RealNVP model
        # assuming a standard normal base distribution
        z, log_det_J = self.forward(
            x, return_log_det=True, c=c if self.conditional else None
        )
        log_pz = -0.5 * jnp.sum(z**2 + jnp.log(2 * jnp.pi), axis=1)
        log_prob_under_realnvp = log_pz + log_det_J
        return log_prob_under_realnvp

    @jax.jit
    def log_prob(
        self, x: jnp.ndarray, c: jnp.ndarray | float | None = None
    ) -> jnp.ndarray:
        """Compute the log probability of the input data.

        Args:
            x (jnp.ndarray): Input data.
            c (jnp.ndarray | None): Conditional data.
                Can be a single sample or a batch of samples.

        Returns:
            Log probabilities of the input data.
        """
        transformed_x = forward_transform(
            x, self.theta_ranges[0], self.theta_ranges[1]
        )
        transform_chain = tfb.Chain(
            [
                tfb.Invert(tfb.NormalCDF()),
                tfb.Scale(1 / (self.theta_ranges[1] - self.theta_ranges[0])),
                tfb.Shift(-self.theta_ranges[0]),
            ]
        )

        def norm_jac(y: jnp.ndarray) -> jnp.ndarray:
            """Calculate the normalising jacobian for the transformation."""
            return transform_chain.inverse_log_det_jacobian(y, event_ndims=0)

        correction = -norm_jac(transformed_x).sum(axis=1)

        c = prepare_conditional_input(self, c, x.shape[0])

        return self.log_prob_under_RealNVP(transformed_x, c) + correction

    def log_like(
        self,
        x: jnp.ndarray,
        logevidence: float,
        prior_density: jnp.ndarray | BaseDensityEstimator,
        c: jnp.ndarray | float | None = None,
    ) -> jnp.ndarray:
        """Compute the marginal log-likelihood of given samples.

        Args:
            x (jnp.ndarray): Samples for which to compute the log-likelihood.
            logevidence (float): Log-evidence term. This is really Z(c) if the
                flow is conditional and Z if not.
            prior_density (jnp.ndarray | BaseDensityEstimator):
                Prior density or density estimator.
            c (jnp.ndarray | float | None): Conditional data.

        Returns:
            Log likelihoods of the input data.
        """
        if isinstance(prior_density, BaseDensityEstimator):
            prior_density = prior_density.log_prob(x)

        if self.conditional:
            # check if c is a single sample or single unique value
            # if no and logevidence is not a single value, raise an error
            c = check_conditional_input(c, self.conditional_shape, x.shape[0])
            if jnp.unique(c, axis=0).shape[0] > 1 and jnp.isscalar(
                logevidence
            ):
                raise ValueError(
                    "If multiple conditional parameters are passed, "
                    "then there should be "
                    "one logevidence per conditional parameter."
                )

        return self.log_prob(x, c) + logevidence - prior_density

    def save(self, filename: str) -> None:
        """Save the trained RealNVP model to a file.

        Args:
            filename: Path to the file where the model will be saved.
        """
        path = Path(filename + ".tmp").resolve()
        if path.exists():
            shutil.rmtree(path)

        state = nnx.state(self)

        checkpointer = orbax.PyTreeCheckpointer()
        checkpointer.save(f"{path}/state", state)

        config = {
            "in_size": self.in_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.nlayers,
            "num_coupling_layers": self.num_coupling_layers,
            "key": self.permutations_key.tolist(),
            "theta_ranges": self.theta_ranges,
            "conditional": self.conditional,
            "conditional_shape": self.conditional_shape
            if self.conditional
            else None,
        }
        with open(f"{path}/config.yaml", "w") as f:
            yaml.dump(config, f)

        metadata = {
            "theta": self.theta,
            "weights": self.weights,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_phi": self.train_phi,
            "test_phi": self.test_phi,
            "train_weights": self.train_weights,
            "test_weights": self.test_weights,
            "val_phi": self.val_phi,
            "val_weights": self.val_weights,
            "margarine_version": _version.__version__,
        }

        if self.conditional:
            metadata.update(
                {
                    "train_c": self.train_c,
                    "test_c": self.test_c,
                    "val_c": self.val_c,
                    "conditional_mean": self.conditional_mean,
                    "conditional_std": self.conditional_std,
                }
            )

        with open(f"{path}/metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        with ZipFile(filename + ".marg", "w") as z:
            for subpath in path.rglob("*"):
                if subpath.is_file():
                    z.write(subpath, arcname=subpath.relative_to(path))

        shutil.rmtree(path)

    @classmethod
    def load(cls, filename: str) -> BaseDensityEstimator | None:
        """Load a trained RealNVP model from a file.

        Args:
            filename: Path to the file from which the model will be loaded.

        Returns:
            Loaded RealNVP model.
        """
        zip_path = Path(f"{filename}.marg")
        path = Path(filename + ".tmp").resolve()
        with ZipFile(zip_path) as z:
            # Extract all files to a folder
            z.extractall(path)

        with open(f"{path}/metadata.yaml") as f:
            metadata = yaml.unsafe_load(f)

        version = metadata.get("margarine_version", None)
        if version < _version.__version__:
            raise ValueError(
                f"The RealNVP model was saved with margarine version "
                f"{version}, but you are loading it with version "
                f"{_version.__version__}. "
                f"We suggest you downgrade or "
                f"retrain the model."
            )
        if version > _version.__version__:
            print(
                f"Warning:The RealNVP model was saved with margarine version "
                f"{version}, but you are loading it with version "
                f"{_version.__version__}. "
                f"This might lead to inconsistencies. "
                "We suggest you upgrade or "
                "retrain the model."
            )

        with open(f"{path}/config.yaml") as f:
            config = yaml.unsafe_load(f)

        conditional = config.get("conditional", False)
        if conditional:
            c = jnp.zeros((1, config["conditional_shape"]))

        instance = cls(
            theta=jnp.zeros((1, config["in_size"])),
            in_size=config["in_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_coupling_layers=config["num_coupling_layers"],
            permutations_key=jnp.array(config["key"], dtype=jnp.uint32),
            theta_ranges=config["theta_ranges"],
            c=c if conditional else None,
        )

        abstract_state = nnx.state(instance)
        checkpointer = orbax.PyTreeCheckpointer()
        state = checkpointer.restore(
            f"{path}/state", item=abstract_state, partial_restore=True
        )
        nnx.update(instance, state)

        instance.theta = metadata["theta"]
        instance.weights = metadata["weights"]
        instance.train_loss = metadata["train_loss"]
        instance.val_loss = metadata["val_loss"]
        instance.train_phi = metadata["train_phi"]
        instance.test_phi = metadata["test_phi"]
        instance.train_weights = metadata["train_weights"]
        instance.test_weights = metadata["test_weights"]
        instance.val_phi = metadata["val_phi"]
        instance.val_weights = metadata["val_weights"]
        if instance.conditional:
            instance.train_c = metadata["train_c"]
            instance.test_c = metadata["test_c"]
            instance.val_c = metadata["val_c"]
            instance.conditional_mean = metadata["conditional_mean"]
            instance.conditional_std = metadata["conditional_std"]
        shutil.rmtree(path)
        return instance
