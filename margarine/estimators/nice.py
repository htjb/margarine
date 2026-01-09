"""Implementation of the NICE estimator."""

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
    forward_transform,
    inverse_transform,
    train_test_split,
)

# surpress some orbax warnings
warnings.filterwarnings("ignore", category=UserWarning, module="orbax")
logging.getLogger("absl").setLevel(logging.ERROR)

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
        layers.append(lambda x: jax.nn.gelu(x))
        for _ in range(self.nlayers):
            layers.append(
                nnx.Linear(self.hidden_size, self.hidden_size, rngs=nnx_rngs)
            )
            layers.append(lambda x: jax.nn.gelu(x))

        layers.append(
            nnx.Linear(self.hidden_size, self.net_in_size, rngs=nnx_rngs)
        )
        self.mlp = nnx.List(
            [nnx.Sequential(*layers) for _ in range(self.num_coupling_layers)]
        )

        self.S = nnx.Param(jnp.zeros(in_size))

    @jax.jit
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """NICE forward pass.

        This is the forward pass of the NICE coupling layer from
        samples in the target space to the base distribution space.

        Args:
            x (jnp.ndarray): Input samples.
        """
        mlp_fns = tuple((lambda xb, mlp=mlp: mlp(xb)) for mlp in self.mlp)

        def body(i: int, x: jnp.ndarray) -> jnp.ndarray:
            """Body function for the coupling layers.

            Args:
                i: Forward counter.
                x: Input samples.

            Returns:
                Transformed samples.
            """
            x = jax.lax.cond(
                i % 2 == 0,
                lambda x: jnp.flip(x, axis=-1),
                lambda x: x,
                x,
            )
            xa, xb = x[:, : self.pass_size], x[:, self.pass_size :]
            m = jax.lax.switch(i, mlp_fns, xb)
            x = jnp.concatenate([xa + m, xb], axis=1)
            return x

        x = jax.lax.fori_loop(0, self.num_coupling_layers, body, x)

        x = jnp.exp(self.S) * x

        if self.num_coupling_layers % 2 == 1:
            x = jnp.flip(x, axis=-1)

        return x

    @jax.jit
    def inverse(self, x: jnp.ndarray) -> jnp.ndarray:
        """NICE inverse pass.

        Args:
            x: Input samples.
        """
        if self.num_coupling_layers % 2 == 1:
            x = jnp.flip(x, axis=-1)

        x = x / jnp.exp(self.S)  # Undo scaling first

        mlp_fns = tuple((lambda xb, mlp=mlp: mlp(xb)) for mlp in self.mlp)

        def body(k: int, x: jnp.ndarray) -> jnp.ndarray:
            """Body function for the inverse coupling layers.

            Args:
                k: Forward counter.
                x: Input samples.

            Returns:
                Inverted samples.
            """
            # convert upward counter to downward counter
            i = self.num_coupling_layers - 1 - k

            xa, xb = x[:, : self.pass_size], x[:, self.pass_size :]
            m = jax.lax.switch(i, mlp_fns, xb)
            x = jnp.concatenate([xa - m, xb], axis=1)
            x = jax.lax.cond(
                i % 2 == 0,
                lambda x: jnp.flip(x, axis=-1),
                lambda x: x,
                x,
            )
            return x

        return jax.lax.fori_loop(0, self.num_coupling_layers, body, x)

    def train(
        self,
        key: jnp.ndarray,
        learning_rate: float = 1e-4,
        epochs: int = 1000,
        patience: int = 50,
        batch_size: int = 256,
    ) -> None:
        """Train the NICE model.

        Args:
            key: JAX random key for data splitting.
            learning_rate: Learning rate for the optimizer.
            epochs: Number of training epochs.
            patience: Patience for early stopping.
            batch_size: Batch size for training.
        """

        @nnx.jit
        def loss_function(
            model: "NICE",
            targets: jnp.ndarray,
            weights: jnp.ndarray,
        ) -> jnp.ndarray:
            """Loss function for training NICE model.

            Args:
                model (NICE): NICE model.
                targets (jnp.ndarray): Samples from the target distribution
                    that have been passed through the forward_transform.
                weights (jnp.ndarray): Weights for the samples.

            Returns:
                jnp.ndarray: Computed loss.
            """
            return -jnp.mean(weights * model.log_prob_under_NICE(targets))

        @nnx.jit
        def train_step(
            model: NICE,
            optimizer: nnx.Optimizer,
            train_phi: jnp.ndarray,
            train_weights: jnp.ndarray,
        ) -> None:
            """Single training step for NICE model.

            Args:
                model (NICE): NICE model.
                optimizer (nnx.Optimizer): Optimizer for
                    updating model parameters.
                train_phi (jnp.ndarray): Training samples.
                train_weights (jnp.ndarray): Weights for the training samples.
            """
            grad = nnx.grad(loss_function)(model, train_phi, train_weights)
            optimizer.update(model, grad)

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
                self.test_phi, self.test_weights, subkey, test_size=0.5
            )
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
                batch_weights = self.train_weights[batch_indices]
                loss = loss_function(self, batch_phi, batch_weights)
                train_step(self, optimizer, batch_phi, batch_weights)
                accumulated_train_loss += loss * len(batch_phi)
            loss = accumulated_train_loss / len(self.train_phi)
            tl.append(loss)

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

        self.train_loss = tl
        self.val_loss = vl

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

    @jax.jit
    def log_prob_under_NICE(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the log probability under the NICE model.

        Args:
            x (jnp.ndarray): Input data.

        Returns:
            Log probabilities of the input data.
        """
        # calculate the actual log prob under the NICE model
        # assuming a standard normal base distribution
        z = self.forward(x)
        log_pz = -0.5 * jnp.sum(z**2 + jnp.log(2 * jnp.pi), axis=1)
        log_det_J = jnp.sum(self.S)
        log_prob_under_nice = log_pz + log_det_J
        return log_prob_under_nice

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
                tfb.Scale(1 / (self.theta_ranges[1] - self.theta_ranges[0])),
                tfb.Shift(-self.theta_ranges[0]),
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
        if isinstance(prior_density, BaseDensityEstimator):
            prior_density = prior_density.log_prob(x)

        return self.log_prob(x) + logevidence - prior_density

    def save(self, filename: str) -> None:
        """Save the trained NICE model to a file.

        Args:
            filename: Path to the file where the model will be saved.
        """
        path = Path(filename).resolve()
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
            "theta_ranges": self.theta_ranges,
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

        with open(f"{path}/metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        with ZipFile(filename + ".marg", "w") as z:
            for subpath in path.rglob("*"):
                if subpath.is_file():
                    z.write(subpath, arcname=subpath.relative_to(path))

        shutil.rmtree(path)

    @classmethod
    def load(cls, filename: str) -> BaseDensityEstimator | None:
        """Load a trained NICE model from a file.

        Args:
            filename (str): Path to the file from which the
                model will be loaded.

        Returns:
            Loaded NICE model.
        """
        zip_path = Path(f"{filename}.marg")
        path = Path(filename + ".tmp").resolve()
        with ZipFile(zip_path) as z:
            # Extract all files to a folder
            z.extractall(path)

        with open(f"{path}/metadata.yaml") as f:
            metadata = yaml.unsafe_load(f)

        version = metadata.get("margarine_version", None)
        if version is None:
            print(
                "Warning: The KDE was saved with a version of margarine ",
                " < 2.0.0. In order to load it you will need to downgrade ",
                "margarine to a version < 2.0.0. e.g. ",
                "pip install margarine<2.0.0",
            )
            return
        if version != _version.__version__:
            print(
                f"Warning: The KDE was saved with margarine version "
                f"{version}, but you are loading it with version "
                f"{_version.__version__}. This may lead to "
                f"incompatibilities."
            )

        with open(f"{path}/config.yaml") as f:
            config = yaml.unsafe_load(f)

        instance = cls(
            theta=jnp.zeros((1, 2)),
            in_size=config["in_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_coupling_layers=config["num_coupling_layers"],
            theta_ranges=config["theta_ranges"],
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

        shutil.rmtree(path)
        return instance
