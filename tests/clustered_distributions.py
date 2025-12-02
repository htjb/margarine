"""Generate clustered distributions for testing.

Code modified from
https://github.com/VincentStimper/normalizing-flows repository.
"""

import jax
import jax.numpy as jnp


class Target:
    """Sample target distributions to test models."""

    def __init__(
        self,
        prop_scale: int | float = 6.0,
        prop_shift: int | float = -3.0,
    ) -> None:
        """Constructor.

        Args:
          prop_scale (jnp.ndarray): Scale for the uniform proposal
          prop_shift (jnp.ndarray): Shift for the uniform proposal
        """
        super().__init__()
        self.prop_scale = jnp.array(prop_scale)
        self.prop_shift = jnp.array(prop_shift)
        self.n_dims = len(self.prop_scale)

    def log_prob(self, z: jnp.ndarray) -> jnp.ndarray:
        """Base log probability function.

        Args:
          z (jnp.ndarray): value or batch of latent variable

        Returns:
          jnp.ndarray: log probability of the distribution for z
        """
        raise NotImplementedError(
            "The log probability is not implemented yet."
        )

    def rejection_sampling(
        self, key: jnp.ndarray, num_steps: int = 1
    ) -> jnp.ndarray:
        """Perform rejection sampling on distribution.

        Args:
          key (jnp.ndarray): JAX random key
          num_steps (int): Number of rejection sampling steps to perform

        Returns:
          jnp.ndarray: Accepted samples
        """
        key, subkey1, subkey2 = jax.random.split(key, 3)
        eps = jax.random.normal(key=subkey1, shape=(num_steps, self.n_dims))
        z_ = self.prop_scale * eps + self.prop_shift
        prob = jax.random.normal(key=subkey2, shape=(num_steps,))
        prob_ = jnp.exp(self.log_prob(z_) - self.max_log_prob)
        accept = prob_ > prob
        z = z_[accept, :]
        return z

    def sample(self, key: jnp.ndarray, num_samples: int = 1) -> jnp.ndarray:
        """Sample from image distribution through rejection sampling.

        Args:
          key: JAX random key
          num_samples: Number of samples to draw

        Returns:
          Samples
        """
        z = jnp.zeros((0, self.n_dims))
        while len(z) < num_samples:
            key, subkey = jax.random.split(key)
            z_ = self.rejection_sampling(subkey, num_samples)
            ind = jnp.min(jnp.array([len(z_), num_samples - len(z)]))
            z = jnp.concat([z, z_[:ind, :]], axis=0)
        return z


class TwoMoons(Target):
    """Bimodal two-dimensional distribution."""

    def __init__(self) -> None:
        """Constructor."""
        super().__init__()
        self.n_dims = 2
        self.max_log_prob = 0.0

    def log_prob(self, z: jnp.ndarray) -> jnp.ndarray:
        """Log probability function for the two moons distribution.

        ```
        log(p) = - 1/2 * ((norm(z) - 2) / 0.2) ** 2
                 + log(  exp(-1/2 * ((z[0] - 2) / 0.3) ** 2)
                       + exp(-1/2 * ((z[0] + 2) / 0.3) ** 2))
        ```

        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        a = jnp.abs(z[:, 0])
        """log_prob = (
            -0.5 * ((torch.norm(z, dim=1) - 2) / 0.2) ** 2
            - 0.5 * ((a - 2) / 0.3) ** 2
            + torch.log(1 + torch.exp(-4 * a / 0.09))
        )"""
        log_prob = (
            -0.5 * ((jnp.linalg.norm(z, axis=1) - 2) / 0.2) ** 2
            - 0.5 * ((a - 2) / 0.3) ** 2
            + jnp.log(1 + jnp.exp(-4 * a / 0.09))
        )
        return log_prob
