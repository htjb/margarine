"""Custom layers for neural networks."""

import jax.numpy as jnp
from flax import nnx


class MaskedLinear(nnx.Module):
    """Masked Linear layer for MADE networks."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: jnp.ndarray,
        kernel_init: nnx.initializers.Initializer | None = None,
        rngs: dict | None = None,
    ) -> None:
        """Initialize MaskedLinear layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            mask: Mask to apply to the weights.
            kernel_init: Initializer for the weights.
            rngs: Random number generators for initialization.
        """
        if kernel_init is None:
            # like glorot normal but with 0.1 stddev
            kernel_init = nnx.initializers.variance_scaling(
                0.1, "fan_avg", "truncated_normal"
            )

        self.linear = nnx.Linear(
            in_features,
            out_features,
            rngs=rngs,
            kernel_init=kernel_init,
            bias_init=nnx.initializers.zeros,
        )
        self.mask = mask

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the masked linear transformation.

        Args:
            x: Input data.

        Returns:
            jnp.ndarray: Transformed data.
        """
        masked_weight = self.linear.kernel * self.mask.T
        return x @ masked_weight + self.linear.bias
