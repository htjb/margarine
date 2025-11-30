"""Implementation of Masked Autoregressive Flow (MAF) estimator."""

from flax import nnx
from tensorflow_probability.substrates import jax as tfp

from margarine.base.baseflow import BaseDensityEstimator

tfd = tfp.distributions
tfb = tfp.bijectors


class MAF(BaseDensityEstimator, nnx.Module):
    """Masked Autoregressive Flow (MAF) density estimator."""

    def __init__(self) -> None:
        """Initialize MAF model."""
        super().__init__()
