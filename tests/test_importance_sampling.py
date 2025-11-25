"""Tests for importance sampling functionality."""

import numpy as np
import pytest
import tensorflow as tf
from anesthetic import MCMCSamples
from scipy.stats import norm

from margarine.maf import MAF
from margarine.marginal_stats import calculate


def D_KL(
    logL: np.ndarray | tf.Tensor, weights: np.ndarray | tf.Tensor
) -> float:
    """Calculate the Kullback-Leibler divergence."""
    return -np.average(logL, weights=weights)


def d_g(
    logL: np.ndarray | tf.Tensor, weights: np.ndarray | tf.Tensor
) -> float:
    """Calculate the BMD statistic."""
    return 2 * np.cov(logL, aweights=weights)


norm = norm(loc=4.2, scale=0.3)
theta = norm.rvs(size=(1000, 2))
logL = norm.logpdf(theta).sum(axis=1)
weights = np.ones(len(theta))

mcmc_samples = MCMCSamples(data=theta, logL=logL)
samples_kl = D_KL(logL, weights)
samples_d = d_g(logL, weights)
names = [i for i in range(theta.shape[-1])]


class TestImportance:
    """Class to test importance sampling functionality."""

    @pytest.fixture
    def maf(self) -> MAF:
        """Fixture to create MAF instance."""
        return MAF(mcmc_samples, parameters=names)

    @pytest.fixture
    def calc(self, maf: MAF) -> calculate:
        """Fixture to create a calculate instance with the trained MAF."""
        return calculate(maf)

    def likelihood(self, parameters: np.ndarray) -> np.ndarray:
        """Define a simple Gaussian likelihood function."""
        y = np.array([0, 6])
        loglikelihood = (
            -0.5 * np.log(2 * np.pi * (1**2))
            - 0.5 * ((y - parameters) / 1) ** 2
        ).sum(axis=-1)
        return loglikelihood

    def prior(self, parameters: np.ndarray) -> np.ndarray:
        """Define a simple Gaussian prior function."""
        y = np.array([6, 0])
        density = (
            -0.5 * np.log(2 * np.pi * (1**2))
            - 0.5 * ((y - parameters) / 1) ** 2
        ).sum(axis=-1)
        return density

    # @pytest.mark.parametrize(("prior", [prior, maf.log_prob]))
    def test_integrate_prior(self, calc: calculate) -> None:
        """Test integration using the prior distribution."""
        calc.integrate(self.likelihood, self.prior, sample_size=1000)

    def test_integrate_maf(self, calc: calculate, maf: MAF) -> None:
        """Test integration using the MAF distribution."""
        calc.integrate(self.likelihood, maf.log_prob, sample_size=1000)

    def test_integrate_unphysical(self, calc: calculate) -> None:
        """Test integration with unphysical logzero value."""
        with pytest.raises(ValueError):
            calc.integrate(
                self.likelihood, self.prior, sample_size=1000, logzero=10
            )

    @pytest.mark.parametrize("ss", [10, 100])
    @pytest.mark.parametrize("bs", [10, 100])
    def test_integrate_batching(
        self, calc: calculate, ss: int, bs: int
    ) -> None:
        """Test integration with different sample and batch sizes."""
        calc.integrate(
            self.likelihood, self.prior, sample_size=ss, batch_size=bs
        )
