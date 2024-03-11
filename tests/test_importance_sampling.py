import numpy as np
from margarine.maf import MAF
from margarine.marginal_stats import calculate
import pytest
from scipy.stats import norm
from anesthetic import MCMCSamples

def D_KL(logL, weights):
    return -np.average(logL, weights=weights)

def d_g(logL, weights):
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
    @pytest.fixture
    def maf(self):
        return MAF(mcmc_samples, parameters=names)

    @pytest.fixture
    def calc(self, maf):
        return calculate(maf)
    
    def likelihood(self, parameters):
        y = np.array([0, 6])
        loglikelihood = (
            -0.5 * np.log(2 * np.pi * (1**2)) - 0.5 * ((y - parameters) / 1) ** 2
        ).sum(axis=-1)
        return loglikelihood


    def prior(self, parameters):
        y = np.array([6, 0])
        density = (
            -0.5 * np.log(2 * np.pi * (1**2)) - 0.5 * ((y - parameters) / 1) ** 2
        ).sum(axis=-1)
        return density

    # @pytest.mark.parametrize(("prior", [prior, maf.log_prob]))
    def test_integrate_prior(self, calc):
        calc.integrate(self.likelihood, self.prior, sample_size=1000)
    
    def test_integrate_maf(self, calc, maf):
        calc.integrate(self.likelihood, maf.log_prob, sample_size=1000)

    def test_integrate_unphysical(self, calc):
        with pytest.raises(ValueError):
            calc.integrate(self.likelihood, self.prior, sample_size=1000, logzero=10)

    @pytest.mark.parametrize("ss", [10, 100])
    @pytest.mark.parametrize("bs", [10, 100])
    def test_integrate_batching(self, calc, ss, bs):
        calc.integrate(self.likelihood, self.prior, sample_size=ss, batch_size=bs)
