import numpy as np
from anesthetic import MCMCSamples
from margarine.maf import MAF
from margarine.marginal_stats import calculate
import pytest
from margarine.kde import KDE
from margarine.clustered import clusterMAF
from numpy.testing import assert_equal, assert_allclose
from scipy.stats import ks_2samp


def likelihood(parameters):
    y = np.array([0, 6])
    loglikelihood = (
        -0.5 * np.log(2 * np.pi * (1**2)) - 0.5 * ((y - parameters) / 1) ** 2
    ).sum(axis=-1)
    return loglikelihood


def prior(parameters):
    y = np.array([6, 0])
    density = (
        -0.5 * np.log(2 * np.pi * (1**2)) - 0.5 * ((y - parameters) / 1) ** 2
    ).sum(axis=-1)
    return density


def D_KL(logL, weights):
    return -np.average(logL, weights=weights)


def d_g(logL, weights):
    return 2 * np.cov(logL, aweights=weights)


ndims = 2
nsamples = 2500
x = np.random.normal(0, 1, nsamples)
y = np.random.normal(6, 1, nsamples)

theta = np.vstack([x, y]).T
weights = np.ones(len(theta))

logL = likelihood(theta)

mcmc_samples = MCMCSamples(data=theta, logL=logL)
samples_kl = D_KL(logL, weights)
samples_d = d_g(logL, weights)
names = [i for i in range(theta.shape[-1])]


def test_maf():
    def check_stats(label):
        if label == "KL Divergence":
            value = samples_kl
            assert_allclose(stats[label], value, rtol=1, atol=1)
        else:
            value = samples_d
            assert_allclose(stats[label], value, rtol=1, atol=1)

    bij = MAF(theta, weights=weights)
    bij.train(10000, early_stop=True)

    stats_label = ["KL Divergence", "BMD"]

    stats = calculate(bij).statistics()
    [check_stats(l) for l in stats_label]

    equal_weight_theta = mcmc_samples.compress(50)[names].values
    x = bij.sample(len(equal_weight_theta))

    res = [
        ks_2samp(equal_weight_theta[:, i], x[:, i])
        for i in range(equal_weight_theta.shape[-1])
    ]
    p_values = [res[i].pvalue for i in range(len(res))]
    for i in range(len(p_values)):
        assert round(p_values[i], 2) > 0.05

    estL = bij.log_like(equal_weight_theta.astype(np.float32), 0.0)
    check_like = 0
    for i in range(len(logL)):
        if np.isclose(np.exp(logL[i]), np.exp(estL[i]), rtol=1, atol=1):
            check_like += 1
    assert (len(logL) - check_like) / len(logL) <= 0.1


def test_maf_kwargs():

    with pytest.raises(TypeError):
        bij = MAF(theta, weights=weights)
        bij.sample(4.5)
    with pytest.raises(TypeError):
        MAF(theta, weights=weights, number_networks=4.4)
    with pytest.raises(TypeError):
        MAF(theta, weights=weights, learning_rate='foobar')
    with pytest.raises(TypeError):
        MAF(theta, weights=weights, hidden_layers='foobar')
    with pytest.raises(TypeError):
        MAF(theta, weights=weights, hidden_layers=[4.5, 50])
    with pytest.raises(TypeError):
        MAF(theta, weights=weights)
        bij.train(epochs=4.5)
    with pytest.raises(TypeError):
        MAF(theta, weights=weights)
        bij.train(epochs=100, early_stop='foo')
    with pytest.raises(TypeError):
        MAF(theta, weights=weights)
        bij.train(epochs=100, clustering=5)
    with pytest.raises(TypeError):
        MAF(theta, weights=weights)
        bij.train(epochs=100, cluster_numeber='foo')
    with pytest.raises(TypeError):
        MAF(theta, weights=weights)
        bij.train(epochs=100, cluster_labels=5)

def test_maf_save_load():

    bij = MAF(theta, weights=weights)
    bij.train(100)
    file = 'saved_maf.pkl'
    bij.save(file)
    loaded_bijector = MAF.load(file)
    for i in range(len(bij.mades)):
        assert_equal(bij.mades[i].get_weights(),
            loaded_bijector.mades[i].get_weights())

def test_kde():

    def check_stats(label):
        if label == "KL Divergence":
            value = samples_kl
            assert_allclose(stats[label], value, rtol=1, atol=1)
        else:
            value = samples_d
            assert_allclose(stats[label], value, rtol=1, atol=1)

    kde = KDE(theta, weights=weights)
    kde.generate_kde()

    stats_label = ["KL Divergence", "BMD"]

    stats = calculate(kde).statistics()
    [check_stats(l) for l in stats_label]

    equal_weight_theta = mcmc_samples.compress(50)[names].values
    x = kde.sample(len(equal_weight_theta))

    res = [ks_2samp(equal_weight_theta[:, i], x[:, i])
           for i in range(equal_weight_theta.shape[-1])]
    p_values = [res[i].pvalue for i in range(len(res))]
    for i in range(len(p_values)):
        assert(round(p_values[i], 2) >0.05)

    estL = kde.log_like(equal_weight_theta, 0.0)
    check_like = 0
    for i in range(len(logL)):
        if np.isclose(np.exp(logL[i]), np.exp(estL[i]), rtol=1, atol=1):
            check_like += 1
    assert((len(logL)-check_like)/len(logL) <= 0.1)

def test_kde_save_load():

    kde = KDE(theta, weights=weights)
    kde.generate_kde()
    file = 'saved_maf.pkl'
    kde.save(file)
    loaded_kde = KDE.load(file)
    assert_equal(kde.kde.covariance, loaded_kde.kde.covariance)

def test_anesthetic():

    kde = KDE(mcmc_samples, parameters=names)
    maf = MAF(mcmc_samples, parameters=names)
    cmaf = clusterMAF(mcmc_samples)

    assert_equal(kde.parameters, names)
    assert_equal(maf.parameters, names)

    # not providing parametes here but deriving them from the
    # anesthetic object columns
    assert(np.all(cmaf.parameters == np.array(names)))


class TestImportance:
    @pytest.fixture
    def maf(self):
        return MAF(mcmc_samples, parameters=names)

    @pytest.fixture
    def calc(self, maf):
        return calculate(maf)

    # @pytest.mark.parametrize(("prior", [prior, maf.log_prob]))
    def test_integrate_prior(self, calc):
        calc.integrate(likelihood, prior, sample_size=1000)
    
    def test_integrate_maf(self, calc, maf):
        calc.integrate(likelihood, maf.log_prob, sample_size=1000)

    def test_integrate_unphysical(self, calc):
        with pytest.raises(ValueError):
            calc.integrate(likelihood, prior, sample_size=1000, logzero=10)

    @pytest.mark.parametrize("ss", [10, 100])
    @pytest.mark.parametrize("bs", [10, 100])
    def test_integrate_batching(self, calc, ss, bs):
        calc.integrate(likelihood, prior, sample_size=ss, batch_size=bs)
