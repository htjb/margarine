import numpy as np
from anesthetic import MCMCSamples
from margarine.maf import MAF
from margarine.marginal_stats import calculate
import pytest
from margarine.kde import KDE
from margarine.clustered import clusterMAF
from numpy.testing import assert_equal, assert_allclose
from scipy.stats import ks_2samp, norm



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

    estL = bij.log_prob(theta.astype(np.float32))

    res = ks_2samp(logL, estL)

    p_values = res.pvalue
    print(p_values)
    assert round(p_values, 2) > 0.05

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

    estL = kde.log_prob(theta.astype(np.float32))
    res = ks_2samp(logL, estL)

    p_values = res.pvalue
    print(p_values)
    assert round(p_values, 2) > 0.05

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

test_maf()
test_kde()