import numpy as np
from anesthetic.samples import NestedSamples
from margarine.maf import MAF
from margarine.marginal_stats import calculate
import pytest
from margarine.kde import KDE
import pytest
from numpy.testing import assert_equal

@pytest.mark.filterwarnings('ignore::RuntimeWarning')

def load_chains(root):
    """
    Function uses anesthetic to load in a set of chains and returns
    the pandas table of samples, a numpy
    array of the parameters in the uniform space and weights.
    """

    samples = NestedSamples(root=root)

    names = ['p' + str(i) for i in range(ndims)]
    theta = samples[names].values
    weights = samples.weights

    return samples, theta, weights, names

ndims=5

root = 'tests/test_samples/test'
samples, theta, weights, names = load_chains(root)

def test_maf():

    def check(i):
        if stats['Value'][i] < samples.D():
            assert(
                np.abs(stats['Value'][i]-samples.D())/
                (stats['Upper Bound'][i] - stats['Value'][i]) <=3)
        else:
            assert(
                np.abs(stats['Value'][i]-samples.D())/
                (stats['Value'][i] - stats['Lower Bound'][i]) <=3)

    bij = MAF(theta, weights)
    bij.train(250)

    x = bij.sample(5000)

    stats = calculate(bij).statistics()
    [check(i) for i in range(2)]

    prior = np.random.uniform(-4, 4, (len(theta), 5))
    stats = calculate(bij, prior_samples=prior,
        prior_weights=np.ones(len(prior))).statistics()
    [check(i) for i in range(2)]

    L = samples.logL.values
    estL = bij.log_like(theta, samples.ns_output()['logZ'].mean())
    for i in range(len(L)):
        if L[i] > -6:
            assert((L[i] - estL[i])/L[i]*100 <= 5)


def test_maf_kwargs():

    with pytest.raises(TypeError):
        bij = MAF(theta, weights)
        bij.sample(4.5)
    with pytest.raises(TypeError):
        MAF(theta, weights, number_networks=4.4)
    with pytest.raises(TypeError):
        MAF(theta, weights, learning_rate='foobar')
    with pytest.raises(TypeError):
        MAF(theta, weights, hidden_layers='foobar')
    with pytest.raises(TypeError):
        MAF(theta, weights, hidden_layers=[4.5, 50])
    with pytest.raises(TypeError):
        MAF(theta, weights)
        bij.train(epochs=4.5)
    with pytest.raises(TypeError):
        MAF(theta, weights)
        bij.train(epochs=100, early_stop='foo')

def test_maf_save_load():

    bij = MAF(theta, weights)
    bij.train(100)
    file = 'saved_maf.pkl'
    bij.save(file)
    loaded_bijector = MAF.load(file)
    for i in range(len(bij.mades)):
        assert_equal(bij.mades[i].get_weights(),
            loaded_bijector.mades[i].get_weights())

def test_kde():

    def check(i):
        if stats['Value'][i] < samples.D():
            assert(
                np.abs(stats['Value'][i]-samples.D())/
                (stats['Upper Bound'][i] - stats['Value'][i]) <=4)
        else:
            assert(
                np.abs(stats['Value'][i]-samples.D())/
                (stats['Value'][i] - stats['Lower Bound'][i]) <=4)

    kde = KDE(theta, weights)
    kde.generate_kde()
    x = kde.sample(5000)

    stats = calculate(kde).statistics()
    [check(i) for i in range(2)]

    prior = np.random.uniform(-4, 4, (len(theta), 5))
    stats = calculate(kde, prior_samples=prior,
        prior_weights=np.ones(len(prior))).statistics()
    [check(i) for i in range(2)]

    L = samples.logL.values
    estL = kde.log_like(theta, samples.ns_output()['logZ'].mean())
    for i in range(len(L)):
        if L[i] > -6:
            assert((L[i] - estL[i])/L[i]*100 <= 5)

def test_kde_save_load():

    kde = KDE(theta, weights)
    kde.generate_kde()
    file = 'saved_maf.pkl'
    kde.save(file)
    loaded_kde = KDE.load(file)
    assert_equal(kde.kde.covariance, loaded_kde.kde.covariance)
