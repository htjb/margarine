import numpy as np
from anesthetic.samples import NestedSamples
from margarine.maf import MAF
from margarine.marginal_stats import maf_calculations, kde_calculations
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

    try:
        weights = samples.weights
    except:
        weights = samples.weight

    return samples, theta, weights

ndims=5

root = 'tests/test_samples/test'
samples, theta, weights = load_chains(root)

def test_maf():

    bij = MAF(theta, weights)
    bij.train(100)

    x = bij.sample(5000)

    stats = maf_calculations(bij, x)
    klerr = samples.ns_output()['D'].std()
    bderr = samples.ns_output()['d'].std()
    assert((stats.klDiv()-samples.D())/klerr <= 3)
    assert((stats.bayesian_dimensionality()-samples.d())/bderr <=3)

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

"""bij.train(100, early_stop=True)

x = bij.sample(5000)

stats = maf_calculations(bij, x)
klerr = samples.ns_output()['D'].std()
bderr = samples.ns_output()['d'].std()
assert((stats.klDiv()-samples.D())/klerr <= 3)
assert((stats.bayesian_dimensionality()-samples.d())/bderr <=3)"""

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

    kde = KDE(theta, weights)
    kde.generate_kde()
    x = kde.sample(5000)

    stats = kde_calculations(kde, x)
    klerr = samples.ns_output()['D'].std()
    bderr = samples.ns_output()['d'].std()
    assert((stats.klDiv()-samples.D())/klerr <= 3)
    assert((stats.bayesian_dimensionality()-samples.d())/bderr <=3)

def test_kde_save_load():

    kde = KDE(theta, weights)
    kde.generate_kde()
    file = 'saved_maf.pkl'
    kde.save(file)
    loaded_kde = KDE.load(file)
    assert_equal(kde.kde.covariance, loaded_kde.kde.covariance)
