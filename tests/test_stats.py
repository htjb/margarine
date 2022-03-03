import numpy as np
import matplotlib.pyplot as plt
from anesthetic.samples import NestedSamples
from bayesstats.maf import MAF
from bayesstats.marginal_stats import maf_calculations, kde_calculations
import pytest
from bayesstats.kde import KDE
import pytest

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

    bij.train(100, early_stop=True)

    x = bij.sample(5000)

    stats = maf_calculations(bij, x)
    klerr = samples.ns_output()['D'].std()
    bderr = samples.ns_output()['d'].std()
    assert((stats.klDiv()-samples.D())/klerr <= 3)
    assert((stats.bayesian_dimensionality()-samples.d())/bderr <=3)

def test_kde():

    kde = KDE(theta, weights)
    kde.generate_kde()
    x = kde.sample(5000)

    stats = kde_calculations(kde, x)
    klerr = samples.ns_output()['D'].std()
    bderr = samples.ns_output()['d'].std()
    assert((stats.klDiv()-samples.D())/klerr <= 3)
    assert((stats.bayesian_dimensionality()-samples.d())/bderr <=3)

    #y = kde(np.random.uniform(0, 1, size=(100, theta.shape[-1])))
    #stats = kde_calculations(kde, y)
    #assert((stats.klDiv()-samples.D())/klerr <= 5)
    #assert((stats.bayesian_dimensionality()-samples.d())/bderr <=5)
