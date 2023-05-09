import numpy as np
from anesthetic import read_chains
from margarine.maf import MAF
from margarine.marginal_stats import calculate
import pytest
from numpy.testing import assert_equal, assert_allclose
from scipy.stats import ks_2samp

@pytest.mark.filterwarnings('ignore::RuntimeWarning')

def load_chains(root):
    """
    Function uses anesthetic to load in a set of chains and returns
    the pandas table of samples, a numpy
    array of the parameters in the uniform space and weights.
    """

    samples = read_chains(root=root)

    names = [i for i in range(ndims)]
    theta = samples[names].values
    weights = samples.get_weights()

    return samples, theta, weights, names

ndims=3

root = 'tests/test_samples/test'
samples, theta, weights, names = load_chains(root)

def test_maf_clustering():

    def likelihood(parameters):
        y = np.array([0.8, 5, 4])
        loglikelihood = (-0.5*np.log(2*np.pi*(0.1**2))-0.5 \
            *((y - [parameters[0], parameters[1], parameters[2]]) / 0.1)**2).sum()
        return loglikelihood

    def check_stats(i):
        if i ==0:
            anesthetic_value = np.mean(samples.D_KL(2000).values)
        else:
            anesthetic_value = np.mean(samples.d_G(2000).values)
        assert_allclose(stats['Value'][i], anesthetic_value, rtol=1, atol=1)

    bij = MAF(theta, weights, clustering=True)
    bij.train(1000, early_stop=True)

    stats = calculate(bij).statistics()
    [check_stats(i) for i in range(2)]

    prior = np.random.uniform(-10, 10, (len(theta), ndims))
    stats = calculate(bij, prior_samples=prior,
        prior_weights=np.ones(len(prior))).statistics()
    [check_stats(i) for i in range(2)]

    equal_weight_theta = samples.compress(100)[names].values
    x = bij.sample(len(equal_weight_theta))

    res = [ks_2samp(equal_weight_theta[:, i], x[:, i]) 
           for i in range(equal_weight_theta.shape[-1])]
    p_values = [res[i].pvalue for i in range(len(res))]
    for i in range(len(p_values)):
        assert(p_values[i] > 0.05)

    L = np.array([likelihood(equal_weight_theta[i]) for i in range(len(equal_weight_theta))])
    estL = bij.log_like(equal_weight_theta, samples.logZ().mean())
    check_like = 0
    for i in range(len(L)):
        if np.isclose(np.exp(L[i]), np.exp(estL[i]), rtol=1, atol=1):
            check_like += 1
    assert((len(L)-check_like)/len(L) <= 0.1)

def test_maf_save_load_clsuter():

    bij = MAF(theta, weights, clustering=True)
    bij.train(100)
    file = 'saved_maf_cluster.pkl'
    bij.save(file)
    loaded_bijector = MAF.load(file)
    for i in range(len(bij.mades)):
        assert_equal(bij.mades[i].get_weights(),
            loaded_bijector.mades[i].get_weights())

def test_maf_cluster_kwargs():

    with pytest.raises(ValueError):
        bij = MAF(theta, weights, cluster_number=3)
    with pytest.raises(ValueError):
        bij = MAF(theta, weights, cluster_labels=np.ones(len(theta)))
    
    bij = MAF(theta, weights, cluster_number=2, 
              cluster_labels=np.ones(len(theta)))
    assert_equal(bij.clustering, True)