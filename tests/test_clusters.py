import numpy as np
from anesthetic import MCMCSamples
from margarine.maf import MAF
from margarine.marginal_stats import calculate
import pytest
from numpy.testing import assert_equal, assert_allclose
from scipy.stats import ks_2samp


nsamples = 5000
x = np.hstack([np.random.normal(0, 1, int(3*nsamples/4)), np.random.normal(8, 0.5, int(1*nsamples/4))])
y = np.hstack([np.random.normal(6, 1, int(2*nsamples/4)), np.random.normal(-5, 0.5, int(2*nsamples/4))])

theta = np.vstack([x, y]).T
weights = np.ones(len(theta))

samples = MCMCSamples(data=theta)
names = [i for i in range(theta.shape[-1])]

"""def test_maf_clustering():

    bij = MAF(theta, weights, clustering=True)
    bij.train(1000, early_stop=True)
    file = 'saved_maf_cluster.pkl'
    bij.save(file)

    loaded_bijector = MAF.load('saved_maf_cluster.pkl')
    for i in range(len(bij.mades)):
        for j in range(len(bij.mades[i])):
            assert_equal(bij.mades[i][j].get_weights(),
                loaded_bijector.mades[i][j].get_weights())

def test_statistics():

    def check_stats(i):
        if i ==0:
            anesthetic_value = np.mean(samples.D_KL(2000).values)
        else:
            anesthetic_value = np.mean(samples.d_G(2000).values)
        assert_allclose(stats['Value'][i], anesthetic_value, rtol=1, atol=1)

    bij = MAF.load('saved_maf_cluster.pkl')
    stats = calculate(bij).statistics()
    [check_stats(i) for i in range(2)]

    prior = np.random.uniform(-10, 10, (len(theta), theta.shape[-1]))
    stats = calculate(bij, prior_samples=prior,
        prior_weights=np.ones(len(prior))).statistics()
    [check_stats(i) for i in range(2)]

def test_sampling():

    bij = MAF.load('saved_maf_cluster.pkl')
    equal_weight_theta = samples.compress(100)[names].values
    x = bij.sample(len(equal_weight_theta))

    res = [ks_2samp(equal_weight_theta[:, i], x[:, i]) 
           for i in range(equal_weight_theta.shape[-1])]
    p_values = [res[i].pvalue for i in range(len(res))]
    for i in range(len(p_values)):
        assert(p_values[i] > 0.05)"""

def test_maf_cluster_kwargs():

    with pytest.raises(ValueError):
        bij = MAF(theta, weights, cluster_number=3)
    with pytest.raises(ValueError):
        bij = MAF(theta, weights, cluster_labels=np.ones(len(theta)))
    
    labels = np.ones(len(theta))
    labels[:len(theta)//2] = 0
    bij = MAF(theta, weights, cluster_number=2, 
              cluster_labels=labels)
    assert_equal(bij.clustering, True)
