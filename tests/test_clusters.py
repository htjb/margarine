import numpy as np
from anesthetic import MCMCSamples
from margarine.clustered import clusterMAF
from margarine.marginal_stats import calculate
import pytest
from numpy.testing import assert_equal, assert_allclose
from scipy.stats import ks_2samp
from clustered_distributions import TwoMoons

def D_KL(logL, weights):
    return -np.average(logL, weights=weights)

def d_g(logL, weights):
    return 2*np.cov(logL, aweights=weights)

nsamples = 2500

tm = TwoMoons()

samples = tm.sample(nsamples)
theta = samples.numpy()
weights = np.ones(len(theta))

logL = tm.log_prob(samples).numpy()

samples = MCMCSamples(data=theta)
samples_kl = D_KL(logL, weights)
samples_d = d_g(logL, weights)
names = [i for i in range(theta.shape[-1])]

def test_maf_clustering():

    bij = clusterMAF(theta, weights=weights)
    bij.train(10000, early_stop=True)
    file = 'saved_maf_cluster.pkl'
    bij.save(file)

    loaded_bijector = clusterMAF.load('saved_maf_cluster.pkl')
    for f in range(len(bij.flow)):
        for i in range(len(bij.flow[f].mades)):
            assert_equal(bij.flow[f].mades[i].get_weights(),
                loaded_bijector.flow[f].mades[i].get_weights())

    def check_stats(label):
        if label == "KL Divergence":
            value = samples_kl
            assert_allclose(stats[label], value, rtol=1, atol=1)
        else:
            value = samples_d
            assert_allclose(stats[label], value, rtol=1, atol=1)

    stats_label = ["KL Divergence", "BMD"]

    stats = calculate(bij).statistics()
    [check_stats(l) for l in stats_label]

    equal_weight_theta = samples.compress(100)[names].values
    x = bij.sample(len(equal_weight_theta))

    res = [ks_2samp(x[:, i], equal_weight_theta[:, i]) 
           for i in range(equal_weight_theta.shape[-1])]
    p_values = [res[i].pvalue for i in range(len(res))]
    for i in range(len(p_values)):
        assert(p_values[i] > 0.05)

def test_maf_cluster_kwargs():

    with pytest.raises(ValueError):
        bij = clusterMAF(theta, weights=weights, cluster_number=3)
    with pytest.raises(ValueError):
        bij = clusterMAF(theta, weightes=weights, 
                         cluster_labels=np.ones(len(theta)))
    
    labels = np.ones(len(theta))
    labels[:len(theta)//2] = 0

def test_cluster_size():
    # testing the while loop for cluster size
    def draw(ndims):
        return np.random.multivariate_normal(
                np.zeros(ndims), np.eye(ndims), ndims*10)

    samples = draw(3)
    flow = clusterMAF(samples)