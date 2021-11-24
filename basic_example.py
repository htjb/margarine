from bayesstats.bijector import Bijector
from bayesstats.marginal_stats import bijector_calculations
from anesthetic.samples import NestedSamples
import matplotlib.pyplot as plt
from hist_plotting import build
import numpy as np

def load_chains(root, logs=[], ndims=5):
    """
    Function uses anesthetic to load in a set of chains and returns
    the pandas table of samples, a numpy
    array of the parameters in the uniform space and weights.
    """

    samples = NestedSamples(root=root)

    try:
        names = ['p' + str(i) for i in range(ndims)]
        theta = samples[names].values
    except:
        names = [i for i in range(ndims)]
        theta = samples[names].values

    for i in range(theta.shape[1]):
        if i in logs:
            theta[:, i] = np.log10(theta[:, i])
    weights = samples.weights

    return samples, theta, weights

# loading the example chains generated with polychord using a gaussian
# log-likelihood with the first three parameters on a log-uniform prior
# between 1e-3 and 1 and the last two parameters on uniform priors
# between -3 and 3.
root = 'log_prior_gaussian_basic/test'
samples, theta, weights = load_chains(root, logs=[0, 1, 2])

# initiating the bijector class. For the bijector and KDE to work the array
# of samples have to be in the uniform parameter space.
bij = Bijector(theta, weights)
# training the bijector. These are pretty fast and I would set epochs to
# ~500 if you want a good replication (depends a bit on the complexity of
# the posterior and needs more investigation).
bij.train(epochs=100)

# transform len(theta) samples from the hypercube using the bijector.
# The samples have equal weights.
x = bij(np.random.uniform(0, 1, size=(len(theta), theta.shape[-1])))

# example saving and loading the bijector (completely unecessary in this
# code but just as an example)
file = 'example_bijector.pkl'
#bij.save(file)
bij = Bijector.load(file)

# can also access the call function like so
x = bij.sample(5000)

# initiating the stats calculator with the bijector and samples from the
# bijector
stats_calculator = bijector_calculations(bij, x)
# performing the calculations and comparing to the results given by
# polychord and anesthetic. Results will be better if the bijector is trained
# for longer.
print('Bijector D={:.2f}'.format(stats_calculator.klDiv()),
    ' Polychord/Anesthetic D={:.2f}'.format(samples.D()))
print('Bijector d={:.2f}'.format(stats_calculator.bayesian_dimensionality()),
    ' Polychord/Anesthetic d={:.2f}'.format(samples.d()))
print('Bijector logZ={:.2f}'.format(stats_calculator.evidence()),
    ' Polychord/Anesthetic logZ={:.2f}'.format(samples.ns_output()['logZ'].mean()))

# KDE example is completely analogous to the bijectors example.
from bayesstats.kde import KDE
from bayesstats.marginal_stats import kde_calculations

# initiating the KDE class with the parameter samples in uniform space and
# the corresponding weights.
kde = KDE(theta, weights)
# generating (analogous to training) the KDE.
kde.generate_kde()

# generating len(theta) samples from the KDE. This is a proper transformation
# of the hypercube onto the KDE distribution but it is very slow. It's useful
# I think for the 'any prior you like' application of this work.
x = kde(np.random.uniform(0, 1, size=(10, theta.shape[-1])))

# again just demonstraiting saving and loading the kde class.
file = 'example_kde.pkl'
kde.save(file)
kde.load(file)

# a much quicker way to generate samples from the kde that can then be
# used to calculate marginal statistics.
x = kde.sample(5000)

# initiating the kde stats caclulator
stats_calculator = kde_calculations(kde, x)
# performing the calculations and comparing with polychord/anesthetic results.
print('KDE D={:.2f}'.format(stats_calculator.klDiv()),
    ' Polychord/Anesthetic D={:.2f}'.format(samples.D()))
print('KDE d={:.2f}'.format(stats_calculator.bayesian_dimensionality()),
    ' Polychord/Anesthetic d={:.2f}'.format(samples.d()))
print('KDE logZ={:.2f}'.format(stats_calculator.evidence()),
    ' Polychord/Anesthetic logZ={:.2f}'.format(samples.ns_output()['logZ'].mean()))
