================================================================
margarine: Posterior Sampling and Marginal Bayesian Statistics
================================================================

Introduction
------------

:margarine: Marginal Bayesian Statistics
:Authors: Harry T.J. Bevins
:Version: 0.1.0
:Homepage:  https://github.com/htjb/margarine
:Documentation: https://margarine.readthedocs.io/

.. image:: https://readthedocs.org/projects/margarine/badge/?version=latest
  :target: https://margarine.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status
.. image:: https://mybinder.org/badge_logo.svg
  :target: https://mybinder.org/v2/gh/htjb/margarine/HEAD?labpath=notebook%2FTutorial.ipynb

Installation
------------

The software should be installed via the git repository using the following
commands in the terminal

.. code:: bash

  git clone https://github.com/htjb/margarine.git # or the equivalent using ssh keys
  cd margarine
  python setup.py install --user

A pip install is coming soon.

Details/Examples
----------------

`margarine` is designed to make the calculation of marginal bayesian statistics
feasible given a set of samples from an MCMC or nested sampling run.

An example of how to use the code can be found on the github in the
jupyter notebook `notebook/Tutorial.ipynb`, alternatively
in the compiled documentation or at
`here <https://mybinder.org/v2/gh/htjb/margarine/7f55f9a9d3f3adb2356cb94b32c599caac8ea1ef?urlpath=lab%2Ftree%2Fnotebook%2FTutorial.ipynb>`_.

Health Warning
--------------

**The code is still in development.**

**This branch contains code to calculate proper log probabilities using the
MAFs and KDEs. It still needs thorough testing.**

Currently the code assumes, when calculating marginal statistics, that the
priors are uniformly distributed. If the priors are more complex then one of
two approaches can currently be taken in order to use `margarine` to calculate
the KL divergence and Bayesian Dimensionality:

a. Before training a density estimator on the samples you can transform them
into the uniform parameter space. This could be as simple as inputting the
logarithm of a log-uniformly distributed parameter or may require more
complex transformations. Since the KL divergence and dimensionality are
invariant under these types of transformations the subsequent values
calculated with `margarine` will be equivalent to the values in the original
parameter space.

b. In some instances there may not be an obvious transformation from the
non-uniform parameter space defined by a complex prior to the uniform
parameter space. In these cases a more manual approach can be taken to
calculate the marginal statistics. In order to calculate the marginal statistics
we need to be able to assess the log probability of both the prior and the
posterior and when the prior is complex then we can use `margarine` to train a
density estimator on both it and the posterior. This gives us access to the
log pdfs and an example of this is shown below

.. code:: python3

  from margarine.kde import KDE
  from margarine.processing import _forward_transform

  # theta is the posterior samples loaded in previously with weights=weights
  # prior is the prior samples with weights=prior_weights

  posterior_kde = KDE(theta, weights)
  posterior_kde.generate_kde()
  posterior_kde_samples = posterior_kde.sample(5000)

  prior_kde = KDE(prior, prior_weights)
  prior_kde.generate_kde()
  prior_kde_samples = prior_kde.sample(5000)

  posterior_logprob = posterior_kde.kde.logpdf(
          _forward_transform(
              posterior_kde_samples,
              posterior_kde.theta_min, posterior_kde.theta_max).T)

  prior_logprob = prior_kde.kde.logpdf(
          _forward_transform(
              prior_kde_samples,
              prior_kde.theta_min, prior_kde.theta_max).T)

  # prior_logprob and posterior_logprob may need masking where non finite.

  logL = weights*posterior_logprob - prior_weights*prior_logprob
  import tensorflow as tf
  KL = tf.reduce_mean(logL)
  dimensionality = 2*(tf.reduce_mean(logL**2) - tf.reduce_mean(logL)**2)

Documentation
-------------

The documentation is available at: https://margarine.readthedocs.io/

To compile it locally you can run

.. code:: bash

  cd docs
  sphinx-build source html-build

after cloning the repo and installing the relevant packages.

Licence and Citation
--------------------

The software is available on the MIT licence.

If you use the code for academic purposes we request that you cite the paper
currently in preparation as Bevins et al. in prep..

Requirements
------------

The code requires the following packages to run:

- `numpy <https://pypi.org/project/numpy/>`__
- `tensorflow <https://pypi.org/project/tensorflow/>`__
- `scipy <https://pypi.org/project/scipy/>`__

To compile the documentation locally you will need:

- `sphinx <https://pypi.org/project/Sphinx/>`__
- `numpydoc <https://pypi.org/project/numpydoc/>`__

To run the test suit you will need:

- `pytest <https://docs.pytest.org/en/stable/>`__

Contributing
------------

Contributions and suggestions for areas of development are welcome and can
be made by opening a issue to report a bug or propose a new feature for discussion.
