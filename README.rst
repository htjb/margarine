================================================================
margarine: Posterior Sampling and Marginal Bayesian Statistics
================================================================

Introduction
------------

:margarine: Marginal Bayesian Statistics
:Authors: Harry T.J. Bevins
:Version: 0.2.0
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

Currently returned log-probabilities are gaussianised due to the
normalisation. There is a missing jacobian term to correct for this
that will be coded and added shortly.

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
