================================================================
bayesstats: Posterior Sampling and Marginal Bayesian Statistics
================================================================

Introduction
------------

:bayesstats: Marginal Bayesian Statistics
:Authors: Harry T.J. Bevins
:Version: 0.1.0
:Homepage:  https://github.com/htjb/bayesstats
:Documentation: https://bayesstats.readthedocs.io/

Installation
------------

The software should be installed via the git repository using the following
commands in the terminal

.. code:: bash

  git clone https://github.com/htjb/bayesstats.git # or the equivalent using ssh keys
  cd bayesstats
  python setup.py install --user

A pip install is comming soon.

Details/Examples
----------------

`bayesstats` is designed to make the calculation of marginal bayesian statistics
feasible given a set of samples from an MCMC or nested sampling run.

The current example can be run by cloning the git repo and following the
instructions below

.. code:: bash

  git clone git@github.com:htjb/bayesstats.git
  cd bayesstats
  python3 basic_example.py

The code should be well documented and is currently only intended as an
example of functionality/application for collaborators. The code is still
being edited and the example will be kept as up to date as possible.

Documentation
-------------

The documentation is available at: https://bayesstats.readthedocs.io/

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
