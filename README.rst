================================================================
margarine: Posterior Sampling and Marginal Bayesian Statistics
================================================================

Introduction
------------

:margarine: Marginal Bayesian Statistics
:Authors: Harry T.J. Bevins
:Version: 1.2.1
:Homepage:  https://github.com/htjb/margarine
:Documentation: https://margarine.readthedocs.io/

.. image:: https://readthedocs.org/projects/margarine/badge/?version=latest
  :target: https://margarine.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status
.. image:: https://mybinder.org/badge_logo.svg
  :target: https://mybinder.org/v2/gh/htjb/margarine/master?labpath=notebook%2FTutorial.ipynb
.. image:: http://img.shields.io/badge/astro.IM-arXiv%3A2205.12841-B31B1B.svg
  :target: https://arxiv.org/abs/2205.12841

Installation
------------

The software should be installed via the git repository using the following
commands in the terminal

.. code:: bash

  git clone https://github.com/htjb/margarine.git # or the equivalent using ssh keys
  cd margarine
  python setup.py install --user

or via a pip install with

.. code:: bash

  pip install margarine

Note that the pip install is not always the most up to date version of the code.

Details/Examples
----------------

`margarine` is designed to make the calculation of marginal bayesian statistics
feasible given a set of samples from an MCMC or nested sampling run.

An example of how to use the code can be found on the github in the
jupyter notebook `notebook/Tutorial.ipynb` or alternatively at
`here <https://mybinder.org/v2/gh/htjb/margarine/master?labpath=notebook%2FTutorial.ipynb>`__.

Documentation
-------------

The documentation is available at: https://margarine.readthedocs.io/

To compile it locally you can run

.. code:: bash

  cd docs
  sphinx-build source html-build

after cloning the repo and installing the relevant packages with

.. code:: bash

  pip install sphinx numpydoc sphinx_rtd_theme

Licence and Citation
--------------------

The software is available on the MIT licence.

If you use the code for academic purposes we request that you cite the following
`paper <https://ui.adsabs.harvard.edu/abs/2022arXiv220512841B/abstract>`__ and
the `MaxEnt22 proceedings <https://ui.adsabs.harvard.edu/search/p_=0&q=author%3A%22Bevins%2C%20H.%20T.%20J.%22&sort=date%20desc%2C%20bibcode%20desc>`__
If you use the clustering implementation please cite the following
`preprint <https://arxiv.org/abs/2305.02930>`__.
You can use the following bibtex

.. code:: bibtex

  @ARTICLE{2023MNRAS.526.4613B,
        author = {{Bevins}, Harry T.~J. and {Handley}, William J. and {Lemos}, Pablo and {Sims}, Peter H. and {de Lera Acedo}, Eloy and {Fialkov}, Anastasia and {Alsing}, Justin},
          title = "{Marginal post-processing of Bayesian inference products with normalizing flows and kernel density estimators}",
        journal = {\mnras},
      keywords = {methods: data analysis, methods: statistical, cosmic background radiation, dark ages, reionization, first stars, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Cosmology and Nongalactic Astrophysics, Computer Science - Machine Learning},
          year = 2023,
          month = dec,
        volume = {526},
        number = {3},
          pages = {4613-4626},
            doi = {10.1093/mnras/stad2997},
  archivePrefix = {arXiv},
        eprint = {2205.12841},
  primaryClass = {astro-ph.IM},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.4613B},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
  }

and

.. code:: bibtex

  @ARTICLE{2022arXiv220711457B,
       author = {{Bevins}, Harry and {Handley}, Will and {Lemos}, Pablo and {Sims}, Peter and {de Lera Acedo}, Eloy and {Fialkov}, Anastasia},
        title = "{Marginal Bayesian Statistics Using Masked Autoregressive Flows and Kernel Density Estimators with Examples in Cosmology}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2022,
        month = jul,
          eid = {arXiv:2207.11457},
        pages = {arXiv:2207.11457},
  archivePrefix = {arXiv},
       eprint = {2207.11457},
  primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv220711457B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
  }

and

.. code:: bibtex

  @ARTICLE{2023arXiv230502930B,
        author = {{Bevins}, Harry and {Handley}, Will},
          title = "{Piecewise Normalizing Flows}",
        journal = {arXiv e-prints},
      keywords = {Statistics - Machine Learning, Computer Science - Machine Learning},
          year = 2023,
          month = may,
            eid = {arXiv:2305.02930},
          pages = {arXiv:2305.02930},
            doi = {10.48550/arXiv.2305.02930},
  archivePrefix = {arXiv},
        eprint = {2305.02930},
  primaryClass = {stat.ML},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230502930B},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
  }


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
