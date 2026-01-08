# margarine: you won't believe it's not your posterior samples!

### Marginal Bayesian Statistics

**Authors:** Harry T.J. Bevins  
**Version:** 2.0.0  
**Homepage:** https://github.com/htjb/margarine  
**Documentation:** https://margarine.readthedocs.io/

[![Documentation Status](https://readthedocs.org/projects/margarine/badge/?version=latest)](https://margarine.readthedocs.io/en/latest/?badge=latest)  
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/htjb/margarine/master?labpath=notebook%2FTutorial.ipynb)  
[![arXiv:2205.12841](http://img.shields.io/badge/astro.IM-arXiv%3A2205.12841-B31B1B.svg)](https://arxiv.org/abs/2205.12841)

---

## Installation

Install from Git:

```bash
git clone https://github.com/htjb/margarine.git  # or use SSH
cd margarine
pip install .
```

Or via pip:

```bash
pip install margarine
```

Note: pip may not always give the latest version.

`margarine` versions ≥1.3.0 work with modern TensorFlow.

---

## Details / Examples

`margarine` helps compute marginal Bayesian statistics from MCMC or nested sampling output.

A usage example is available in the GitHub repo via:

* `notebook/Tutorial.ipynb`  
* Interactive version: https://mybinder.org/v2/gh/htjb/margarine/master?labpath=notebook%2FTutorial.ipynb

---

## Documentation

Available at: https://margarine.readthedocs.io/

To build locally:

```bash
cd docs
sphinx-build source html-build
```

Install docs dependencies:

```bash
pip install sphinx numpydoc sphinx_rtd_theme
```

---

## Licence & Citation

Licensed under MIT.

If used for academic work, please cite:

* Main paper: https://ui.adsabs.harvard.edu/abs/2022arXiv220512841B/abstract  
* MaxEnt22 proceedings: https://ui.adsabs.harvard.edu/search/p_=0&q=author%3A%22Bevins%2C%20H.%20T.%20J.%22&sort=date%20desc%2C%20bibcode%20desc  

If using the clustering implementation, cite: https://arxiv.org/abs/2305.02930

### BibTeX

```bibtex
@ARTICLE{2023MNRAS.526.4613B,
  author = {{Bevins}, Harry T.~J. and {Handley}, William J. and {Lemos}, Pablo and {Sims}, Peter H. and {de Lera Acedo}, Eloy and {Fialkov}, Anastasia and {Alsing}, Justin},
  title = "{Marginal post-processing of Bayesian inference products with normalizing flows and kernel density estimators}",
  journal = {\mnras},
  year = 2023,
  month = dec,
  volume = {526},
  number = {3},
  pages = {4613-4626},
  doi = {10.1093/mnras/stad2997},
  eprint = {2205.12841},
  primaryClass = {astro-ph.IM}
}
```

```bibtex
@ARTICLE{2022arXiv220711457B,
  author = {{Bevins}, Harry and {Handley}, Will and {Lemos}, Pablo and {Sims}, Peter and {de Lera Acedo}, Eloy and {Fialkov}, Anastasia},
  title = "{Marginal Bayesian Statistics Using Masked Autoregressive Flows and Kernel Density Estimators with Examples in Cosmology}",
  journal = {arXiv e-prints},
  year = 2022,
  month = jul,
  eprint = {2207.11457},
  primaryClass = {astro-ph.CO}
}
```

```bibtex
@ARTICLE{2023arXiv230502930B,
  author = {{Bevins}, Harry and {Handley}, Will},
  title = "{Piecewise Normalizing Flows}",
  journal = {arXiv e-prints},
  year = 2023,
  month = may,
  doi = {10.48550/arXiv.2305.02930},
  eprint = {2305.02930},
  primaryClass = {stat.ML}
}
```

---

## Contributing

Contributions and feature suggestions welcome.  
Open an issue to report bugs or discuss ideas.

See `CONTRIBUTING.md` for details.

---

## Roadmap to 2.0.0

Planned additions:

- JAX support for KDE and MAF with GPU acceleration  
- Base class density estimator to simplify extensions  
- Neural Spline Flow density estimator

