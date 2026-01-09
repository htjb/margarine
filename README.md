# margarine: density estimation made easy

**Authors:** Harry T.J. Bevins  
**Version:** 2.0.1  
**Homepage:** https://github.com/htjb/margarine  
**Documentation:** https://margarine.readthedocs.io/

[![Documentation Status](https://readthedocs.org/projects/margarine/badge/?version=latest)](https://margarine.readthedocs.io/en/latest/?badge=latest) [![arXiv:2205.12841](http://img.shields.io/badge/astro.IM-arXiv%3A2205.12841-DCFF87.svg)](https://arxiv.org/abs/2205.12841) 
[![arXiv:2305.02930](http://img.shields.io/badge/astro.IM-arXiv%3A2305.02930-DCFF87.svg)](https://arxiv.org/abs/2305.02930)
[![arXiv:2207.11457](http://img.shields.io/badge/astro.IM-arXiv%3A2207.11457-DCFF87.svg)](https://arxiv.org/abs/2207.11457)
[![PyPI version](https://badge.fury.io/py/margarine.svg)](https://badge.fury.io/py/margarine)
[![Licence: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


`margarine` provides a suite of density estimation tools including KDEs, normalizing flows like NICE and RealNVP as well as a novel method for improved performance on multimodal distributions. 

The code can be used to:

- Emulate posterior distributions from weightened samples (e.g. MCMC, nested sampling)
- Build non-trivial priors from samples
- Perform density estimation tasks in general machine learning applications
- Emulate correctly normalised marginal likelihoods
- Calculate statistics like the KL divergence between different density estimators and marginal model dimensionalities.

---

## Installation

From version 2.0.0 margarine moved to JAX for improved performance. Older versions (1.x.x) using TensorFlow are still available via pip with the last release being 1.4.2.

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

---

## Getting Started

All of the density estimators in `margarine` have a common interface and set of methods including `train()`, `sample()`, `log_prob()`, `log_like()`, `save()` and `load()`. The below example shows how to train a RealNVP and generate samples.

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from margarine.estimators.realnvp import RealNVP

nsamples = 5000
key = jax.random.PRNGKey(0)

original_samples = jax.random.multivariate_normal(
    key,
    mean=jnp.array([0.0, 0.0]),
    cov=jnp.array([[1.0, 0.8], [0.8, 1.0]]),
    shape=(nsamples,),
)

weights = jnp.ones(len(original_samples))

realnvp_estimator = RealNVP(
        original_samples,
        weights=weights,
        in_size=2,
        hidden_size=50,
        num_layers=6,
        num_coupling_layers=6,
    )

key, subkey = jax.random.split(key)

realnvp_estimator.train(
            subkey,
            learning_rate=1e-3,
            epochs=2000,
            patience=50,
            batch_size=1000,
        )

generated_samples = realnvp_estimator.sample(key, num_samples=nsamples)

plt.scatter(
    original_samples[:, 0], original_samples[:, 1], alpha=0.5, label="Original Samples"
)
plt.scatter(
    generated_samples[:, 0], generated_samples[:, 1], alpha=0.5, label="Generated Samples"
)
plt.legend()
plt.title("RealNVP: Original vs Generated Samples")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
```

for more details see the documentation.

---

## Documentation

Available at: https://margarine.readthedocs.io/. To build locally:

```bash
pip install ".[docs]"
mkdocs serve
```


---

## Licence & Citation

Licensed under MIT.

If used for academic work, please cite:

* Main paper: https://arxiv.org/abs/2205.12841
* MaxEnt22 proceedings: https://arxiv.org/abs/2207.11457
* Piecewise Normalising Flows Paper: https://arxiv.org/abs/2305.02930

---

## Contributing

Contributions and feature suggestions welcome. Open an issue to report bugs or discuss ideas. See `CONTRIBUTING.md` for details.

The future goals of the project are outlined in `ROADMAP.md`.


