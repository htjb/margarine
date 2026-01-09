"""Statistics functions for margarine."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import tqdm
from jax.scipy.special import logsumexp

from margarine.base.baseflow import BaseDensityEstimator


def kldivergence(
    density_estimator_p: BaseDensityEstimator,
    density_estimator_q: BaseDensityEstimator,
    samples_p: jnp.ndarray | None = None,
    weights: jnp.ndarray | None = None,
    key: jnp.ndarray = jax.random.PRNGKey(0),
) -> float:
    """Kullback-Leibler divergence between two density estimators.

    Args:
        density_estimator_p (BaseDensityEstimator): The first
        density estimator.
        density_estimator_q (BaseDensityEstimator): The second
        density estimator.
        samples_p (jnp.ndarray | None): Optional samples from the
        first density estimator. If None, samples will be drawn
        from density_estimator_p.
        weights (jnp.ndarray | None): Optional weights for the
        samples. If None, uniform weights will be used.
        key (jnp.ndarray): JAX random key for sampling.

    Returns:
        kld (float): The Kullback-Leibler divergence D_KL(P || Q).
    """
    if samples_p is None:
        samples_p = density_estimator_p.sample(key, num_samples=10000)
    log_p = density_estimator_p.log_prob(samples_p)
    log_q = density_estimator_q.log_prob(samples_p)
    kld = jnp.average(log_p - log_q, weights=weights)
    return kld


def model_dimensionality(
    density_estimator_p: BaseDensityEstimator,
    density_estimator_q: BaseDensityEstimator,
    samples_p: jnp.ndarray | None = None,
    weights: jnp.ndarray | None = None,
    key: jnp.ndarray = jax.random.PRNGKey(0),
) -> float:
    """Model dimensionality between two density estimators.

    Args:
        density_estimator_p (BaseDensityEstimator): The first
        density estimator.
        density_estimator_q (BaseDensityEstimator): The second
        density estimator.
        samples_p (jnp.ndarray | None): Optional samples from the
        first density estimator. If None, samples will be drawn
        from density_estimator_p.
        weights (jnp.ndarray | None): Optional weights for the
        samples. If None, uniform weights will be used.
        key (jnp.ndarray): JAX random key for sampling.

    Returns:
        dim (float): The model dimensionality.
    """
    if samples_p is None:
        samples_p = density_estimator_p.sample(key, num_samples=10000)
    log_p = density_estimator_p.log_prob(samples_p)
    log_q = density_estimator_q.log_prob(samples_p)
    delta_log = log_p - log_q
    dim = 2 * jnp.cov(delta_log, aweights=weights)
    return dim


def integrate(
    density_estimator: BaseDensityEstimator,
    likelihood: Callable,
    prior: BaseDensityEstimator,
    batch_size: int = 1000,
    sample_size: int = 10000,
    logzero: float = -1e30,
    key: jnp.ndarray = jax.random.PRNGKey(0),
) -> dict:
    """Importance sampling integration of a likelihood function.

    Args:
        density_estimator (BaseDensityEstimator): A density estimator
        for the likelihood function.
        likelihood (Callable): The likelihood function to integrate.
        prior (BaseDensityEstimator): A density estimator
        for the prior probability.
        batch_size (int): The number of samples to draw at each iteration.
        sample_size (int): The number of samples to draw in total.
        logzero (float): The definition of zero for the
            loglikelihood function.
        key (jnp.ndarray): JAX random key for sampling.

    Returns:
        stats (dict): Dictionary containing useful statistics
    """
    xs = jnp.empty((sample_size, density_estimator.theta.shape[-1]))
    fs = jnp.empty(sample_size)
    gs = jnp.empty(sample_size)
    pis = jnp.empty(sample_size)

    n_todo = sample_size
    trials = 0

    with tqdm.tqdm(total=sample_size) as pbar:
        while n_todo > 0:  # draw samples until we have enough accepted
            key, subkey = jax.random.split(key)
            x = density_estimator.sample(subkey, batch_size)
            f = jnp.array(list(map(likelihood, x)))
            g = density_estimator.log_prob(x)
            # determine which samples are accepted based on logzero
            in_bounds = jnp.logical_and(f >= logzero, g >= logzero)
            n_accept = x[in_bounds].shape[0]
            if n_accept <= n_todo:
                # accepted samples
                xs = xs.at[
                    sample_size - n_todo : sample_size - n_todo + n_accept
                ].set(x[in_bounds])
                # corresponding likelihoods
                fs = fs.at[
                    sample_size - n_todo : sample_size - n_todo + n_accept
                ].set(f[in_bounds])
                # corresponding flow log-probs
                gs = gs.at[
                    sample_size - n_todo : sample_size - n_todo + n_accept
                ].set(g[in_bounds])
                # corresponding prior log-probs
                pis = pis.at[
                    sample_size - n_todo : sample_size - n_todo + n_accept
                ].set(prior.log_prob(x[in_bounds]))
                trials += batch_size
            else:
                n_accept = n_todo
                xs = xs.at[sample_size - n_todo :].set(x[in_bounds][:n_accept])
                fs = fs.at[sample_size - n_todo :].set(f[in_bounds][:n_accept])
                gs = gs.at[sample_size - n_todo :].set(g[in_bounds][:n_accept])
                pis = pis.at[sample_size - n_todo :].set(
                    prior.log_prob(x[in_bounds])[:n_accept]
                )
                last_index = in_bounds[-1]
                trials += last_index + 1
            n_todo -= n_accept
            pbar.update(n_accept)
            # prevent excessive calls
            if trials > 10 * sample_size:
                raise ValueError(
                    "Too many unsuccessful trials, this"
                    + "typically indicates mismatch between"
                    + "flow and likelihood"
                )

        # calculate the importance weights = prior * likelihood / flow
        weights = jnp.exp(fs + pis - gs)
        # effective sample size
        eff = jnp.sum(weights) ** 2 / jnp.sum(weights**2) / sample_size
        # estimate of the integral and its standard error
        integral = sample_size / trials * weights.mean()
        log_integral = logsumexp(fs + pis - gs) - jnp.log(trials)

        stderr = jnp.sqrt(
            (jnp.sum(weights**2) / trials - integral**2) / (trials - 1)
        )
        log_stderr = jnp.log(stderr)

    stats = {
        "x": xs,
        "y": fs,
        "weights": weights,
        "efficiency": eff,
        "trials": trials,
        "log_integral": log_integral,
        "log_stderr": log_stderr,
    }
    return stats
