"""Statistics functions for margarine."""

from collections.abc import Callable

import jax.numpy as jnp
import tqdm
from jax.scipy.special import logsumexp

from margarine.density.base import BaseDensityEstimator


def kldivergence(
    density_estimator_p: BaseDensityEstimator,
    density_estimator_q: BaseDensityEstimator,
    samples_p: jnp.ndarray | None = None,
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

    Returns:
        kld (float): The Kullback-Leibler divergence D_KL(P || Q).
    """
    if samples_p is None:
        samples_p = density_estimator_p.sample(10000)
    log_p = density_estimator_p.log_prob(samples_p)
    log_q = density_estimator_q.log_prob(samples_p)
    kld = jnp.mean(log_p - log_q)
    return kld


def model_dimensionality(
    density_estimator_p: BaseDensityEstimator,
    density_estimator_q: BaseDensityEstimator,
    samples_p: jnp.ndarray | None = None,
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

    Returns:
        dim (float): The model dimensionality.
    """
    if samples_p is None:
        samples_p = density_estimator_p.sample(10000)
    log_p = density_estimator_p.log_prob(samples_p)
    log_q = density_estimator_q.log_prob(samples_p)
    delta_log = log_p - log_q
    dim = jnp.var(delta_log)
    return dim


def integrate(
    density_estimator: BaseDensityEstimator,
    likelihood: Callable,
    prior: BaseDensityEstimator,
    batch_size: int = 1000,
    sample_size: int = 10000,
    logzero: float = -1e30,
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
        while n_todo > 0:
            x = density_estimator.sample(batch_size)
            f = jnp.array(list(map(likelihood, x)))
            g = density_estimator.log_prob(x).numpy()
            in_bounds = jnp.logical_and(f >= logzero, g >= logzero)
            n_accept = x[in_bounds].shape[0]
            if n_accept <= n_todo:
                xs[sample_size - n_todo : sample_size - n_todo + n_accept] = x[
                    in_bounds
                ]
                fs[sample_size - n_todo : sample_size - n_todo + n_accept] = f[
                    in_bounds
                ]
                gs[sample_size - n_todo : sample_size - n_todo + n_accept] = g[
                    in_bounds
                ]
                pis[sample_size - n_todo : sample_size - n_todo + n_accept] = (
                    prior.log_prob(x[in_bounds])
                )
                trials += batch_size
            else:
                n_accept = n_todo
                xs[sample_size - n_todo :] = x[in_bounds][:n_accept]
                fs[sample_size - n_todo :] = f[in_bounds][:n_accept]
                gs[sample_size - n_todo :] = g[in_bounds][:n_accept]
                pis[sample_size - n_todo :] = prior.log_prob(x[in_bounds])[
                    :n_accept
                ]
                last_index = in_bounds[-1]
                trials += last_index + 1
            n_todo -= n_accept
            pbar.update(n_accept)
            if trials > 10 * sample_size:
                raise ValueError(
                    "Too many unsuccessful trials, this"
                    + "typically indicates mismatch between"
                    + "flow and likelihood"
                )

        weights = jnp.exp(fs + pis - gs)

        eff = jnp.sum(weights) ** 2 / jnp.sum(weights**2) / sample_size
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
