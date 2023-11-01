import numpy as np

nDims = 5
nDerived = 0
sigma = 0.1


def likelihood(theta):
    """Simple Gaussian Likelihood"""

    nDims = len(theta)
    r2 = sum(theta**2)
    logL = -np.log(2 * np.pi * sigma * sigma) * nDims / 2.0
    logL += -r2 / 2 / sigma / sigma

    return logL


import anesthetic as ns
from pypolychord.output import PolyChordOutput

from scipy.stats import multivariate_normal

prior = multivariate_normal(np.ones(nDims), np.eye(nDims))
pc_out = PolyChordOutput("chains", "gaussian")
chains = ns.read_chains("chains/gaussian", columns=np.arange(nDims))
Zs = chains.logZ(100)
from margarine.maf import MAF
from margarine.marginal_stats import calculate

flow = MAF.load("maf.pkl")

calc = calculate(flow, prior_de=prior)

stats = calc.integrate(likelihood, prior.logpdf, sample_size=100000)

# convert the logstderr to anesthetic style
from scipy.special import logsumexp

err = np.abs(
    stats["log_integral"]
    - logsumexp((stats["log_integral"], stats["log_stderr"]))
)

print(f"IS integral: {stats['log_integral'] :.3f} +/- {err :.3f}")
print(f"IS efficiency: {stats['efficiency'] :.3f}")



print(f"NS integral: {Zs.mean():.3f} +/- {Zs.std() :.3f}")
print(f"NS efficiency: {pc_out.nequals/pc_out.nlike :.3f}")


print("Done")
