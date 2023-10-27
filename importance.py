import numpy as np

nDims = 4
nDerived = 0
sigma = 1.0


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
prior=multivariate_normal(np.ones(nDims), np.eye(nDims))
pc_out=PolyChordOutput("chains", "gaussian")
chains = ns.read_chains("chains/gaussian", columns=np.arange(nDims))
Zs=chains.logZ(100)
from margarine.maf import MAF
from margarine.marginal_stats import calculate

flow = MAF.load("maf.pkl")

calc = calculate(flow,prior_de=prior)

stats = calc.integrate(likelihood,sample_size=100000)

print(f"IS integral: {stats['integral'] :.8f} +/- {stats['stderr'] :.8f}")
print(f"IS efficiency: {stats['efficiency'] :.3f}")
print(f"IS log int: {stats['log_integral'] :.3}")


print(f"NS integral: {np.exp(Zs).mean():.8f} +/- {np.exp(Zs).std() :.8f}")
print(f"NS efficiency: {pc_out.nequals/pc_out.nlike :.3f}")
print(f"NS log int: {Zs.mean() :.3}")

print("Done")
