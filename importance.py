import numpy as np

nDims = 4
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
pc_out=PolyChordOutput("chains", "gaussian")
chains = ns.read_chains("chains/gaussian", columns=np.arange(nDims))
Zs=chains.logZ(100)
from margarine.maf import MAF
from margarine.marginal_stats import calculate

flow = MAF.load("maf.pkl")

calc = calculate(flow)

stats = calc.integrate(likelihood)

print(f"IS integral: {stats['integral'] :.3f} +/- {stats['stderr'] :.3f}")
print(f"IS efficiency: {stats['efficiency'] :.3f}")

print(f"NS integral: {np.exp(Zs).mean():.3f} +/- {np.exp(Zs).std() :.3f}")
print(f"NS efficiency: {pc_out.nequals/pc_out.nlike :.3f}")

print("Done")
