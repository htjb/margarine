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

chains = ns.read_chains("chains/gaussian", columns=np.arange(nDims))
from margarine.maf import MAF
from margarine.marginal_stats import calculate

flow = MAF.load("maf.pkl")

calc = calculate(flow)

stats = calc.integrate(likelihood)
print("Done")
