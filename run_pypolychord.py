from numpy import pi, log
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior

try:
    from mpi4py import MPI
except ImportError:
    pass


# | Define a four-dimensional spherical gaussian likelihood,
# | width sigma=0.1, centered on the 0 with one derived parameter.
# | The derived parameter is the squared radius

nDims = 4
nDerived = 0
sigma = 0.1


def likelihood(theta):
    """Simple Gaussian Likelihood"""

    nDims = len(theta)
    r2 = sum(theta**2)
    logL = -log(2 * pi * sigma * sigma) * nDims / 2.0
    logL += -r2 / 2 / sigma / sigma

    return logL, []


# | Define a box uniform prior from -1 to 1


def prior(hypercube):
    """Uniform prior from [-1,1]^D."""
    return UniformPrior(0, 1)(hypercube)


# | Optional dumper function giving run-time read access to
# | the live points, dead points, weights and evidences


def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])


# | Initialise the settings

settings = PolyChordSettings(nDims, nDerived)
settings.file_root = "gaussian"
settings.nlive = 200
settings.do_clustering = True
settings.read_resume = False

# | Run PolyChord

output = pypolychord.run_polychord(
    likelihood, nDims, nDerived, settings, prior, dumper
)
