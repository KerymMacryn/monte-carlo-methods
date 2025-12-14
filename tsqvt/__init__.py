"""
TSQVT: Twistorial Spectral Quantum Vacuum Theory
=================================================

A Python package for deriving Standard Model parameters from spectral geometry.

The package implements the complete computational pipeline for TSQVT:
- Spectral manifold construction
- Condensation field dynamics
- Gauge coupling derivation via spectral action
- Renormalization group running
- Experimental predictions

Quick Start
-----------
>>> import tsqvt
>>> from tsqvt.core import SpectralManifold
>>> from tsqvt.gauge import compute_gauge_couplings
>>> 
>>> manifold = SpectralManifold()
>>> couplings = compute_gauge_couplings(manifold)
>>> print(f"α⁻¹ = {1/couplings['alpha']:.2f}")

Modules
-------
core
    Core theoretical framework: SpectralManifold, CondensationField, KreinSpace
spectral
    Spectral action formalism: heat kernel, Dirac operators, finite geometry
gauge
    Gauge coupling calculations: C4 coefficients, projectors, SM algebra
rg
    Renormalization group: running, thresholds, matching
experimental
    Experimental predictions: collapse time, BEC, metamaterials
utils
    Utilities: physical constants, plotting functions

References
----------
.. [1] Makraini, K. (2025). Emergent Lorentzian Spacetime and Gauge Dynamics 
       from Twistorial Spectral Data. Next Research (Elsevier).
.. [2] Connes, A. (1994). Noncommutative Geometry. Academic Press.
.. [3] Chamseddine, A. H., & Connes, A. (1997). The spectral action principle. 
       Comm. Math. Phys., 186, 731-750.
"""

__version__ = "1.0.0"
__author__ = "Kerym Makraini"
__email__ = "kerym.makraini@example.com"
__license__ = "MIT"

# Core imports
from tsqvt.core import SpectralManifold, CondensationField, KreinSpace
from tsqvt.gauge import compute_gauge_couplings, compute_C4_coefficients
from tsqvt.rg import RGRunner

# Version info
def get_version():
    """Return the package version."""
    return __version__

# Package info
def info():
    """Print package information."""
    print(f"TSQVT v{__version__}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")
    print("\nModules: core, spectral, gauge, rg, experimental, utils")

__all__ = [
    # Version
    "__version__",
    "get_version",
    "info",
    # Core
    "SpectralManifold",
    "CondensationField", 
    "KreinSpace",
    # Gauge
    "compute_gauge_couplings",
    "compute_C4_coefficients",
    # RG
    "RGRunner",
]
