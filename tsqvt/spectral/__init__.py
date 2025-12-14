"""
Spectral Module
===============

Spectral action formalism for TSQVT.

This module implements the heat kernel expansion and spectral action
computation following Chamseddine-Connes.

Classes
-------
HeatKernel
    Heat kernel expansion and Seeley-DeWitt coefficients.
DiracOperator
    Dirac operator construction from spectral data.
FiniteGeometry
    Finite noncommutative geometry (internal space).

Functions
---------
compute_seeley_dewitt
    Compute Seeley-DeWitt coefficients a_{2n}.
spectral_action
    Compute the full spectral action.

References
----------
.. [1] Chamseddine, A. H., & Connes, A. (1997). The spectral action principle.
.. [2] Gilkey, P. B. (1995). Invariance Theory, the Heat Equation, and the 
       Atiyah-Singer Index Theorem.
.. [3] Vassilevich, D. V. (2003). Heat kernel expansion: user's manual.
"""

from tsqvt.spectral.heat_kernel import HeatKernel, compute_seeley_dewitt
from tsqvt.spectral.dirac_operators import DiracOperator, FiniteDirac
from tsqvt.spectral.finite_geometry import FiniteGeometry, SMAlgebra

__all__ = [
    "HeatKernel",
    "compute_seeley_dewitt",
    "DiracOperator",
    "FiniteDirac",
    "FiniteGeometry",
    "SMAlgebra",
]
