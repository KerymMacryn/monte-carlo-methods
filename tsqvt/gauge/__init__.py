"""
Gauge Module
============

Gauge coupling calculations from spectral action.

This module computes gauge coupling constants from the spectral
action coefficients C_4^{(a)} following the Chamseddine-Connes formula.

Functions
---------
compute_gauge_couplings
    Compute all SM gauge couplings from spectral data.
compute_C4_coefficients
    Compute C_4 coefficients for each gauge factor.

Classes
-------
GaugeCoupling
    Container for gauge coupling with uncertainty.
StandardModelGauge
    Full Standard Model gauge structure.

References
----------
.. [1] Chamseddine, A. H., & Connes, A. (1997). The spectral action principle.
.. [2] Chamseddine, A. H., & Connes, A. (2012). Resilience of the Spectral SM.
"""

from tsqvt.gauge.coefficients import compute_C4_coefficients, C4Calculator
from tsqvt.gauge.projectors import GaugeProjector, compute_projectors
from tsqvt.gauge.standard_model import (
    compute_gauge_couplings,
    StandardModelGauge,
    GaugeCoupling,
)

__all__ = [
    "compute_C4_coefficients",
    "C4Calculator",
    "GaugeProjector",
    "compute_projectors",
    "compute_gauge_couplings",
    "StandardModelGauge",
    "GaugeCoupling",
]
