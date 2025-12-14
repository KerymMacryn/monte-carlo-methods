"""
Utilities Module
================

Physical constants, plotting functions, and helper utilities.

Constants
---------
HBAR, C, G : float
    Fundamental physical constants.
M_PLANCK, M_Z, M_W : float
    Characteristic mass scales.
ALPHA_EM : float
    Fine structure constant.

Functions
---------
plot_rg_running
    Plot RG evolution of couplings.
plot_comparison
    Compare TSQVT predictions with experiment.
"""

from tsqvt.utils.constants import (
    HBAR, C, G, K_B,
    M_PLANCK, M_Z, M_W, M_HIGGS,
    ALPHA_EM, ALPHA_S, SIN2_THETA_W,
    V_HIGGS, G_FERMI,
    SM_MASSES, SM_MIXINGS,
)

__all__ = [
    # Fundamental constants
    "HBAR", "C", "G", "K_B",
    # Mass scales
    "M_PLANCK", "M_Z", "M_W", "M_HIGGS",
    # Couplings
    "ALPHA_EM", "ALPHA_S", "SIN2_THETA_W",
    # Electroweak
    "V_HIGGS", "G_FERMI",
    # SM data
    "SM_MASSES", "SM_MIXINGS",
]
