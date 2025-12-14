"""
Core Module
===========

Core theoretical framework for TSQVT.

Classes
-------
SpectralManifold
    The spectral manifold Σ_spec carrying eigenvalue data.
CondensationField
    The condensation parameter ρ(x,t) governing phase transitions.
KreinSpace
    Krein space with indefinite inner product for twistorial structure.

Examples
--------
>>> from tsqvt.core import SpectralManifold, CondensationField
>>> manifold = SpectralManifold(volume=1.85e-61)
>>> field = CondensationField(vev=0.742)
>>> print(f"Twist angle: {manifold.twist_angle:.3f} rad")
"""

from tsqvt.core.spectral_manifold import SpectralManifold
from tsqvt.core.condensation import CondensationField
from tsqvt.core.krein_space import KreinSpace

__all__ = [
    "SpectralManifold",
    "CondensationField",
    "KreinSpace",
]
