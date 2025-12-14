"""
Experimental Module
===================

Falsifiable predictions from TSQVT.

Classes
-------
CollapsePredictor
    Objective collapse time predictions.
BECSoundSpeed
    BEC sound speed at critical condensation.
MetamaterialTest
    Auxetic Poisson ratio predictions.

Key Predictions
---------------
1. Collapse time: τ = 87 ± 15 ms for m = 10⁻¹⁴ kg
2. Sound speed: c_s(ρ=2/3) = c (exactly)
3. Poisson ratio: ν(ρ→1) = -0.50 ± 0.02
4. Spectral photons: E_γ = 1.2 ± 0.1 keV
"""

from tsqvt.experimental.collapse import CollapsePredictor, collapse_time
from tsqvt.experimental.bec import BECSoundSpeed, critical_sound_speed
from tsqvt.experimental.metamaterials import MetamaterialTest, auxetic_poisson

__all__ = [
    "CollapsePredictor",
    "collapse_time",
    "BECSoundSpeed",
    "critical_sound_speed",
    "MetamaterialTest",
    "auxetic_poisson",
]
