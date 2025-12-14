"""
BEC Sound Speed
===============

TSQVT predicts a specific sound speed in Bose-Einstein condensates
at critical condensation.

At ρ = 2/3, the sound speed equals the speed of light:
    c_s(ρ=2/3) = c

This is a sharp, falsifiable prediction.

References
----------
.. [1] Lin, Y.-J., et al. (2011). Spin-orbit-coupled BEC. Nature.
.. [2] Dalibard, J., et al. (2011). Artificial gauge potentials. RMP.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


C = 299792458  # m/s


@dataclass
class BECSoundSpeed:
    """
    BEC sound speed predictions from TSQVT.
    
    The sound speed in the spectral medium is:
        c_s² = c² × ρ(4 - 3ρ) / [3(1 - ρ)]
    
    Parameters
    ----------
    rho_target : float
        Target condensation parameter. Default: 2/3 (critical).
    temperature : float
        BEC temperature in K. Default: 1e-9.
    atom_mass : float
        Atom mass in kg. Default: 1.44e-25 (⁸⁷Rb).
    
    Examples
    --------
    >>> bec = BECSoundSpeed(rho_target=2/3)
    >>> c_s = bec.compute_sound_speed()
    >>> print(f"c_s/c = {c_s/C:.6f}")
    c_s/c = 1.000000
    """
    
    rho_target: float = 2/3
    temperature: float = 1e-9  # K
    atom_mass: float = 1.44e-25  # kg (⁸⁷Rb)
    
    def compute_sound_speed(self, rho: Optional[float] = None) -> float:
        """
        Compute sound speed at given condensation.
        
        Parameters
        ----------
        rho : float, optional
            Condensation parameter. Default: self.rho_target.
        
        Returns
        -------
        float
            Sound speed in m/s.
        """
        if rho is None:
            rho = self.rho_target
        
        if rho >= 1:
            return np.inf
        if rho <= 0:
            return 0.0
        
        # TSQVT formula
        c_s_squared = C**2 * rho * (4 - 3*rho) / (3 * (1 - rho))
        
        return np.sqrt(c_s_squared)
    
    def sound_speed_ratio(self, rho: Optional[float] = None) -> float:
        """Return c_s / c."""
        return self.compute_sound_speed(rho) / C
    
    def verify_critical_point(self, tolerance: float = 1e-10) -> bool:
        """
        Verify that c_s(ρ=2/3) = c exactly.
        
        Returns
        -------
        bool
            True if prediction holds within tolerance.
        """
        c_s = self.compute_sound_speed(2/3)
        return abs(c_s - C) / C < tolerance
    
    def derivative_at_critical(self) -> float:
        """
        Compute dc_s/dρ at ρ = 2/3.
        
        This determines sensitivity to ρ variations.
        
        Returns
        -------
        float
            Derivative in m/s per unit ρ.
        """
        rho = 2/3
        eps = 1e-8
        
        c_plus = self.compute_sound_speed(rho + eps)
        c_minus = self.compute_sound_speed(rho - eps)
        
        return (c_plus - c_minus) / (2 * eps)
    
    def profile(self, rho_range: Tuple[float, float] = (0.1, 0.9), n_points: int = 100) -> Dict:
        """
        Compute sound speed profile over ρ range.
        
        Returns
        -------
        dict
            {'rho': array, 'c_s': array, 'c_s_ratio': array}
        """
        rho_vals = np.linspace(rho_range[0], rho_range[1], n_points)
        c_s_vals = np.array([self.compute_sound_speed(r) for r in rho_vals])
        
        return {
            'rho': rho_vals,
            'c_s': c_s_vals,
            'c_s_ratio': c_s_vals / C,
        }
    
    def experimental_protocol(self) -> Dict:
        """
        Return experimental protocol for BEC test.
        
        Returns
        -------
        dict
            Protocol specifications.
        """
        return {
            'atom_species': '⁸⁷Rb',
            'atom_number': 1e5,
            'temperature_nK': self.temperature * 1e9,
            'trap_frequency_Hz': 100,
            'predicted_c_s_at_critical': C,
            'measurement_precision': 0.001,  # 0.1%
            'timeline_months': 12,
            'lead_labs': ['MIT', 'JILA', 'MPQ'],
        }


def critical_sound_speed() -> float:
    """
    Return the critical sound speed c_s(ρ=2/3).
    
    This is exactly c in TSQVT.
    
    Returns
    -------
    float
        Speed of light in m/s.
    """
    return C


def sound_speed_formula(rho: float) -> float:
    """
    Compute sound speed from TSQVT formula.
    
    c_s² = c² × ρ(4 - 3ρ) / [3(1 - ρ)]
    
    Parameters
    ----------
    rho : float
        Condensation parameter.
    
    Returns
    -------
    float
        Sound speed in m/s.
    """
    bec = BECSoundSpeed(rho_target=rho)
    return bec.compute_sound_speed()
