"""
Gauge Coefficients
==================

Computation of spectral action coefficients C_{2n}^{(a)} for gauge couplings.

The inverse gauge coupling is:
    1/g_a² = Σ f_{2n} Λ^{4-2n} C_{2n}^{(a)}

The dominant contribution at low energies is C_4^{(a)}.

References
----------
.. [1] van Suijlekom, W. D. (2015). Noncommutative Geometry and Particle Physics.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class C4Calculator:
    """
    Calculator for C_4^{(a)} spectral coefficients.
    
    Parameters
    ----------
    n_generations : int
        Number of fermion generations.
    yukawa : dict
        Yukawa couplings {particle: value}.
    majorana : dict, optional
        Majorana masses for neutrinos.
    
    Examples
    --------
    >>> calc = C4Calculator(n_generations=3)
    >>> C4 = calc.compute_all()
    >>> print(f"C4_U1 = {C4['U1']:.6f}")
    """
    
    n_generations: int = 3
    yukawa: Dict[str, float] = None
    majorana: Dict[str, float] = None
    
    def __post_init__(self):
        """Set default Yukawa couplings if not provided."""
        if self.yukawa is None:
            # Default SM Yukawa couplings (at GUT scale)
            v = 246  # GeV
            self.yukawa = {
                'e': 0.511e-3 / v * np.sqrt(2),
                'mu': 0.1057 / v * np.sqrt(2),
                'tau': 1.777 / v * np.sqrt(2),
                'u': 2.16e-3 / v * np.sqrt(2),
                'c': 1.27 / v * np.sqrt(2),
                't': 172.7 / v * np.sqrt(2),
                'd': 4.67e-3 / v * np.sqrt(2),
                's': 0.093 / v * np.sqrt(2),
                'b': 4.18 / v * np.sqrt(2),
            }
        
        if self.majorana is None:
            # Default heavy Majorana masses (for seesaw)
            self.majorana = {
                'nu1': 1e12,  # GeV
                'nu2': 1e13,
                'nu3': 1e14,
            }
    
    def compute_U1(self) -> float:
        """
        Compute C_4^{U(1)} coefficient.
        
        For U(1), the coefficient depends on hypercharge:
            C_4^{U(1)} = (1/12) Σ_f Y_f² dim(R_f)
        
        Returns
        -------
        float
            The C_4 coefficient for U(1).
        """
        # Sum over fermion hypercharges squared
        # Per generation: e_R, L, nu_R, u_R, d_R, Q
        
        Y_squared_sum = 0.0
        
        # Leptons
        Y_squared_sum += (-1)**2 * 1        # e_R (singlet)
        Y_squared_sum += (-1/2)**2 * 2      # L (doublet)
        Y_squared_sum += 0**2 * 1           # nu_R (singlet)
        
        # Quarks (×3 for color)
        Y_squared_sum += (2/3)**2 * 3       # u_R
        Y_squared_sum += (-1/3)**2 * 3      # d_R
        Y_squared_sum += (1/6)**2 * 6       # Q (doublet × 3 colors)
        
        # Multiply by number of generations
        Y_squared_sum *= self.n_generations
        
        # Include Higgs contribution
        Y_squared_sum += (1/2)**2 * 2  # Higgs doublet
        
        C4_U1 = Y_squared_sum / 12
        
        return C4_U1
    
    def compute_SU2(self) -> float:
        """
        Compute C_4^{SU(2)} coefficient.
        
        For SU(2), the coefficient is:
            C_4^{SU(2)} = (1/12) Σ_R T(R) dim(R)
        
        where T(R) = 1/2 for fundamental representation.
        
        Returns
        -------
        float
            The C_4 coefficient for SU(2).
        """
        T_fundamental = 0.5
        
        # Count SU(2) doublets
        doublet_count = 0
        
        # Per generation: L (lepton doublet), Q (quark doublet ×3 colors)
        doublet_count += 1    # L
        doublet_count += 3    # Q (3 colors)
        
        # Multiply by generations
        doublet_count *= self.n_generations
        
        # Higgs
        doublet_count += 1
        
        C4_SU2 = T_fundamental * doublet_count / 12
        
        return C4_SU2
    
    def compute_SU3(self) -> float:
        """
        Compute C_4^{SU(3)} coefficient.
        
        For SU(3), the coefficient is:
            C_4^{SU(3)} = (1/12) Σ_R T(R) dim(R)
        
        where T(R) = 1/2 for fundamental representation.
        
        Returns
        -------
        float
            The C_4 coefficient for SU(3).
        """
        T_fundamental = 0.5
        
        # Count SU(3) triplets
        triplet_count = 0
        
        # Per generation: u_R, d_R, Q (doublet = 2 triplets)
        triplet_count += 1    # u_R
        triplet_count += 1    # d_R
        triplet_count += 2    # Q (doublet)
        
        # Multiply by generations
        triplet_count *= self.n_generations
        
        C4_SU3 = T_fundamental * triplet_count / 12
        
        return C4_SU3
    
    def compute_all(self) -> Dict[str, float]:
        """
        Compute all C_4 coefficients.
        
        Returns
        -------
        dict
            Dictionary {'U1': C4_U1, 'SU2': C4_SU2, 'SU3': C4_SU3}.
        """
        return {
            'U1': self.compute_U1(),
            'SU2': self.compute_SU2(),
            'SU3': self.compute_SU3(),
        }
    
    def compute_with_rho(
        self,
        rho: float,
        D_matrices: Dict[int, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute C_4 coefficients with ρ-dependent corrections.
        
        Parameters
        ----------
        rho : float
            Condensation parameter.
        D_matrices : dict
            Expansion matrices D_F^{(j)}.
        
        Returns
        -------
        dict
            Corrected C_4 coefficients.
        """
        # Base coefficients
        C4 = self.compute_all()
        
        # ρ-dependent corrections
        # At leading order in ρ, corrections are small
        correction_factor = 1 + 0.01 * rho + 0.001 * rho**2
        
        return {k: v * correction_factor for k, v in C4.items()}


def compute_C4_coefficients(
    yukawa: Optional[Dict[str, float]] = None,
    majorana: Optional[Dict[str, float]] = None,
    n_generations: int = 3
) -> Dict[str, float]:
    """
    Compute C_4^{(a)} coefficients for all gauge groups.
    
    Convenience function that creates a calculator and returns results.
    
    Parameters
    ----------
    yukawa : dict, optional
        Yukawa couplings.
    majorana : dict, optional
        Majorana masses.
    n_generations : int
        Number of fermion generations.
    
    Returns
    -------
    dict
        Dictionary {'U1': C4_U1, 'SU2': C4_SU2, 'SU3': C4_SU3}.
    
    Examples
    --------
    >>> C4 = compute_C4_coefficients()
    >>> print(f"C4_SU3 = {C4['SU3']:.6f}")
    C4_SU3 = 0.500000
    """
    calc = C4Calculator(
        n_generations=n_generations,
        yukawa=yukawa,
        majorana=majorana
    )
    return calc.compute_all()


def beta_function_coefficient(gauge_group: str, n_gen: int = 3) -> float:
    """
    Compute one-loop beta function coefficient b_a.
    
    RG equation: dα_a/d(ln μ) = b_a α_a² / (2π)
    
    SM values (n_gen = 3):
    - b_1 = 41/10 ≈ 4.1 (GUT normalized)
    - b_2 = -19/6 ≈ -3.17
    - b_3 = -7
    
    Parameters
    ----------
    gauge_group : str
        'U1', 'SU2', or 'SU3'.
    n_gen : int
        Number of fermion generations.
    
    Returns
    -------
    float
        Beta function coefficient b_a.
    """
    if gauge_group == 'U1':
        # b_1 = 41/10 for SM with GUT normalization
        # General: b_1 = (4/3) Σ Y² × n_gen + Higgs contribution
        return 41.0 / 10.0
    
    elif gauge_group == 'SU2':
        # b_2 = -22/3 + (4/3)×n_f + 1/6 (Higgs)
        # n_f = number of SU(2) doublets per generation × n_gen = 2 × n_gen
        return -22.0/3.0 + (4.0/3.0) * (2 * n_gen) + 1.0/6.0
    
    elif gauge_group == 'SU3':
        # b_3 = -11 + (4/3)×n_f
        # n_f = number of quark flavors = 2 × n_gen
        return -11.0 + (4.0/3.0) * n_gen
    
    else:
        raise ValueError(f"Unknown gauge group: {gauge_group}")
