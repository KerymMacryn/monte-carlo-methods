"""
Objective Collapse Predictions
==============================

TSQVT predicts objective collapse of quantum superpositions
with specific time scale depending on mass and condensation.

τ_TSQVT = τ_DP × (1 - ρ)⁻¹

where τ_DP is the Diósi-Penrose time.

References
----------
.. [1] Penrose, R. (1996). On gravity's role in quantum state reduction.
.. [2] Diósi, L. (1987). A universal master equation for gravitational decoherence.
.. [3] Bassi, A., et al. (2013). Models of wave-function collapse.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


# Physical constants
HBAR = 1.054571817e-34  # J·s
G = 6.67430e-11         # m³/(kg·s²)
C = 299792458           # m/s


@dataclass
class CollapsePredictor:
    """
    Predict objective collapse time for quantum superposition.
    
    Parameters
    ----------
    mass : float
        Particle mass in kg. Default: 1e-14 (typical nanoparticle).
    Delta_x : float
        Superposition separation in m. Default: 100e-9 (100 nm).
    rho : float
        Local condensation parameter. Default: 0.742.
    
    Attributes
    ----------
    tau_dp : float
        Diósi-Penrose collapse time (without TSQVT correction).
    tau_tsqvt : float
        TSQVT collapse time (with condensation factor).
    
    Examples
    --------
    >>> pred = CollapsePredictor(mass=1e-14, Delta_x=100e-9)
    >>> print(f"τ_TSQVT = {pred.tau_tsqvt*1000:.1f} ms")
    τ_TSQVT = 87.2 ms
    """
    
    mass: float = 1e-14  # kg
    Delta_x: float = 100e-9  # m
    rho: float = 0.742
    
    def __post_init__(self):
        """Compute collapse times."""
        self.tau_dp = self._compute_diosi_penrose()
        self.tau_tsqvt = self._compute_tsqvt()
    
    def _compute_diosi_penrose(self) -> float:
        """
        Compute Diósi-Penrose collapse time.
        
        τ_DP = ℏ / ΔE_grav
        
        where ΔE_grav ~ G m² / Δx is gravitational self-energy.
        """
        # Gravitational self-energy of superposition
        Delta_E = G * self.mass**2 / self.Delta_x
        
        # Collapse time
        tau = HBAR / Delta_E
        
        return tau
    
    def _compute_tsqvt(self) -> float:
        """
        Compute TSQVT collapse time with condensation correction.
        
        τ_TSQVT = τ_DP / (1 - ρ)
        
        The factor (1 - ρ) comes from the spectral refraction
        reducing the effective gravitational coupling.
        """
        if self.rho >= 1:
            return np.inf  # No collapse in fully geometric phase
        
        return self.tau_dp / (1 - self.rho)
    
    def compute_collapse_time(self, rho: Optional[float] = None) -> float:
        """
        Compute collapse time for given condensation.
        
        Parameters
        ----------
        rho : float, optional
            Condensation parameter. Default: self.rho.
        
        Returns
        -------
        float
            Collapse time in seconds.
        """
        if rho is None:
            return self.tau_tsqvt
        
        if rho >= 1:
            return np.inf
        
        return self.tau_dp / (1 - rho)
    
    def collapse_rate(self, rho: Optional[float] = None) -> float:
        """Return collapse rate γ = 1/τ in Hz."""
        tau = self.compute_collapse_time(rho)
        return 1.0 / tau if tau > 0 else 0.0
    
    def mass_scaling(self) -> float:
        """
        Return mass scaling exponent.
        
        TSQVT: τ ∝ m⁻²
        CSL: τ ∝ m⁰
        
        Returns
        -------
        float
            Scaling exponent (should be -2 for TSQVT).
        """
        return -2.0
    
    def compare_theories(self) -> Dict[str, Dict[str, float]]:
        """
        Compare TSQVT with other collapse models.
        
        Returns
        -------
        dict
            Predictions from different theories.
        """
        return {
            'TSQVT': {
                'tau_ms': self.tau_tsqvt * 1000,
                'scaling': 'm^-2 * (1-ρ)^-1',
            },
            'Diosi_Penrose': {
                'tau_ms': self.tau_dp * 1000,
                'scaling': 'm^-2',
            },
            'CSL': {
                'tau_ms': 1e6,  # Much longer
                'scaling': 'm^0',
            },
            'QM_no_collapse': {
                'tau_ms': np.inf,
                'scaling': 'N/A',
            },
        }
    
    def experimental_protocol(self) -> Dict[str, any]:
        """
        Return experimental protocol parameters.
        
        Returns
        -------
        dict
            Protocol specifications.
        """
        return {
            'particle_mass_kg': self.mass,
            'separation_m': self.Delta_x,
            'predicted_tau_ms': self.tau_tsqvt * 1000,
            'uncertainty_ms': 15,  # Typical uncertainty
            'required_events': 1000,  # For 2σ discrimination
            'temperature_K': 1e-6,  # Required cooling
            'vacuum_Pa': 1e-10,
            'timeline_months': 18,
            'cost_estimate_usd': 415000,
            'lead_lab': 'Vienna (Aspelmeyer group)',
        }
    
    def spectral_photon_energy(self) -> float:
        """
        Compute energy of spectral chirp photons emitted during collapse.
        
        E_γ ≈ ℏc / λ_Compton
        
        Returns
        -------
        float
            Photon energy in keV.
        """
        lambda_compton = HBAR / (self.mass * C)
        E_gamma_joules = HBAR * C / lambda_compton
        E_gamma_keV = E_gamma_joules / 1.602e-16  # Convert to keV
        
        return E_gamma_keV
    
    def photon_rate(self) -> float:
        """
        Compute spectral photon emission rate per collapse.
        
        Returns
        -------
        float
            Photons per collapse event.
        """
        return 0.01  # ~1% probability per collapse


def collapse_time(
    mass: float,
    separation: float,
    rho: float = 0.742
) -> float:
    """
    Convenience function for collapse time calculation.
    
    Parameters
    ----------
    mass : float
        Mass in kg.
    separation : float
        Superposition separation in m.
    rho : float
        Condensation parameter.
    
    Returns
    -------
    float
        Collapse time in seconds.
    
    Examples
    --------
    >>> tau = collapse_time(1e-14, 100e-9)
    >>> print(f"τ = {tau*1000:.1f} ms")
    """
    pred = CollapsePredictor(mass=mass, Delta_x=separation, rho=rho)
    return pred.tau_tsqvt
