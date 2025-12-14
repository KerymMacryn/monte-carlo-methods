"""
Condensation Field
==================

Implementation of the condensation parameter ρ(x,t) that governs the
phase transition between spectral (ρ→0) and geometric (ρ→1) phases.

The condensation field is the central innovation of TSQVT, providing:
- UV regularization via ρ-dependent cutoff
- Mass hierarchy generation
- Objective collapse mechanism

References
----------
.. [1] Makraini, K. (2025). Emergent Lorentzian Spacetime from Twistorial Spectral Data.
"""

import numpy as np
from typing import Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
import warnings


@dataclass
class CondensationField:
    """
    Condensation parameter ρ(x,t) for TSQVT.
    
    The condensation field interpolates between:
    - ρ = 0: Pure spectral phase (no classical spacetime)
    - ρ = 1: Fully geometric phase (classical spacetime)
    
    Physical interpretation:
    - ρ controls the effective UV cutoff: Λ_eff = Λ_Planck / √ρ
    - ρ determines fermion masses via m_f ∝ exp(-S[ρ])
    - ρ dynamics generates objective collapse
    
    Parameters
    ----------
    vev : float
        Vacuum expectation value ⟨ρ⟩ at electroweak scale. Default: 0.742.
    critical_value : float
        Critical condensation ρ_c for phase transition. Default: 2/3.
    planck_scale : float
        Planck mass in GeV. Default: 1.22e19.
    
    Attributes
    ----------
    is_critical : bool
        Whether current vev is at critical point.
    effective_cutoff : float
        Effective UV cutoff Λ_eff in GeV.
    
    Examples
    --------
    >>> field = CondensationField(vev=0.742)
    >>> rho = field.evaluate(np.array([0, 0, 0, 0]))
    >>> print(f"ρ = {rho:.3f}")
    ρ = 0.742
    >>> print(f"Effective cutoff: {field.effective_cutoff:.2e} GeV")
    """
    
    vev: float = 0.742
    critical_value: float = 2/3
    planck_scale: float = 1.22e19  # GeV
    
    # Internal state
    _profile: Optional[Callable] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if not 0 < self.vev < 1:
            raise ValueError(f"vev must be in (0, 1), got {self.vev}")
        if not 0 < self.critical_value < 1:
            raise ValueError(f"critical_value must be in (0, 1), got {self.critical_value}")
    
    @property
    def is_critical(self) -> bool:
        """Check if vev is at critical point (within 1%)."""
        return abs(self.vev - self.critical_value) / self.critical_value < 0.01
    
    @property
    def effective_cutoff(self) -> float:
        """Effective UV cutoff Λ_eff = Λ_Planck / √ρ in GeV."""
        return self.planck_scale / np.sqrt(self.vev)
    
    def evaluate(
        self, 
        x: np.ndarray,
        t: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """
        Evaluate condensation field at spacetime point(s).
        
        Parameters
        ----------
        x : ndarray
            Spacetime coordinates. Shape (4,) for single point or (N, 4) for N points.
        t : float, optional
            Time coordinate (overrides x[0] if provided).
        
        Returns
        -------
        float or ndarray
            Condensation value(s) ρ(x,t).
        """
        x = np.asarray(x)
        
        # Use custom profile if set
        if self._profile is not None:
            return self._profile(x, t)
        
        # Default: constant vev with small fluctuations
        if x.ndim == 1:
            return self.vev
        else:
            return np.full(x.shape[0], self.vev)
    
    def set_profile(self, profile: Callable):
        """
        Set custom spatial profile for ρ(x,t).
        
        Parameters
        ----------
        profile : callable
            Function f(x, t) -> ρ taking coordinates and returning condensation.
        
        Examples
        --------
        >>> field = CondensationField()
        >>> field.set_profile(lambda x, t: 0.5 + 0.2 * np.sin(x[0]))
        """
        self._profile = profile
    
    def fermion_condensation(self, generation: int) -> float:
        """
        Get average condensation for a fermion generation.
        
        Different generations have different effective ⟨ρ⟩:
        - Generation 1 (e, u, d): Very spectral (ρ ~ 0.03)
        - Generation 2 (μ, c, s): Partially condensed (ρ ~ 0.10)
        - Generation 3 (τ, t, b): Near critical (ρ ~ 0.50)
        
        Parameters
        ----------
        generation : int
            Fermion generation (1, 2, or 3).
        
        Returns
        -------
        float
            Average condensation for that generation.
        """
        if generation not in [1, 2, 3]:
            raise ValueError("Generation must be 1, 2, or 3")
        
        # Empirical values from mass fits
        rho_gen = {
            1: 0.03,   # Electrons very spectral
            2: 0.10,   # Muons partially condensed
            3: 0.50,   # Taus near critical
        }
        
        return rho_gen[generation]
    
    def mass_suppression(self, generation: int) -> float:
        """
        Compute mass suppression factor exp(-S_eff[ρ]).
        
        First generation is most suppressed (smallest masses).
        
        Parameters
        ----------
        generation : int
            Fermion generation (1, 2, or 3).
        
        Returns
        -------
        float
            Mass suppression factor relative to EW scale.
        """
        rho_f = self.fermion_condensation(generation)
        
        # Effective action contribution
        # S_eff = (2π/α_weak) * (1 - ρ_f/ρ_c)
        # Smaller ρ_f → larger S_eff → more suppression
        alpha_weak = 1/30  # Approximate
        rho_c = self.critical_value
        
        # S_eff inversely related to condensation
        # Higher condensation → less suppression → heavier mass
        S_eff = (2 * np.pi / alpha_weak) * (1 - rho_f / rho_c)
        
        return np.exp(-S_eff)
    
    def sound_speed_squared(self, rho: Optional[float] = None) -> float:
        """
        Compute sound speed squared c_s² in spectral medium.
        
        TSQVT predicts: c_s² = c² exactly at ρ = 2/3.
        
        The formula is:
            c_s²/c² = (9/2) × ρ × (1 - ρ)
        
        This gives:
        - c_s² = 0 at ρ = 0 (spectral phase, no propagation)
        - c_s² = c² at ρ = 2/3 (critical point)
        - c_s² = 0 at ρ = 1 (geometric phase limit)
        
        Parameters
        ----------
        rho : float, optional
            Condensation value. Default: self.vev.
        
        Returns
        -------
        float
            Sound speed squared in units of c².
        """
        if rho is None:
            rho = self.vev
        
        if rho >= 1 or rho <= 0:
            return 0.0
        
        # Formula: (9/2) ρ (1-ρ) gives exactly 1 at ρ = 2/3
        return 4.5 * rho * (1 - rho)
    
    def collapse_rate(
        self, 
        mass: float, 
        separation: float,
        rho: Optional[float] = None
    ) -> float:
        """
        Compute objective collapse rate γ_TSQVT.
        
        The collapse rate depends on gravitational self-energy and condensation:
            γ = γ_0 * ΔE_grav / ℏ * (1 - ρ)
        
        Parameters
        ----------
        mass : float
            Particle mass in kg.
        separation : float
            Superposition separation in m.
        rho : float, optional
            Local condensation. Default: self.vev.
        
        Returns
        -------
        float
            Collapse rate in s⁻¹.
        """
        if rho is None:
            rho = self.vev
        
        # Physical constants
        G = 6.674e-11  # m³/(kg·s²)
        hbar = 1.055e-34  # J·s
        
        # Gravitational self-energy
        Delta_E = G * mass**2 / separation
        
        # TSQVT collapse rate (with condensation factor)
        gamma_0 = 1.0  # Dimensionless coupling (order 1)
        gamma = gamma_0 * Delta_E / hbar * (1 - rho)
        
        return gamma
    
    def collapse_time(
        self, 
        mass: float, 
        separation: float,
        rho: Optional[float] = None
    ) -> float:
        """
        Compute characteristic collapse time τ = 1/γ.
        
        Parameters
        ----------
        mass : float
            Particle mass in kg.
        separation : float
            Superposition separation in m.
        rho : float, optional
            Local condensation. Default: self.vev.
        
        Returns
        -------
        float
            Collapse time in seconds.
        """
        gamma = self.collapse_rate(mass, separation, rho)
        if gamma == 0:
            return np.inf
        return 1.0 / gamma
    
    def poisson_ratio(self, rho: Optional[float] = None) -> float:
        """
        Compute Poisson ratio of spectral medium.
        
        The Poisson ratio interpolates between normal and auxetic behavior:
            ν = 0.5 - ρ
        
        At ρ = 0: ν = 0.5 (incompressible limit)
        At ρ = 1: ν = -0.5 (auxetic limit)
        
        Parameters
        ----------
        rho : float, optional
            Condensation value. Default: self.vev.
        
        Returns
        -------
        float
            Poisson ratio.
        """
        if rho is None:
            rho = self.vev
        
        # Simple linear interpolation that gives correct limits
        return 0.5 - rho
    
    def to_dict(self) -> dict:
        """Convert field parameters to dictionary."""
        return {
            'vev': self.vev,
            'critical_value': self.critical_value,
            'planck_scale': self.planck_scale,
            'effective_cutoff': self.effective_cutoff,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CondensationField':
        """Create field from dictionary."""
        return cls(
            vev=data.get('vev', 0.742),
            critical_value=data.get('critical_value', 2/3),
            planck_scale=data.get('planck_scale', 1.22e19),
        )
    
    def __repr__(self) -> str:
        return f"CondensationField(vev={self.vev:.3f}, Λ_eff={self.effective_cutoff:.2e} GeV)"
