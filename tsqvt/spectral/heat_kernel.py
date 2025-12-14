"""
Heat Kernel Expansion
=====================

Implementation of heat kernel expansion for spectral action computation.

The heat kernel K(t, x, y) of an operator D² satisfies:
    (∂_t + D²)K = 0,  K(0, x, y) = δ(x-y)

The trace has asymptotic expansion:
    Tr(exp(-tD²)) ~ Σ t^{(n-d)/2} a_n(D²)

References
----------
.. [1] Gilkey, P. B. (1995). Invariance Theory, the Heat Equation.
.. [2] Vassilevich, D. V. (2003). Heat kernel expansion: user's manual.
.. [3] Avramidi, I. G. (2000). Heat Kernel and Quantum Gravity.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass, field
import warnings


@dataclass
class HeatKernel:
    """
    Heat kernel for Dirac-type operators.
    
    Computes the Seeley-DeWitt coefficients a_{2n}(D²) that appear in
    the asymptotic expansion of the heat trace.
    
    Parameters
    ----------
    dimension : int
        Spacetime dimension. Default: 4.
    cutoff_function : callable, optional
        Cutoff function f(x) for spectral action. Default: exp(-x).
    
    Attributes
    ----------
    moments : dict
        Moments f_{2n} = ∫ t^{n-1} f̃(t) dt of the cutoff function.
    
    Examples
    --------
    >>> hk = HeatKernel(dimension=4)
    >>> a2 = hk.compute_a2(R=0, E=0)
    >>> print(f"a_2 = {a2:.4f}")
    """
    
    dimension: int = 4
    cutoff_function: Optional[Callable] = None
    
    # Computed attributes
    moments: Dict[int, float] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """Initialize cutoff function moments."""
        if self.cutoff_function is None:
            # Default: exponential cutoff f(x) = exp(-x)
            # Moments: f_{2n} = Γ(n) = (n-1)!
            self.moments = {
                0: 1.0,      # f_0 = 1
                2: 1.0,      # f_2 = Γ(1) = 1
                4: 1.0,      # f_4 = Γ(2) = 1
                6: 2.0,      # f_6 = Γ(3) = 2
                8: 6.0,      # f_8 = Γ(4) = 6
            }
        else:
            # Compute moments numerically
            self._compute_moments()
    
    def _compute_moments(self, n_max: int = 4):
        """Compute moments of cutoff function numerically."""
        from scipy.integrate import quad
        from scipy.fft import fft
        
        # f_{2n} = ∫_0^∞ t^{n-1} f̃(t) dt
        # where f̃ is Laplace transform of f
        
        for n in range(n_max + 1):
            def integrand(t):
                return t**(n - 1) * self.cutoff_function(t)
            
            result, _ = quad(integrand, 0, np.inf)
            self.moments[2*n] = result
    
    def compute_a0(self) -> float:
        """
        Compute a_0(D²) = (4π)^{-d/2} ∫ tr(1) dvol.
        
        Returns
        -------
        float
            The a_0 coefficient (proportional to volume).
        """
        # For unit volume: a_0 = (4π)^{-d/2} * dim(bundle)
        return (4 * np.pi) ** (-self.dimension / 2)
    
    def compute_a2(
        self, 
        R: float = 0.0, 
        E: float = 0.0,
        bundle_dim: int = 4
    ) -> float:
        """
        Compute a_2(D²) coefficient.
        
        a_2 = (4π)^{-d/2} * (1/6) ∫ tr(6E + R) dvol
        
        Parameters
        ----------
        R : float
            Scalar curvature.
        E : float
            Endomorphism term (potential).
        bundle_dim : int
            Dimension of the vector bundle.
        
        Returns
        -------
        float
            The a_2 coefficient.
        """
        prefactor = (4 * np.pi) ** (-self.dimension / 2) / 6
        return prefactor * bundle_dim * (6 * E + R)
    
    def compute_a4(
        self,
        R: float = 0.0,
        R_munu: float = 0.0,
        R_mnrs: float = 0.0,
        E: float = 0.0,
        Omega: float = 0.0,
        bundle_dim: int = 4
    ) -> float:
        """
        Compute a_4(D²) coefficient.
        
        This is the most important coefficient for gauge kinetic terms:
        a_4 = (4π)^{-d/2} * (1/360) ∫ tr(...) dvol
        
        Parameters
        ----------
        R : float
            Scalar curvature.
        R_munu : float
            Ricci tensor squared |Ric|².
        R_mnrs : float
            Riemann tensor squared |Riem|².
        E : float
            Endomorphism term.
        Omega : float
            Curvature of connection |F|².
        bundle_dim : int
            Dimension of the vector bundle.
        
        Returns
        -------
        float
            The a_4 coefficient.
        """
        prefactor = (4 * np.pi) ** (-self.dimension / 2) / 360
        
        # Full formula:
        # a_4 = (1/360)[60 E;μμ + 60 R E + 180 E² + 12 R;μμ + 5 R² 
        #       - 2 Ric² + 2 Riem² + 30 Ω²]
        
        # Simplified for flat space with gauge fields:
        geometric = 5 * R**2 - 2 * R_munu + 2 * R_mnrs
        matter = 60 * R * E + 180 * E**2
        gauge = 30 * Omega
        
        return prefactor * bundle_dim * (geometric + matter + gauge)
    
    def gauge_kinetic_coefficient(
        self,
        dynkin_index: float,
        n_fermions: int,
        rho_expansion: Dict[int, np.ndarray]
    ) -> float:
        """
        Compute gauge kinetic term coefficient from spectral data.
        
        C_4^{(a)} = (1/12) Σ_R k_R^{(a)} T_R^{(4)}
        
        Parameters
        ----------
        dynkin_index : float
            Dynkin index k_R^{(a)} for the representation.
        n_fermions : int
            Number of fermion species.
        rho_expansion : dict
            Expansion matrices D_F^{(j)} in condensation parameter.
        
        Returns
        -------
        float
            The C_4^{(a)} coefficient.
        """
        # Extract matrices
        D0 = rho_expansion.get(0, np.eye(n_fermions))
        D1 = rho_expansion.get(1, np.zeros((n_fermions, n_fermions)))
        D2 = rho_expansion.get(2, np.zeros((n_fermions, n_fermions)))
        
        # Universal coefficients α_i from heat kernel
        alpha_1 = 1.0
        alpha_2 = -1/2
        alpha_3 = 1/6
        
        # Trace contribution
        T4 = np.trace(
            np.eye(n_fermions) + 
            alpha_1 * D0 @ D0 +
            alpha_2 * D0 @ D1 +
            alpha_3 * D1 @ D1
        )
        
        C4 = dynkin_index * float(np.real(T4)) / 12
        
        return C4


def compute_seeley_dewitt(
    operator_data: Dict,
    order: int = 4,
    dimension: int = 4
) -> Dict[int, float]:
    """
    Compute Seeley-DeWitt coefficients a_{2n} up to given order.
    
    Parameters
    ----------
    operator_data : dict
        Dictionary containing:
        - 'R': scalar curvature
        - 'Ric': Ricci tensor components
        - 'Riem': Riemann tensor components
        - 'E': endomorphism term
        - 'Omega': connection curvature
        - 'bundle_dim': dimension of vector bundle
    order : int
        Maximum order 2n to compute. Default: 4.
    dimension : int
        Spacetime dimension.
    
    Returns
    -------
    dict
        Dictionary {0: a_0, 2: a_2, 4: a_4, ...}.
    
    Examples
    --------
    >>> data = {'R': 0, 'E': 0, 'Omega': 1.0, 'bundle_dim': 4}
    >>> coeffs = compute_seeley_dewitt(data, order=4)
    >>> print(f"a_4 = {coeffs[4]:.6f}")
    """
    hk = HeatKernel(dimension=dimension)
    
    R = operator_data.get('R', 0.0)
    E = operator_data.get('E', 0.0)
    Omega = operator_data.get('Omega', 0.0)
    bundle_dim = operator_data.get('bundle_dim', 4)
    
    # Ricci and Riemann squared (default to Euclidean values if not provided)
    Ric_sq = operator_data.get('Ric', 0.0)
    Riem_sq = operator_data.get('Riem', 0.0)
    
    coefficients = {}
    
    if order >= 0:
        coefficients[0] = hk.compute_a0()
    
    if order >= 2:
        coefficients[2] = hk.compute_a2(R, E, bundle_dim)
    
    if order >= 4:
        coefficients[4] = hk.compute_a4(R, Ric_sq, Riem_sq, E, Omega, bundle_dim)
    
    return coefficients


def spectral_action(
    cutoff: float,
    seeley_dewitt: Dict[int, float],
    moments: Optional[Dict[int, float]] = None,
    dimension: int = 4
) -> float:
    """
    Compute spectral action Tr f(D²/Λ²).
    
    The spectral action has asymptotic expansion:
        S = Σ f_{2n} Λ^{d-2n} a_{2n}(D²)
    
    Parameters
    ----------
    cutoff : float
        UV cutoff scale Λ in GeV.
    seeley_dewitt : dict
        Seeley-DeWitt coefficients {2n: a_{2n}}.
    moments : dict, optional
        Cutoff function moments {2n: f_{2n}}.
    dimension : int
        Spacetime dimension.
    
    Returns
    -------
    float
        The spectral action value.
    
    Examples
    --------
    >>> coeffs = {0: 1.0, 2: 0.1, 4: 0.01}
    >>> S = spectral_action(1e16, coeffs)
    >>> print(f"S = {S:.2e}")
    """
    if moments is None:
        # Default exponential cutoff moments
        moments = {0: 1.0, 2: 1.0, 4: 1.0, 6: 2.0}
    
    action = 0.0
    
    for n, a_2n in seeley_dewitt.items():
        if n in moments:
            f_2n = moments[n]
            power = dimension - n
            action += f_2n * (cutoff ** power) * a_2n
    
    return action


class SpectralActionExpansion:
    """
    Full spectral action expansion with ρ-dependent terms.
    
    For TSQVT, the spectral action includes contributions from
    the condensation field:
        S[D, ρ] = Σ f_{2n} Λ_eff(ρ)^{d-2n} a_{2n}(D²(ρ))
    
    Parameters
    ----------
    manifold : SpectralManifold
        The spectral manifold.
    field : CondensationField
        The condensation field.
    
    Examples
    --------
    >>> from tsqvt.core import SpectralManifold, CondensationField
    >>> manifold = SpectralManifold()
    >>> field = CondensationField(vev=0.742)
    >>> expansion = SpectralActionExpansion(manifold, field)
    >>> S = expansion.compute(cutoff=1e16)
    """
    
    def __init__(self, manifold, field):
        self.manifold = manifold
        self.field = field
        self.heat_kernel = HeatKernel(dimension=4)
    
    def effective_cutoff(self, rho: Optional[float] = None) -> float:
        """Compute ρ-dependent effective cutoff."""
        if rho is None:
            rho = self.field.vev
        return self.field.effective_cutoff * np.sqrt(rho / self.field.vev)
    
    def compute(
        self,
        cutoff: float,
        order: int = 4,
        include_fermions: bool = True
    ) -> Dict[str, float]:
        """
        Compute full spectral action expansion.
        
        Parameters
        ----------
        cutoff : float
            UV cutoff scale Λ in GeV.
        order : int
            Maximum order in heat kernel expansion.
        include_fermions : bool
            Whether to include fermionic contributions.
        
        Returns
        -------
        dict
            Dictionary with 'bosonic', 'fermionic', 'total' contributions.
        """
        rho = self.field.vev
        Lambda_eff = self.effective_cutoff(rho)
        
        # Bosonic spectral action (gravity + gauge)
        bosonic_data = {
            'R': 0,  # Flat space
            'E': 0,
            'Omega': 1.0,  # Gauge field contribution
            'bundle_dim': 4 * self.manifold.n_generations,
        }
        
        coeffs = compute_seeley_dewitt(bosonic_data, order=order)
        S_bosonic = spectral_action(Lambda_eff, coeffs, self.heat_kernel.moments)
        
        # Fermionic contribution (from Dirac action)
        S_fermionic = 0.0
        if include_fermions:
            # Simplified fermionic contribution
            S_fermionic = 0.1 * S_bosonic  # Placeholder
        
        return {
            'bosonic': S_bosonic,
            'fermionic': S_fermionic,
            'total': S_bosonic + S_fermionic,
            'effective_cutoff': Lambda_eff,
        }
