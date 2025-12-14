"""
Dirac Operators
===============

Construction of Dirac operators for TSQVT spectral triples.

The full Dirac operator has the form:
    D = D_M ⊗ 1_F + Γ ⊗ D_F(ρ)

where D_M acts on spacetime and D_F(ρ) is the finite internal operator.

References
----------
.. [1] Connes, A. (1994). Noncommutative Geometry. Academic Press.
.. [2] van Suijlekom, W. D. (2015). Noncommutative Geometry and Particle Physics.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field


@dataclass
class DiracOperator:
    """
    Spacetime Dirac operator D_M.
    
    The spacetime Dirac operator in curved space is:
        D_M = iγ^μ(∂_μ + ω_μ)
    
    where ω_μ is the spin connection.
    
    Parameters
    ----------
    dimension : int
        Spacetime dimension. Default: 4.
    signature : tuple
        Metric signature. Default: (1, 3) for Lorentzian.
    
    Attributes
    ----------
    gamma : list of ndarray
        Gamma matrices satisfying {γ^μ, γ^ν} = 2η^{μν}.
    gamma5 : ndarray
        Chirality operator (for even dimensions).
    
    Examples
    --------
    >>> D = DiracOperator(dimension=4)
    >>> print(D.gamma[0].shape)
    (4, 4)
    """
    
    dimension: int = 4
    signature: Tuple[int, int] = (1, 3)
    
    # Computed attributes
    gamma: List[np.ndarray] = field(default_factory=list, init=False)
    gamma5: np.ndarray = field(init=False)
    
    def __post_init__(self):
        """Construct gamma matrices."""
        self.gamma = self._construct_gamma_matrices()
        self.gamma5 = self._construct_gamma5()
    
    def _construct_gamma_matrices(self) -> List[np.ndarray]:
        """
        Construct gamma matrices in Weyl representation.
        
        γ^0 = (0, 1; 1, 0)
        γ^i = (0, σ^i; -σ^i, 0)
        """
        # Pauli matrices
        sigma_1 = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_3 = np.array([[1, 0], [0, -1]], dtype=complex)
        I2 = np.eye(2, dtype=complex)
        
        # 4x4 gamma matrices (Weyl/chiral representation)
        gamma_0 = np.block([[np.zeros((2, 2)), I2],
                           [I2, np.zeros((2, 2))]])
        
        gamma_1 = np.block([[np.zeros((2, 2)), sigma_1],
                           [-sigma_1, np.zeros((2, 2))]])
        
        gamma_2 = np.block([[np.zeros((2, 2)), sigma_2],
                           [-sigma_2, np.zeros((2, 2))]])
        
        gamma_3 = np.block([[np.zeros((2, 2)), sigma_3],
                           [-sigma_3, np.zeros((2, 2))]])
        
        return [gamma_0, gamma_1, gamma_2, gamma_3]
    
    def _construct_gamma5(self) -> np.ndarray:
        """Construct γ^5 = iγ^0γ^1γ^2γ^3."""
        g5 = 1j * self.gamma[0] @ self.gamma[1] @ self.gamma[2] @ self.gamma[3]
        return g5
    
    def anticommutator(self, mu: int, nu: int) -> np.ndarray:
        """Compute {γ^μ, γ^ν}."""
        return self.gamma[mu] @ self.gamma[nu] + self.gamma[nu] @ self.gamma[mu]
    
    def verify_clifford(self, tol: float = 1e-10) -> bool:
        """
        Verify Clifford algebra: {γ^μ, γ^ν} = 2η^{μν}.
        
        Returns
        -------
        bool
            True if Clifford relations are satisfied.
        """
        # Metric
        eta = np.diag([1, -1, -1, -1])
        
        for mu in range(4):
            for nu in range(4):
                anticomm = self.anticommutator(mu, nu)
                expected = 2 * eta[mu, nu] * np.eye(4)
                if not np.allclose(anticomm, expected, atol=tol):
                    return False
        return True
    
    def slash(self, p: np.ndarray) -> np.ndarray:
        """
        Compute Feynman slash p̸ = γ^μ p_μ.
        
        Parameters
        ----------
        p : ndarray
            4-momentum (E, px, py, pz).
        
        Returns
        -------
        ndarray
            The slashed momentum matrix.
        """
        p = np.asarray(p)
        result = np.zeros((4, 4), dtype=complex)
        
        # Raise index with metric
        p_up = np.array([p[0], -p[1], -p[2], -p[3]])
        
        for mu in range(4):
            result += self.gamma[mu] * p_up[mu]
        
        return result
    
    def project_left(self) -> np.ndarray:
        """Left-chiral projector P_L = (1 - γ^5)/2."""
        return (np.eye(4) - self.gamma5) / 2
    
    def project_right(self) -> np.ndarray:
        """Right-chiral projector P_R = (1 + γ^5)/2."""
        return (np.eye(4) + self.gamma5) / 2


@dataclass
class FiniteDirac:
    """
    Finite internal Dirac operator D_F(ρ).
    
    The finite Dirac operator acts on the internal Hilbert space
    and encodes Yukawa couplings and mass terms. It has ρ-expansion:
    
        D_F(ρ) = D_F^{(0)} + ρ D_F^{(1)} + ρ² D_F^{(2)} + O(ρ³)
    
    Parameters
    ----------
    n_generations : int
        Number of fermion generations. Default: 3.
    n_fermions_per_gen : int
        Fermions per generation. Default: 32 (SM with right-handed ν).
    
    Attributes
    ----------
    dimension : int
        Total dimension of internal Hilbert space.
    D0, D1, D2 : ndarray
        Expansion matrices in ρ.
    
    Examples
    --------
    >>> Df = FiniteDirac(n_generations=3)
    >>> D_rho = Df.evaluate(rho=0.742)
    >>> print(f"Eigenvalues: {np.linalg.eigvalsh(D_rho)[:5]}")
    """
    
    n_generations: int = 3
    n_fermions_per_gen: int = 32
    
    # Computed attributes
    dimension: int = field(init=False)
    D0: np.ndarray = field(init=False, repr=False)
    D1: np.ndarray = field(init=False, repr=False)
    D2: np.ndarray = field(init=False, repr=False)
    
    def __post_init__(self):
        """Initialize finite Dirac matrices."""
        self.dimension = self.n_generations * self.n_fermions_per_gen
        self._construct_matrices()
    
    def _construct_matrices(self):
        """Construct D_F^{(j)} matrices from spectral data."""
        N = self.dimension
        
        # D_F^{(0)}: Diagonal part (bare masses)
        self.D0 = np.zeros((N, N), dtype=complex)
        
        # D_F^{(1)}: Off-diagonal (Yukawa-like)
        self.D1 = np.zeros((N, N), dtype=complex)
        
        # D_F^{(2)}: Second-order corrections
        self.D2 = np.zeros((N, N), dtype=complex)
        
        # Populate with realistic structure
        self._populate_yukawa_structure()
    
    def _populate_yukawa_structure(self):
        """
        Populate matrices with SM-like Yukawa structure.
        
        The Yukawa matrix has hierarchical structure with
        golden ratio between generations.
        """
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # Characteristic scale
        v = 246  # GeV (Higgs VEV)
        
        # Fermion masses (in GeV) for each type
        # Charged leptons: e, μ, τ
        m_leptons = np.array([0.000511, 0.1057, 1.777])
        
        # Up-type quarks: u, c, t
        m_up = np.array([0.00216, 1.27, 172.7])
        
        # Down-type quarks: d, s, b
        m_down = np.array([0.00467, 0.093, 4.18])
        
        # Build Yukawa matrices
        Y_lepton = np.diag(m_leptons) / v * np.sqrt(2)
        Y_up = np.diag(m_up) / v * np.sqrt(2)
        Y_down = np.diag(m_down) / v * np.sqrt(2)
        
        # D_F^{(0)} contains bare mass terms
        # Block structure: [leptons | quarks] × [L | R] × [generations]
        
        # Simplified: diagonal mass insertions
        masses = np.concatenate([m_leptons, m_up, m_down])
        for i, m in enumerate(masses):
            idx = i * (self.n_fermions_per_gen // 9)
            if idx < self.dimension:
                self.D0[idx, idx] = m
        
        # D_F^{(1)} contains Yukawa couplings
        # Off-diagonal mixing
        for i in range(min(9, self.dimension)):
            for j in range(min(9, self.dimension)):
                if i != j:
                    # CKM-like mixing
                    mixing = 0.1 * phi ** (-abs(i - j))
                    self.D1[i, j] = mixing * np.sqrt(masses[i] * masses[j]) if i < len(masses) and j < len(masses) else 0
        
        # D_F^{(2)} contains second-order corrections
        self.D2 = 0.01 * self.D1 @ self.D1
        
        # Ensure Hermiticity
        self.D0 = (self.D0 + self.D0.conj().T) / 2
        self.D1 = (self.D1 + self.D1.conj().T) / 2
        self.D2 = (self.D2 + self.D2.conj().T) / 2
    
    def set_yukawa(
        self,
        yukawa_e: np.ndarray,
        yukawa_u: np.ndarray,
        yukawa_d: np.ndarray
    ):
        """
        Set Yukawa matrices explicitly.
        
        Parameters
        ----------
        yukawa_e : ndarray
            3×3 charged lepton Yukawa matrix.
        yukawa_u : ndarray
            3×3 up-type quark Yukawa matrix.
        yukawa_d : ndarray
            3×3 down-type quark Yukawa matrix.
        """
        # Store in D_F^{(1)} block structure
        # This is a simplified implementation
        v = 246  # GeV
        
        # Update diagonal blocks
        for i in range(3):
            self.D1[i, i] = yukawa_e[i, i] * v / np.sqrt(2)
            self.D1[i+3, i+3] = yukawa_u[i, i] * v / np.sqrt(2)
            self.D1[i+6, i+6] = yukawa_d[i, i] * v / np.sqrt(2)
    
    def evaluate(self, rho: float) -> np.ndarray:
        """
        Evaluate D_F(ρ) at given condensation value.
        
        Parameters
        ----------
        rho : float
            Condensation parameter value.
        
        Returns
        -------
        ndarray
            The finite Dirac operator matrix.
        """
        return self.D0 + rho * self.D1 + rho**2 * self.D2
    
    def eigenvalues(self, rho: float) -> np.ndarray:
        """Get eigenvalues of D_F(ρ)."""
        D = self.evaluate(rho)
        return np.linalg.eigvalsh(D)
    
    def mass_spectrum(self, rho: float, v_higgs: float = 246.0) -> Dict[str, np.ndarray]:
        """
        Compute fermion mass spectrum from D_F(ρ).
        
        Parameters
        ----------
        rho : float
            Condensation parameter.
        v_higgs : float
            Higgs VEV in GeV.
        
        Returns
        -------
        dict
            Dictionary with 'leptons', 'up_quarks', 'down_quarks' masses.
        """
        D = self.evaluate(rho)
        eigs = np.sort(np.abs(self.eigenvalues(rho)))
        
        # Extract masses from eigenvalue structure
        # Simplified: first 3 are leptons, next 3 up quarks, etc.
        n_types = min(3, len(eigs) // 3)
        
        return {
            'leptons': eigs[:n_types] if n_types > 0 else np.array([]),
            'up_quarks': eigs[n_types:2*n_types] if n_types > 0 else np.array([]),
            'down_quarks': eigs[2*n_types:3*n_types] if n_types > 0 else np.array([]),
        }
    
    def get_expansion_matrices(self) -> Dict[int, np.ndarray]:
        """Return expansion matrices as dictionary."""
        return {0: self.D0, 1: self.D1, 2: self.D2}
    
    def verify_order_one(self, algebra_generators: List[np.ndarray], tol: float = 1e-8) -> float:
        """
        Verify order-one condition: [[D_F, a], b°] = 0.
        
        Parameters
        ----------
        algebra_generators : list of ndarray
            Generators of the algebra A_F.
        tol : float
            Tolerance for residual.
        
        Returns
        -------
        float
            Maximum residual (should be < tol for valid D_F).
        """
        D = self.D0  # Check at leading order
        max_residual = 0.0
        
        for a in algebra_generators:
            for b in algebra_generators:
                # [D, a]
                comm_Da = D @ a - a @ D
                # [[D,a], b°] where b° is opposite algebra action
                b_op = b.conj().T
                double_comm = comm_Da @ b_op - b_op @ comm_Da
                
                residual = np.max(np.abs(double_comm))
                max_residual = max(max_residual, residual)
        
        return max_residual
    
    def __repr__(self) -> str:
        return f"FiniteDirac(n_gen={self.n_generations}, dim={self.dimension})"
