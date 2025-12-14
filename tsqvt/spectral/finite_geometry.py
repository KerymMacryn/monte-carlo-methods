"""
Finite Geometry
===============

Finite noncommutative geometry for the Standard Model algebra.

The internal space is described by the spectral triple:
    (A_F, H_F, D_F)

where A_F = C ⊕ H ⊕ M_3(C) is the Standard Model algebra.

References
----------
.. [1] Connes, A. (2006). Noncommutative geometry and the SM with neutrino mixing.
.. [2] Chamseddine, A. H., & Connes, A. (2012). Resilience of the Spectral SM.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class SMAlgebra:
    """
    Standard Model algebra A_F = C ⊕ H ⊕ M_3(C).
    
    The algebra encodes the gauge structure:
    - C: U(1) hypercharge
    - H (quaternions): SU(2) weak isospin
    - M_3(C): SU(3) color
    
    Parameters
    ----------
    n_generations : int
        Number of fermion generations. Default: 3.
    
    Attributes
    ----------
    generators : dict
        Generators for each simple factor.
    dimension : int
        Total algebra dimension.
    
    Examples
    --------
    >>> algebra = SMAlgebra()
    >>> print(f"Generators: {list(algebra.generators.keys())}")
    Generators: ['U1', 'SU2', 'SU3']
    """
    
    n_generations: int = 3
    
    # Computed
    generators: Dict[str, List[np.ndarray]] = field(default_factory=dict, init=False)
    dimension: int = field(init=False)
    
    def __post_init__(self):
        """Initialize algebra generators."""
        self.dimension = 1 + 4 + 9  # C + H + M_3(C)
        self._construct_generators()
    
    def _construct_generators(self):
        """Construct generators for each factor."""
        # U(1) generator (just 1)
        self.generators['U1'] = [np.array([[1.0]])]
        
        # SU(2) generators (Pauli matrices / 2)
        sigma_1 = np.array([[0, 1], [1, 0]], dtype=complex) / 2
        sigma_2 = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
        sigma_3 = np.array([[1, 0], [0, -1]], dtype=complex) / 2
        self.generators['SU2'] = [sigma_1, sigma_2, sigma_3]
        
        # SU(3) generators (Gell-Mann matrices / 2)
        lambda_1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex) / 2
        lambda_2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex) / 2
        lambda_3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex) / 2
        lambda_4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex) / 2
        lambda_5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex) / 2
        lambda_6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex) / 2
        lambda_7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex) / 2
        lambda_8 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex) / (2 * np.sqrt(3))
        
        self.generators['SU3'] = [lambda_1, lambda_2, lambda_3, lambda_4,
                                   lambda_5, lambda_6, lambda_7, lambda_8]
    
    def casimir(self, group: str) -> float:
        """
        Compute quadratic Casimir for a group.
        
        Parameters
        ----------
        group : str
            Group name ('U1', 'SU2', or 'SU3').
        
        Returns
        -------
        float
            Quadratic Casimir C_2(G).
        """
        casimirs = {
            'U1': 0,
            'SU2': 3/4,  # C_2(SU(2)) for fundamental
            'SU3': 4/3,  # C_2(SU(3)) for fundamental
        }
        return casimirs.get(group, 0)
    
    def dynkin_index(self, group: str, rep: str = 'fundamental') -> float:
        """
        Compute Dynkin index for a representation.
        
        T(R) δ^{ab} = Tr(T^a_R T^b_R)
        
        Parameters
        ----------
        group : str
            Group name.
        rep : str
            Representation name.
        
        Returns
        -------
        float
            Dynkin index T(R).
        """
        indices = {
            ('SU2', 'fundamental'): 1/2,
            ('SU2', 'adjoint'): 2,
            ('SU3', 'fundamental'): 1/2,
            ('SU3', 'adjoint'): 3,
        }
        return indices.get((group, rep), 0)
    
    def hypercharge(self, particle: str) -> float:
        """
        Get U(1)_Y hypercharge for a particle.
        
        Parameters
        ----------
        particle : str
            Particle name (e.g., 'eR', 'L', 'uR', 'dR', 'Q').
        
        Returns
        -------
        float
            Hypercharge Y.
        """
        charges = {
            'eR': -1,
            'L': -1/2,
            'nuR': 0,
            'uR': 2/3,
            'dR': -1/3,
            'Q': 1/6,
            'H': 1/2,
        }
        return charges.get(particle, 0)


@dataclass
class FiniteGeometry:
    """
    Finite noncommutative geometry for the Standard Model.
    
    The spectral triple (A_F, H_F, D_F, J, γ) satisfies:
    - KO-dimension 6 (mod 8)
    - J² = 1, Jγ = -γJ, JD = DJ
    - Order-one condition
    
    Parameters
    ----------
    n_generations : int
        Number of fermion generations. Default: 3.
    
    Attributes
    ----------
    algebra : SMAlgebra
        The Standard Model algebra.
    hilbert_dim : int
        Dimension of H_F.
    J : ndarray
        Real structure (charge conjugation).
    gamma : ndarray
        Grading (chirality).
    
    Examples
    --------
    >>> geom = FiniteGeometry()
    >>> print(f"Hilbert space dim: {geom.hilbert_dim}")
    Hilbert space dim: 96
    """
    
    n_generations: int = 3
    
    # Computed
    algebra: SMAlgebra = field(init=False)
    hilbert_dim: int = field(init=False)
    J: np.ndarray = field(init=False, repr=False)
    gamma: np.ndarray = field(init=False, repr=False)
    
    def __post_init__(self):
        """Initialize finite geometry."""
        self.algebra = SMAlgebra(self.n_generations)
        
        # H_F dimension: 32 per generation (16 particles + 16 antiparticles)
        self.hilbert_dim = 32 * self.n_generations
        
        self._construct_real_structure()
        self._construct_grading()
    
    def _construct_real_structure(self):
        """Construct real structure J (charge conjugation)."""
        N = self.hilbert_dim
        
        # J swaps particles and antiparticles
        # Simple structure: J = antidiag(1)
        self.J = np.zeros((N, N), dtype=complex)
        for i in range(N // 2):
            self.J[i, N - 1 - i] = 1
            self.J[N - 1 - i, i] = 1
    
    def _construct_grading(self):
        """Construct grading γ (chirality)."""
        N = self.hilbert_dim
        
        # γ = +1 for left-handed, -1 for right-handed
        # Block structure: first half left, second half right
        self.gamma = np.diag(
            [1.0] * (N // 2) + [-1.0] * (N // 2)
        )
    
    def verify_ko_dimension(self, D: np.ndarray, tol: float = 1e-10) -> Dict[str, bool]:
        """
        Verify KO-dimension conditions.
        
        For KO = 6:
        - J² = 1
        - JD = DJ
        - Jγ = -γJ
        - γD = -Dγ (anticommutation)
        
        Parameters
        ----------
        D : ndarray
            Finite Dirac operator.
        tol : float
            Tolerance for checks.
        
        Returns
        -------
        dict
            Results of each check.
        """
        results = {}
        
        # J² = 1
        J_sq = self.J @ self.J
        results['J_squared'] = np.allclose(J_sq, np.eye(self.hilbert_dim), atol=tol)
        
        # JD = DJ
        JD = self.J @ D
        DJ = D @ self.J
        results['JD_commutes'] = np.allclose(JD, DJ, atol=tol)
        
        # Jγ = -γJ
        J_gamma = self.J @ self.gamma
        gamma_J = self.gamma @ self.J
        results['J_gamma_anticommutes'] = np.allclose(J_gamma, -gamma_J, atol=tol)
        
        # γD = -Dγ (chirality anticommutes with D)
        gamma_D = self.gamma @ D
        D_gamma = D @ self.gamma
        results['gamma_D_anticommutes'] = np.allclose(gamma_D, -D_gamma, atol=tol)
        
        return results
    
    def particle_content(self) -> Dict[str, Dict]:
        """
        Return Standard Model particle content per generation.
        
        Returns
        -------
        dict
            Particle content with quantum numbers.
        """
        return {
            'leptons': {
                'e_R': {'SU3': '1', 'SU2': '1', 'Y': -1},
                'L': {'SU3': '1', 'SU2': '2', 'Y': -1/2},
                'nu_R': {'SU3': '1', 'SU2': '1', 'Y': 0},
            },
            'quarks': {
                'u_R': {'SU3': '3', 'SU2': '1', 'Y': 2/3},
                'd_R': {'SU3': '3', 'SU2': '1', 'Y': -1/3},
                'Q': {'SU3': '3', 'SU2': '2', 'Y': 1/6},
            },
        }
    
    def representation_projector(self, particle: str) -> np.ndarray:
        """
        Get projector onto a particle representation subspace.
        
        Parameters
        ----------
        particle : str
            Particle name.
        
        Returns
        -------
        ndarray
            Projector matrix.
        """
        N = self.hilbert_dim
        P = np.zeros((N, N), dtype=complex)
        
        # Map particle to indices (simplified)
        particle_indices = {
            'e_R': range(0, self.n_generations),
            'L': range(self.n_generations, 3 * self.n_generations),
            'nu_R': range(3 * self.n_generations, 4 * self.n_generations),
            'u_R': range(4 * self.n_generations, 7 * self.n_generations),
            'd_R': range(7 * self.n_generations, 10 * self.n_generations),
            'Q': range(10 * self.n_generations, 16 * self.n_generations),
        }
        
        if particle in particle_indices:
            for i in particle_indices[particle]:
                if i < N:
                    P[i, i] = 1.0
        
        return P
    
    def gauge_projector(self, gauge_group: str) -> np.ndarray:
        """
        Get projector for gauge group action.
        
        Parameters
        ----------
        gauge_group : str
            'U1', 'SU2', or 'SU3'.
        
        Returns
        -------
        ndarray
            Projector onto fermions charged under the group.
        """
        N = self.hilbert_dim
        P = np.zeros((N, N), dtype=complex)
        
        particles = self.particle_content()
        
        for category in ['leptons', 'quarks']:
            for particle, qnumbers in particles[category].items():
                # Check if particle transforms under this gauge group
                if gauge_group == 'U1':
                    if qnumbers['Y'] != 0:
                        P += self.representation_projector(particle)
                elif gauge_group == 'SU2':
                    if qnumbers['SU2'] == '2':
                        P += self.representation_projector(particle)
                elif gauge_group == 'SU3':
                    if qnumbers['SU3'] == '3':
                        P += self.representation_projector(particle)
        
        return P
    
    def compute_C4_coefficient(
        self,
        gauge_group: str,
        D_matrices: Dict[int, np.ndarray]
    ) -> float:
        """
        Compute C_4^{(a)} coefficient for gauge group a.
        
        C_4^{(a)} = (1/12) Σ_R k_R^{(a)} T_R^{(4)}
        
        Parameters
        ----------
        gauge_group : str
            'U1', 'SU2', or 'SU3'.
        D_matrices : dict
            Expansion matrices {0: D^{(0)}, 1: D^{(1)}, 2: D^{(2)}}.
        
        Returns
        -------
        float
            The C_4 coefficient.
        """
        # Get projector
        P = self.gauge_projector(gauge_group)
        
        # Get Dynkin index
        k = self.algebra.dynkin_index(gauge_group, 'fundamental')
        
        # Get D matrices
        D0 = D_matrices.get(0, np.eye(self.hilbert_dim))
        D1 = D_matrices.get(1, np.zeros((self.hilbert_dim, self.hilbert_dim)))
        
        # Heat kernel coefficients
        alpha_1 = 1.0
        alpha_2 = -0.5
        
        # Trace contribution
        T4 = np.trace(
            P @ (np.eye(self.hilbert_dim) + alpha_1 * D0 @ D0 + alpha_2 * D0 @ D1)
        )
        
        C4 = k * float(np.real(T4)) / 12
        
        return C4
    
    def unimodularity_residual(self, Q: np.ndarray) -> float:
        """
        Check unimodularity condition Tr_H_F(Q) = 0.
        
        Parameters
        ----------
        Q : ndarray
            U(1) generator.
        
        Returns
        -------
        float
            Residual |Tr(Q)|.
        """
        return abs(np.trace(Q))
    
    def __repr__(self) -> str:
        return f"FiniteGeometry(n_gen={self.n_generations}, dim_H={self.hilbert_dim})"
