"""
Gauge Projectors
================

Projectors onto gauge group representations in the finite Hilbert space.

These projectors P_a are used in computing C_4^{(a)} coefficients:
    C_4^{(a)} ∝ Tr(P_a · M(D_F))

References
----------
.. [1] van Suijlekom, W. D. (2015). NCG and Particle Physics, Ch. 11.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class GaugeProjector:
    """
    Projector onto gauge representation subspace.
    
    Parameters
    ----------
    gauge_group : str
        'U1', 'SU2', or 'SU3'.
    hilbert_dim : int
        Dimension of finite Hilbert space.
    n_generations : int
        Number of fermion generations.
    
    Attributes
    ----------
    projector : ndarray
        The projector matrix P_a.
    
    Examples
    --------
    >>> proj = GaugeProjector('SU3', hilbert_dim=96)
    >>> P = proj.projector
    >>> print(f"Rank: {np.trace(P):.0f}")
    """
    
    gauge_group: str
    hilbert_dim: int = 96
    n_generations: int = 3
    
    def __post_init__(self):
        """Construct projector matrix."""
        self.projector = self._build_projector()
    
    def _build_projector(self) -> np.ndarray:
        """Build projector onto charged particles."""
        N = self.hilbert_dim
        P = np.zeros((N, N), dtype=complex)
        
        # Fermion structure per generation:
        # [e_R, nu_L, e_L, nu_R] (leptons: 4)
        # [u_R(3), d_R(3), u_L(3), d_L(3)] (quarks: 12)
        # Total per gen: 16, plus antiparticles: 32
        
        fermions_per_gen = N // self.n_generations
        
        for gen in range(self.n_generations):
            base = gen * fermions_per_gen
            
            if self.gauge_group == 'U1':
                # All charged fermions
                self._add_charged_U1(P, base, fermions_per_gen)
            
            elif self.gauge_group == 'SU2':
                # Doublets only
                self._add_doublets_SU2(P, base, fermions_per_gen)
            
            elif self.gauge_group == 'SU3':
                # Quarks only
                self._add_triplets_SU3(P, base, fermions_per_gen)
        
        return P
    
    def _add_charged_U1(self, P: np.ndarray, base: int, size: int):
        """Add U(1) charged particles to projector."""
        # All fermions have hypercharge except possibly nu_R
        for i in range(min(size, P.shape[0] - base)):
            idx = base + i
            if idx < P.shape[0]:
                # Weight by Y²
                Y = self._get_hypercharge(i, size)
                P[idx, idx] = Y**2
    
    def _add_doublets_SU2(self, P: np.ndarray, base: int, size: int):
        """Add SU(2) doublets to projector."""
        # Doublets: L (leptons), Q (quarks)
        # Simplified: indices 1-2 (L), 6-11 (Q)
        doublet_indices = [1, 2]  # L
        doublet_indices += list(range(6, 12))  # Q (6 = 2×3)
        
        for i in doublet_indices:
            idx = base + i
            if idx < P.shape[0]:
                P[idx, idx] = 1.0
    
    def _add_triplets_SU3(self, P: np.ndarray, base: int, size: int):
        """Add SU(3) triplets (quarks) to projector."""
        # Quarks: u_R (3), d_R (3), Q (6)
        triplet_indices = list(range(4, 16))  # All quark indices
        
        for i in triplet_indices:
            idx = base + i
            if idx < P.shape[0]:
                P[idx, idx] = 1.0
    
    def _get_hypercharge(self, local_idx: int, size: int) -> float:
        """Get hypercharge for particle at local index."""
        # Simplified mapping
        charges = {
            0: -1,      # e_R
            1: -1/2,    # nu_L
            2: -1/2,    # e_L
            3: 0,       # nu_R
            4: 2/3,     # u_R (×3)
            5: 2/3,
            6: 2/3,
            7: -1/3,    # d_R (×3)
            8: -1/3,
            9: -1/3,
            10: 1/6,    # Q (×6)
            11: 1/6,
            12: 1/6,
            13: 1/6,
            14: 1/6,
            15: 1/6,
        }
        return charges.get(local_idx % 16, 0)
    
    def trace_with_operator(self, O: np.ndarray) -> complex:
        """
        Compute Tr(P_a · O).
        
        Parameters
        ----------
        O : ndarray
            Operator matrix.
        
        Returns
        -------
        complex
            Trace value.
        """
        return np.trace(self.projector @ O)
    
    def rank(self) -> int:
        """Return rank of projector (number of charged particles)."""
        return int(np.round(np.real(np.trace(self.projector))))


def compute_projectors(
    hilbert_dim: int = 96,
    n_generations: int = 3
) -> Dict[str, GaugeProjector]:
    """
    Compute projectors for all SM gauge groups.
    
    Parameters
    ----------
    hilbert_dim : int
        Dimension of H_F.
    n_generations : int
        Number of generations.
    
    Returns
    -------
    dict
        Dictionary {'U1': P_U1, 'SU2': P_SU2, 'SU3': P_SU3}.
    """
    return {
        'U1': GaugeProjector('U1', hilbert_dim, n_generations),
        'SU2': GaugeProjector('SU2', hilbert_dim, n_generations),
        'SU3': GaugeProjector('SU3', hilbert_dim, n_generations),
    }


def hypercharge_generator(hilbert_dim: int = 96, n_gen: int = 3) -> np.ndarray:
    """
    Construct U(1)_Y hypercharge generator Q.
    
    Parameters
    ----------
    hilbert_dim : int
        Dimension of H_F.
    n_gen : int
        Number of generations.
    
    Returns
    -------
    ndarray
        Diagonal matrix with hypercharges.
    """
    Q = np.zeros((hilbert_dim, hilbert_dim), dtype=complex)
    
    # Hypercharges per particle type (16 per generation)
    Y_values = [
        -1,      # e_R
        -1/2,    # nu_L
        -1/2,    # e_L
        0,       # nu_R
        2/3, 2/3, 2/3,      # u_R (3 colors)
        -1/3, -1/3, -1/3,   # d_R (3 colors)
        1/6, 1/6, 1/6,      # u_L (3 colors)
        1/6, 1/6, 1/6,      # d_L (3 colors)
    ]
    
    for gen in range(n_gen):
        for i, Y in enumerate(Y_values):
            idx = gen * len(Y_values) + i
            if idx < hilbert_dim:
                Q[idx, idx] = Y
    
    return Q
