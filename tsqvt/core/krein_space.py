"""
Krein Space
===========

Implementation of Krein spaces with indefinite inner product for
twistorial structure in TSQVT.

Krein spaces generalize Hilbert spaces by allowing indefinite metrics,
essential for incorporating the signature of twistor space.

References
----------
.. [1] Penrose, R. & Rindler, W. (1984). Spinors and Space-Time, Vol. 2.
.. [2] Bognar, J. (1974). Indefinite Inner Product Spaces. Springer.
"""

import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings


@dataclass
class KreinSpace:
    """
    Krein space with indefinite inner product.
    
    A Krein space is a vector space with an indefinite Hermitian form
    ⟨·,·⟩_K that can take both positive and negative values.
    
    For TSQVT, the Krein structure arises from the twistor metric:
        η = diag(+1, +1, -1, -1)  (signature (2,2))
    
    Parameters
    ----------
    dimension : int
        Total dimension of the space. Default: 4.
    signature : tuple of int
        Number of (positive, negative) eigenvalues of the metric.
        Default: (2, 2) for twistor space.
    
    Attributes
    ----------
    metric : ndarray
        The indefinite metric tensor η.
    is_definite : bool
        Whether the space is actually definite (degenerate case).
    
    Examples
    --------
    >>> krein = KreinSpace(dimension=4, signature=(2, 2))
    >>> v = np.array([1, 0, 1, 0])
    >>> print(f"⟨v,v⟩_K = {krein.inner_product(v, v)}")
    ⟨v,v⟩_K = 0.0
    """
    
    dimension: int = 4
    signature: Tuple[int, int] = (2, 2)
    
    # Computed attributes
    metric: np.ndarray = field(init=False, repr=False)
    is_definite: bool = field(init=False)
    
    def __post_init__(self):
        """Initialize metric tensor and validate parameters."""
        p, q = self.signature
        
        if p + q != self.dimension:
            raise ValueError(
                f"Signature ({p}, {q}) incompatible with dimension {self.dimension}"
            )
        
        # Construct diagonal metric
        self.metric = np.diag(
            [1.0] * p + [-1.0] * q
        )
        
        self.is_definite = (p == 0 or q == 0)
    
    def inner_product(
        self, 
        v: np.ndarray, 
        w: np.ndarray
    ) -> Union[float, complex]:
        """
        Compute Krein inner product ⟨v, w⟩_K = v† η w.
        
        Parameters
        ----------
        v, w : ndarray
            Vectors in the Krein space.
        
        Returns
        -------
        float or complex
            The indefinite inner product.
        
        Notes
        -----
        Unlike Hilbert spaces, ⟨v,v⟩_K can be negative or zero for v ≠ 0.
        """
        v = np.asarray(v)
        w = np.asarray(w)
        
        if v.shape[-1] != self.dimension or w.shape[-1] != self.dimension:
            raise ValueError(
                f"Vector dimension must match space dimension {self.dimension}"
            )
        
        return np.dot(np.conj(v), np.dot(self.metric, w))
    
    def norm_squared(self, v: np.ndarray) -> float:
        """
        Compute squared Krein norm ⟨v,v⟩_K.
        
        Parameters
        ----------
        v : ndarray
            Vector in the Krein space.
        
        Returns
        -------
        float
            Squared norm (can be negative, zero, or positive).
        """
        return float(np.real(self.inner_product(v, v)))
    
    def classify_vector(self, v: np.ndarray) -> str:
        """
        Classify vector by its Krein norm.
        
        Parameters
        ----------
        v : ndarray
            Vector to classify.
        
        Returns
        -------
        str
            'positive', 'negative', or 'null' (lightlike).
        """
        norm_sq = self.norm_squared(v)
        
        if np.abs(norm_sq) < 1e-10:
            return 'null'
        elif norm_sq > 0:
            return 'positive'
        else:
            return 'negative'
    
    def project_positive(self, v: np.ndarray) -> np.ndarray:
        """
        Project vector onto positive-definite subspace.
        
        Parameters
        ----------
        v : ndarray
            Vector to project.
        
        Returns
        -------
        ndarray
            Projection onto the positive subspace.
        """
        p, q = self.signature
        v = np.asarray(v)
        
        result = np.zeros_like(v)
        result[:p] = v[:p]
        
        return result
    
    def project_negative(self, v: np.ndarray) -> np.ndarray:
        """
        Project vector onto negative-definite subspace.
        
        Parameters
        ----------
        v : ndarray
            Vector to project.
        
        Returns
        -------
        ndarray
            Projection onto the negative subspace.
        """
        p, q = self.signature
        v = np.asarray(v)
        
        result = np.zeros_like(v)
        result[p:] = v[p:]
        
        return result
    
    def is_orthogonal(
        self, 
        v: np.ndarray, 
        w: np.ndarray, 
        tol: float = 1e-10
    ) -> bool:
        """
        Check if two vectors are Krein-orthogonal.
        
        Parameters
        ----------
        v, w : ndarray
            Vectors to check.
        tol : float
            Tolerance for numerical comparison.
        
        Returns
        -------
        bool
            True if ⟨v,w⟩_K = 0.
        """
        return np.abs(self.inner_product(v, w)) < tol
    
    def fundamental_decomposition(
        self, 
        v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose vector into positive and negative parts.
        
        Every vector v in a Krein space admits a unique decomposition:
            v = v_+ + v_-
        where v_+ is in the positive subspace and v_- in the negative.
        
        Parameters
        ----------
        v : ndarray
            Vector to decompose.
        
        Returns
        -------
        tuple of ndarray
            (v_positive, v_negative) components.
        """
        return self.project_positive(v), self.project_negative(v)
    
    def gramian(self, vectors: np.ndarray) -> np.ndarray:
        """
        Compute Gram matrix for a set of vectors.
        
        G_ij = ⟨v_i, v_j⟩_K
        
        Parameters
        ----------
        vectors : ndarray
            Array of shape (n_vectors, dimension).
        
        Returns
        -------
        ndarray
            Gram matrix of shape (n_vectors, n_vectors).
        """
        vectors = np.asarray(vectors)
        n = vectors.shape[0]
        
        G = np.zeros((n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                G[i, j] = self.inner_product(vectors[i], vectors[j])
        
        return G
    
    def is_krein_unitary(
        self, 
        U: np.ndarray, 
        tol: float = 1e-10
    ) -> bool:
        """
        Check if matrix U is Krein-unitary (U† η U = η).
        
        Parameters
        ----------
        U : ndarray
            Square matrix to check.
        tol : float
            Tolerance for comparison.
        
        Returns
        -------
        bool
            True if U preserves the Krein inner product.
        """
        U = np.asarray(U)
        
        if U.shape != (self.dimension, self.dimension):
            return False
        
        # Check U† η U = η
        result = np.conj(U.T) @ self.metric @ U
        
        return np.allclose(result, self.metric, atol=tol)
    
    def twistor_to_spacetime(
        self, 
        Z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert twistor to spacetime point (incidence relation).
        
        For a twistor Z^α = (ω^A, π_A'), the incidence relation is:
            ω^A = i x^{AA'} π_A'
        
        Parameters
        ----------
        Z : ndarray
            Twistor with 4 components (ω^0, ω^1, π_0', π_1').
        
        Returns
        -------
        tuple
            (x, flag) where x is spacetime point and flag indicates if real.
        
        Notes
        -----
        Not all twistors correspond to real spacetime points.
        Null twistors (⟨Z,Z⟩_K = 0) do correspond to real points.
        """
        Z = np.asarray(Z)
        
        omega = Z[:2]  # ω^A
        pi = Z[2:]     # π_A'
        
        # Check if null (real spacetime point)
        is_null = self.classify_vector(Z) == 'null'
        
        if np.abs(pi[0]) > 1e-10:
            # Standard patch
            zeta = pi[1] / pi[0]
            x_plus = omega[1] / pi[0]
            x_minus = -omega[0] / pi[0]
            
            # Reconstruct spacetime coordinates
            # x^{AA'} = (t+z, x+iy; x-iy, t-z)
            t = np.real(x_plus + x_minus) / 2
            z = np.real(x_plus - x_minus) / 2
            x = np.real((omega[0] + omega[1]) / (2 * pi[0]))
            y = np.imag((omega[0] - omega[1]) / (2 * pi[0]))
            
            spacetime = np.array([t, x, y, z])
        else:
            # Point at infinity
            spacetime = np.array([np.inf, np.inf, np.inf, np.inf])
            is_null = False
        
        return spacetime, is_null
    
    def spacetime_to_twistor(
        self, 
        x: np.ndarray, 
        helicity: int = 1
    ) -> np.ndarray:
        """
        Convert spacetime point to twistor (with chosen helicity).
        
        Parameters
        ----------
        x : ndarray
            Spacetime point (t, x, y, z).
        helicity : int
            Helicity quantum number (+1 or -1).
        
        Returns
        -------
        ndarray
            Twistor Z^α corresponding to the point.
        """
        x = np.asarray(x)
        t, x_coord, y, z = x
        
        # Choose reference spinor based on helicity
        if helicity > 0:
            pi = np.array([1.0, 0.0])
        else:
            pi = np.array([0.0, 1.0])
        
        # Incidence relation: ω^A = i x^{AA'} π_A'
        # x^{AA'} matrix:
        x_matrix = np.array([
            [t + z, x + 1j*y],
            [x - 1j*y, t - z]
        ])
        
        omega = 1j * x_matrix @ pi
        
        return np.concatenate([omega, pi])
    
    def __repr__(self) -> str:
        return f"KreinSpace(dim={self.dimension}, signature={self.signature})"
