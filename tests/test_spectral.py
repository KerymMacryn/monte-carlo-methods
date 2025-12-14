"""
Tests for Spectral Module
=========================
"""

import pytest
import numpy as np
from tsqvt.spectral import HeatKernel, DiracOperator, FiniteDirac, FiniteGeometry


class TestHeatKernel:
    """Tests for HeatKernel class."""
    
    def test_default_creation(self):
        """Test default heat kernel creation."""
        hk = HeatKernel()
        assert hk.dimension == 4
    
    def test_a0_positive(self):
        """Test a_0 coefficient is positive."""
        hk = HeatKernel(dimension=4)
        a0 = hk.compute_a0()
        assert a0 > 0
    
    def test_a2_flat_space(self):
        """Test a_2 vanishes for flat space with no potential."""
        hk = HeatKernel(dimension=4)
        a2 = hk.compute_a2(R=0, E=0)
        assert a2 == 0
    
    def test_a4_gauge_contribution(self):
        """Test a_4 has gauge field contribution."""
        hk = HeatKernel(dimension=4)
        a4_no_gauge = hk.compute_a4(Omega=0)
        a4_with_gauge = hk.compute_a4(Omega=1.0)
        assert a4_with_gauge > a4_no_gauge


class TestDiracOperator:
    """Tests for DiracOperator class."""
    
    def test_gamma_matrices_dimension(self):
        """Test gamma matrices have correct dimension."""
        D = DiracOperator()
        for gamma in D.gamma:
            assert gamma.shape == (4, 4)
    
    def test_clifford_algebra(self):
        """Test Clifford algebra relations."""
        D = DiracOperator()
        assert D.verify_clifford()
    
    def test_gamma5_properties(self):
        """Test γ^5 properties."""
        D = DiracOperator()
        
        # (γ^5)² = 1
        g5_sq = D.gamma5 @ D.gamma5
        np.testing.assert_array_almost_equal(g5_sq, np.eye(4))
    
    def test_chirality_projectors(self):
        """Test chirality projectors."""
        D = DiracOperator()
        
        P_L = D.project_left()
        P_R = D.project_right()
        
        # P_L + P_R = 1
        np.testing.assert_array_almost_equal(P_L + P_R, np.eye(4))
        
        # P_L² = P_L
        np.testing.assert_array_almost_equal(P_L @ P_L, P_L)


class TestFiniteDirac:
    """Tests for FiniteDirac class."""
    
    def test_dimension(self):
        """Test finite Dirac dimension."""
        Df = FiniteDirac(n_generations=3, n_fermions_per_gen=32)
        assert Df.dimension == 96
    
    def test_hermiticity(self):
        """Test D_F matrices are Hermitian."""
        Df = FiniteDirac()
        
        np.testing.assert_array_almost_equal(Df.D0, Df.D0.conj().T)
        np.testing.assert_array_almost_equal(Df.D1, Df.D1.conj().T)
    
    def test_evaluate_at_rho(self):
        """Test evaluation at different ρ values."""
        Df = FiniteDirac()
        
        D_0 = Df.evaluate(0)
        D_half = Df.evaluate(0.5)
        D_1 = Df.evaluate(1)
        
        # D(0) should just be D_0
        np.testing.assert_array_almost_equal(D_0, Df.D0)
    
    def test_eigenvalues_real(self):
        """Test eigenvalues are real for Hermitian operator."""
        Df = FiniteDirac()
        eigs = Df.eigenvalues(0.5)
        
        assert np.all(np.isreal(eigs))


class TestFiniteGeometry:
    """Tests for FiniteGeometry class."""
    
    def test_hilbert_dim(self):
        """Test Hilbert space dimension."""
        geom = FiniteGeometry(n_generations=3)
        assert geom.hilbert_dim == 96
    
    def test_real_structure(self):
        """Test J² = 1."""
        geom = FiniteGeometry()
        J_sq = geom.J @ geom.J
        np.testing.assert_array_almost_equal(J_sq, np.eye(geom.hilbert_dim))
    
    def test_grading_squared(self):
        """Test γ² = 1."""
        geom = FiniteGeometry()
        gamma_sq = geom.gamma @ geom.gamma
        np.testing.assert_array_almost_equal(gamma_sq, np.eye(geom.hilbert_dim))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
