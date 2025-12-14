"""
Tests for Core Module
=====================
"""

import pytest
import numpy as np
from tsqvt.core import SpectralManifold, CondensationField, KreinSpace


class TestSpectralManifold:
    """Tests for SpectralManifold class."""
    
    def test_default_creation(self):
        """Test default manifold creation."""
        manifold = SpectralManifold()
        assert manifold.dimension == 6
        assert manifold.n_generations == 3
    
    def test_custom_volume(self):
        """Test manifold with custom volume."""
        manifold = SpectralManifold(volume=1e-60)
        assert manifold.volume == 1e-60
    
    def test_euler_characteristic(self):
        """Test Euler characteristic computation."""
        manifold = SpectralManifold(hodge_numbers=(3, 243))
        assert manifold.euler_characteristic == 2 * (3 - 243)
    
    def test_area_ratio(self):
        """Test generation area ratios."""
        manifold = SpectralManifold()
        phi = (1 + np.sqrt(5)) / 2
        
        a1 = manifold.area_ratio(1)
        a2 = manifold.area_ratio(2)
        a3 = manifold.area_ratio(3)
        
        # Should follow golden ratio structure
        assert a2 / a1 == pytest.approx(phi, rel=0.01)
    
    def test_invalid_generation(self):
        """Test invalid generation raises error."""
        manifold = SpectralManifold()
        with pytest.raises(ValueError):
            manifold.area_ratio(4)


class TestCondensationField:
    """Tests for CondensationField class."""
    
    def test_default_creation(self):
        """Test default field creation."""
        field = CondensationField()
        assert field.vev == 0.742
        assert 0 < field.vev < 1
    
    def test_invalid_vev(self):
        """Test invalid vev raises error."""
        with pytest.raises(ValueError):
            CondensationField(vev=1.5)
        with pytest.raises(ValueError):
            CondensationField(vev=-0.1)
    
    def test_evaluate_constant(self):
        """Test constant field evaluation."""
        field = CondensationField(vev=0.5)
        x = np.array([0, 0, 0, 0])
        assert field.evaluate(x) == 0.5
    
    def test_effective_cutoff(self):
        """Test effective cutoff computation."""
        field = CondensationField(vev=0.5, planck_scale=1e19)
        # Λ_eff = Λ_Planck / √ρ = 1e19 / √0.5 ≈ 1.41e19
        expected = 1e19 / np.sqrt(0.5)
        assert field.effective_cutoff == pytest.approx(expected, rel=1e-6)
    
    def test_sound_speed_critical(self):
        """Test sound speed at critical point."""
        field = CondensationField()
        c_s_sq = field.sound_speed_squared(2/3)
        # At ρ = 2/3, c_s² = c² exactly
        assert c_s_sq == pytest.approx(1.0, rel=1e-10)
    
    def test_poisson_ratio_limit(self):
        """Test Poisson ratio at ρ → 1."""
        field = CondensationField()
        nu = field.poisson_ratio(0.999)
        assert nu == pytest.approx(-0.5, rel=0.01)
    
    def test_collapse_time_positive(self):
        """Test collapse time is positive."""
        field = CondensationField()
        tau = field.collapse_time(mass=1e-14, separation=1e-7)
        assert tau > 0


class TestKreinSpace:
    """Tests for KreinSpace class."""
    
    def test_default_creation(self):
        """Test default Krein space creation."""
        krein = KreinSpace()
        assert krein.dimension == 4
        assert krein.signature == (2, 2)
    
    def test_invalid_signature(self):
        """Test invalid signature raises error."""
        with pytest.raises(ValueError):
            KreinSpace(dimension=4, signature=(3, 3))
    
    def test_inner_product_signature(self):
        """Test inner product respects signature."""
        krein = KreinSpace(dimension=4, signature=(2, 2))
        
        v_pos = np.array([1, 0, 0, 0])
        v_neg = np.array([0, 0, 1, 0])
        
        assert krein.norm_squared(v_pos) > 0
        assert krein.norm_squared(v_neg) < 0
    
    def test_null_vector(self):
        """Test null vector classification."""
        krein = KreinSpace(dimension=4, signature=(2, 2))
        
        # v = (1, 0, 1, 0) is null in (2,2) signature
        v = np.array([1, 0, 1, 0])
        assert krein.classify_vector(v) == 'null'
    
    def test_orthogonality(self):
        """Test Krein orthogonality."""
        krein = KreinSpace(dimension=4, signature=(2, 2))
        
        v1 = np.array([1, 0, 0, 0])
        v2 = np.array([0, 1, 0, 0])
        
        assert krein.is_orthogonal(v1, v2)
    
    def test_fundamental_decomposition(self):
        """Test fundamental decomposition."""
        krein = KreinSpace(dimension=4, signature=(2, 2))
        
        v = np.array([1, 2, 3, 4])
        v_plus, v_minus = krein.fundamental_decomposition(v)
        
        # Should reconstruct original
        np.testing.assert_array_almost_equal(v_plus + v_minus, v)


class TestIntegration:
    """Integration tests for core module."""
    
    def test_manifold_field_interaction(self):
        """Test manifold and field work together."""
        manifold = SpectralManifold()
        field = CondensationField(vev=0.742)
        
        # Get generation-specific condensation
        rho_1 = field.fermion_condensation(1)
        rho_3 = field.fermion_condensation(3)
        
        # Generation 1 should be more spectral (lower ρ)
        assert rho_1 < rho_3
    
    def test_mass_suppression_hierarchy(self):
        """Test mass suppression follows generation hierarchy."""
        field = CondensationField()
        
        s1 = field.mass_suppression(1)
        s2 = field.mass_suppression(2)
        s3 = field.mass_suppression(3)
        
        # Generation 1 most suppressed
        assert s1 < s2 < s3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
