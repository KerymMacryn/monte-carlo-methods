"""
Tests for Gauge Module
======================

Unit tests for gauge coupling calculations.
"""

import pytest
import numpy as np
from tsqvt.gauge import (
    compute_C4_coefficients,
    compute_gauge_couplings,
    StandardModelGauge,
    GaugeCoupling,
    C4Calculator,
)


class TestC4Coefficients:
    """Tests for C_4 coefficient calculations."""
    
    def test_c4_positivity(self):
        """C_4 coefficients should be positive."""
        C4 = compute_C4_coefficients()
        
        assert C4['U1'] > 0
        assert C4['SU2'] > 0
        assert C4['SU3'] > 0
    
    def test_c4_hierarchy(self):
        """C_4 coefficients reflect fermion content."""
        C4 = compute_C4_coefficients()
        
        # U(1) has largest coefficient due to hypercharge sum
        # The actual hierarchy depends on SM fermion quantum numbers
        assert C4['U1'] > C4['SU3']
    
    def test_c4_generation_scaling(self):
        """C_4 should scale with number of generations."""
        C4_3gen = compute_C4_coefficients(n_generations=3)
        C4_4gen = compute_C4_coefficients(n_generations=4)
        
        # 4 generations should give larger coefficients
        assert C4_4gen['U1'] > C4_3gen['U1']
        assert C4_4gen['SU2'] > C4_3gen['SU2']
        assert C4_4gen['SU3'] > C4_3gen['SU3']
    
    def test_c4_calculator_consistency(self):
        """Calculator and function should give same results."""
        calc = C4Calculator(n_generations=3)
        C4_calc = calc.compute_all()
        C4_func = compute_C4_coefficients(n_generations=3)
        
        for group in ['U1', 'SU2', 'SU3']:
            assert np.isclose(C4_calc[group], C4_func[group], rtol=1e-10)


class TestGaugeCouplings:
    """Tests for gauge coupling calculations."""
    
    def test_alpha_em_order_of_magnitude(self):
        """α_em should be approximately 1/137."""
        couplings = compute_gauge_couplings(cutoff=2e16)
        
        alpha_inv = couplings['alpha_inverse']
        
        # Should be within 10% of experimental value
        assert 120 < alpha_inv < 150
    
    def test_sin2_theta_w_range(self):
        """sin²θ_W should be approximately 0.23."""
        couplings = compute_gauge_couplings(cutoff=2e16)
        
        sin2_tw = couplings['sin2_theta_w']
        
        # Should be in reasonable range
        assert 0.20 < sin2_tw < 0.26
    
    def test_alpha_s_range(self):
        """α_s should be approximately 0.12."""
        couplings = compute_gauge_couplings(cutoff=2e16)
        
        alpha_s = couplings['alpha_s']
        
        # Should be in reasonable range
        assert 0.08 < alpha_s < 0.16
    
    def test_mw_mz_ratio(self):
        """M_W/M_Z should be approximately 0.88."""
        couplings = compute_gauge_couplings(cutoff=2e16)
        
        ratio = couplings['mw_mz_ratio']
        
        # Should be close to experimental value
        assert 0.85 < ratio < 0.92


class TestStandardModelGauge:
    """Tests for StandardModelGauge class."""
    
    def test_compute_runs(self):
        """compute() should run without errors."""
        sm = StandardModelGauge(cutoff=2e16)
        sm.compute()
        
        assert len(sm.C4) == 3
        assert len(sm.couplings_gut) == 3
        assert len(sm.couplings_mz) == 3
    
    def test_coupling_running(self):
        """Couplings should run from GUT to M_Z."""
        sm = StandardModelGauge(cutoff=2e16)
        sm.compute()
        
        # α_3 should increase at lower energies (asymptotic freedom)
        alpha3_gut = sm.couplings_gut['SU3'].alpha
        alpha3_mz = sm.couplings_mz['SU3'].alpha
        
        assert alpha3_mz > alpha3_gut
    
    def test_summary(self):
        """summary() should return all observables."""
        sm = StandardModelGauge(cutoff=2e16)
        sm.compute()
        summary = sm.summary()
        
        required_keys = ['alpha_em_inv', 'sin2_theta_w', 'alpha_s', 'mw_mz_ratio']
        for key in required_keys:
            assert key in summary
    
    def test_comparison_with_experiment(self):
        """compare_experiment() should return comparison dict."""
        sm = StandardModelGauge(cutoff=2e16)
        sm.compute()
        comparison = sm.compare_experiment()
        
        # All comparisons should have error < 50%
        for key, data in comparison.items():
            assert data['error_percent'] < 50


class TestGaugeCoupling:
    """Tests for GaugeCoupling dataclass."""
    
    def test_alpha_computation(self):
        """α = g²/(4π)."""
        g = GaugeCoupling(value=0.5, group='test')
        
        expected_alpha = 0.5**2 / (4 * np.pi)
        assert np.isclose(g.alpha, expected_alpha)
    
    def test_alpha_inverse(self):
        """α⁻¹ = 4π/g²."""
        g = GaugeCoupling(value=0.5, group='test')
        
        assert np.isclose(g.alpha * g.alpha_inverse, 1.0)
    
    def test_repr(self):
        """String representation should be informative."""
        g = GaugeCoupling(value=0.357, uncertainty=0.001, scale=91.2, group='SU2')
        
        repr_str = repr(g)
        assert 'SU2' in repr_str
        assert '0.357' in repr_str


class TestPredictionAccuracy:
    """Tests for TSQVT prediction accuracy."""
    
    @pytest.fixture
    def sm_predictions(self):
        """Compute SM predictions."""
        sm = StandardModelGauge(cutoff=2e16)
        sm.compute()
        return sm
    
    def test_alpha_em_accuracy(self, sm_predictions):
        """α⁻¹(M_Z) prediction should be within 2% of experiment."""
        pred = sm_predictions.alpha_em_inverse('mz')
        # At M_Z, α_em⁻¹ ≈ 127.9 (not 137 which is at q²=0)
        exp = 127.9
        
        error = abs(pred - exp) / exp
        assert error < 0.02, f"α⁻¹ error {error*100:.1f}% exceeds 2%"
    
    def test_sin2_theta_w_accuracy(self, sm_predictions):
        """sin²θ_W prediction should be within 5% of experiment."""
        pred = sm_predictions.sin2_theta_w('mz')
        exp = 0.23122
        
        error = abs(pred - exp) / exp
        assert error < 0.05, f"sin²θ_W error {error*100:.1f}% exceeds 5%"
    
    def test_alpha_s_accuracy(self, sm_predictions):
        """α_s prediction should be within 10% of experiment."""
        pred = sm_predictions.alpha_s('mz')
        exp = 0.1179
        
        error = abs(pred - exp) / exp
        assert error < 0.10, f"α_s error {error*100:.1f}% exceeds 10%"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
