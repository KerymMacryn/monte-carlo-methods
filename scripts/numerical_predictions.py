#!/usr/bin/env python3
"""
TSQVT_numerical_predictions_CALIBRATED.py

Versión con coeficientes espectrales C4 calibrados desde observables experimentales.
Los C4 se derivan invirtiendo las ecuaciones RG desde α⁻¹(mZ) y sin²θW.

Autor: Kerym Macryn
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import math

# =============================================================================
# CONSTANTES FÍSICAS
# =============================================================================

M_Z = 91.1876           # GeV
ALPHA_INV_EXP = 137.035999084
SIN2_TW_EXP = 0.23122
ALPHA_S_MZ = 0.1180

K1 = 5.0 / 3.0          # Normalización GUT para U(1)
B1 = 41.0 / 10.0        # Beta coeff U(1) GUT-normalized
B2 = -19.0 / 6.0        # Beta coeff SU(2)


class UncertaintyAnalysis:
    """Análisis Monte Carlo con parámetros calibrados"""
    
    def __init__(self, n_samples=10000):
        self.n_samples = n_samples
        
        # =====================================================================
        # PARÁMETROS CALIBRADOS DESDE OBSERVABLES
        # =====================================================================
        # C4 derivados invirtiendo RG desde α⁻¹(mZ), sin²θW experimentales
        # Escala: A = 2×10¹⁶ GeV, f₄ = 0.25
        
        self.params = {
            # Geométricos de Σ_spec (estos NO afectan α, sin²θW directamente)
            'V_Sigma': (1.85e-61, 0.05e-61),      # m^4
            'theta_twist': (0.198, 0.002),        # rad
            'rho_EW': (0.742, 0.003),             # adimensional
            'xi_Yukawa': (2.34, 0.01),            # adimensional
            
            # Momentos cutoff function
            'f_0': (1.0, 0.05),
            'f_2': (0.5, 0.02),
            'f_4': (0.25, 0.01),                  # Fijado para consistencia
            'f_6': (0.125, 0.005),
            
            # *** COEFICIENTES ESPECTRALES CALIBRADOS ***
            # Derivados de: g²(A) = 1/(f₄ × C4) con A = 2×10¹⁶ GeV
            'C4_U1': (67.3216, 0.6732),           # ±1% (antes: 0.0234)
            'C4_SU2': (4.7883, 0.0479),           # ±1% (antes: 0.0123)
            'C4_SU3': (16.0, 0.16),               # Estimado (polo Landau en SM)
            
            # Escala GUT
            'A': (2.0e16, 0.1e16),                # GeV
        }
        
        self.results = {}

    def sample_parameters(self):
        """Generar muestras gaussianas de parámetros"""
        samples = {}
        for name, (mean, std) in self.params.items():
            samples[name] = np.random.normal(mean, std, self.n_samples)
        return samples
    
    def _rg_evolve_forward(self, g_A, b, A):
        """
        Evolución RG forward: g(A) → g(mZ)
        
        1/g(mZ)² = 1/g(A)² + (b/8π²) × ln(mZ/A)
        """
        L = math.log(M_Z / A)  # Negativo ya que mZ < A
        inv_g_A_sq = 1.0 / (g_A**2)
        inv_g_mZ_sq = inv_g_A_sq + (b / (8.0 * math.pi**2)) * L
        
        if inv_g_mZ_sq <= 0:
            return float('inf')  # Pole
        
        return 1.0 / math.sqrt(inv_g_mZ_sq)
    
    def compute_alpha_mZ(self, samples):
        """Calcular α⁻¹(mZ) para cada muestra"""
        alpha_invs = np.empty(self.n_samples)
        
        for i in range(self.n_samples):
            C4_U1 = samples['C4_U1'][i]
            C4_SU2 = samples['C4_SU2'][i]
            f4 = samples['f_4'][i]
            A = samples['A'][i]
            
            # g(A) desde coeficientes espectrales
            # g² = 1/(f₄ × C4)
            gY_A = 1.0 / math.sqrt(max(f4 * C4_U1, 1e-30))
            g1_A = math.sqrt(K1) * gY_A  # GUT normalization
            g2_A = 1.0 / math.sqrt(max(f4 * C4_SU2, 1e-30))
            
            # Evolucionar a mZ
            g1_mZ = self._rg_evolve_forward(g1_A, B1, A)
            g2_mZ = self._rg_evolve_forward(g2_A, B2, A)
            
            # e = g₁g₂ / √(g₁² + g₂²)
            e_mZ = g1_mZ * g2_mZ / math.sqrt(g1_mZ**2 + g2_mZ**2)
            alpha_mZ = e_mZ**2 / (4.0 * math.pi)
            alpha_invs[i] = 1.0 / alpha_mZ
        
        return alpha_invs
    
    def compute_sin2_thetaW(self, samples):
        """Calcular sin²θ_W para cada muestra"""
        sin2_values = np.empty(self.n_samples)
        
        for i in range(self.n_samples):
            C4_U1 = samples['C4_U1'][i]
            C4_SU2 = samples['C4_SU2'][i]
            f4 = samples['f_4'][i]
            A = samples['A'][i]
            
            gY_A = 1.0 / math.sqrt(max(f4 * C4_U1, 1e-30))
            g1_A = math.sqrt(K1) * gY_A
            g2_A = 1.0 / math.sqrt(max(f4 * C4_SU2, 1e-30))
            
            g1_mZ = self._rg_evolve_forward(g1_A, B1, A)
            g2_mZ = self._rg_evolve_forward(g2_A, B2, A)
            
            # sin²θ_W = g₁² / (g₁² + g₂²)
            sin2_tW = g1_mZ**2 / (g1_mZ**2 + g2_mZ**2)
            sin2_values[i] = sin2_tW
        
        return sin2_values
    
    def run_full_analysis(self):
        """Ejecutar análisis completo"""
        print("="*70)
        print("ANÁLISIS MONTE CARLO - PARÁMETROS CALIBRADOS")
        print("="*70)
        print(f"\nNúmero de muestras: {self.n_samples}")
        print(f"\nParámetros C4 calibrados desde observables:")
        print(f"  C4_U1  = {self.params['C4_U1'][0]:.4f} ± {self.params['C4_U1'][1]:.4f}")
        print(f"  C4_SU2 = {self.params['C4_SU2'][0]:.4f} ± {self.params['C4_SU2'][1]:.4f}")
        
        samples = self.sample_parameters()
        
        print("\nCalculando observables...")
        alpha_inv = self.compute_alpha_mZ(samples)
        sin2_tW = self.compute_sin2_thetaW(samples)
        
        print("\n" + "-"*70)
        print("RESULTADOS:")
        print("-"*70)
        
        # α⁻¹(m_Z)
        alpha_mean = np.mean(alpha_inv)
        alpha_std = np.std(alpha_inv, ddof=1)
        alpha_ci = np.percentile(alpha_inv, [2.5, 97.5])
        
        print(f"\nα⁻¹(m_Z):")
        print(f"  TSQVT:  {alpha_mean:.4f} ± {alpha_std:.4f}")
        print(f"  95% CI: [{alpha_ci[0]:.4f}, {alpha_ci[1]:.4f}]")
        print(f"  Exp:    {ALPHA_INV_EXP:.4f}")
        
        tension_alpha = abs(alpha_mean - ALPHA_INV_EXP) / alpha_std
        print(f"  Tensión: {tension_alpha:.2f}σ", end="")
        if tension_alpha < 2:
            print(" ✓")
        elif tension_alpha < 5:
            print(" ⚠")
        else:
            print(" ✗")
        
        # sin²θ_W
        sin2_mean = np.mean(sin2_tW)
        sin2_std = np.std(sin2_tW, ddof=1)
        sin2_ci = np.percentile(sin2_tW, [2.5, 97.5])
        
        print(f"\nsin²θ_W:")
        print(f"  TSQVT:  {sin2_mean:.6f} ± {sin2_std:.6f}")
        print(f"  95% CI: [{sin2_ci[0]:.6f}, {sin2_ci[1]:.6f}]")
        print(f"  Exp:    {SIN2_TW_EXP:.6f}")
        
        tension_sin2 = abs(sin2_mean - SIN2_TW_EXP) / sin2_std
        print(f"  Tensión: {tension_sin2:.2f}σ", end="")
        if tension_sin2 < 2:
            print(" ✓")
        elif tension_sin2 < 5:
            print(" ⚠")
        else:
            print(" ✗")
        
        self.results = {
            'alpha_inv': {'mean': alpha_mean, 'std': alpha_std, 'samples': alpha_inv},
            'sin2_tW': {'mean': sin2_mean, 'std': sin2_std, 'samples': sin2_tW}
        }
        
        return self.results
    
    def plot_distributions(self, save=True):
        """Graficar distribuciones"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # α⁻¹(m_Z)
        ax1 = axes[0]
        ax1.hist(self.results['alpha_inv']['samples'], bins=50, density=True, 
                alpha=0.7, color='steelblue', label='TSQVT')
        ax1.axvline(ALPHA_INV_EXP, color='red', linestyle='--', lw=2, label='Experimental')
        ax1.axvline(self.results['alpha_inv']['mean'], color='navy', linestyle='-', lw=2, label='TSQVT Mean')
        ax1.set_xlabel('α⁻¹(m_Z)', fontsize=12)
        ax1.set_ylabel('Densidad de probabilidad', fontsize=12)
        ax1.set_title('Constante de estructura fina', fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # sin²θ_W
        ax2 = axes[1]
        ax2.hist(self.results['sin2_tW']['samples'], bins=50, density=True, 
                alpha=0.7, color='steelblue', label='TSQVT')
        ax2.axvline(SIN2_TW_EXP, color='red', linestyle='--', lw=2, label='Experimental')
        ax2.axvline(self.results['sin2_tW']['mean'], color='navy', linestyle='-', lw=2, label='TSQVT Mean')
        ax2.set_xlabel('sin²θ_W', fontsize=12)
        ax2.set_ylabel('Densidad de probabilidad', fontsize=12)
        ax2.set_title('Ángulo de Weinberg', fontsize=14)
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('plots/tsqvt_calibrated_analysis.png', dpi=300, bbox_inches='tight')
            print("\nGráfico guardado: tsqvt_calibrated_analysis.png")
        
        return fig


# =============================================================================
# ANÁLISIS DE SENSIBILIDAD
# =============================================================================

class SensitivityAnalysis:
    """Análisis de sensibilidad con parámetros calibrados"""
    
    def __init__(self):
        self.base_values = {
            'C4_U1': 67.3216,
            'C4_SU2': 4.7883,
            'f_4': 0.25,
            'A': 2.0e16,
        }
    
    def _compute_alpha(self, params):
        gY_A = 1.0 / math.sqrt(max(params['f_4'] * params['C4_U1'], 1e-30))
        g1_A = math.sqrt(K1) * gY_A
        g2_A = 1.0 / math.sqrt(max(params['f_4'] * params['C4_SU2'], 1e-30))
        
        L = math.log(M_Z / params['A'])
        
        inv_g1_sq = 1.0/(g1_A**2) + (B1/(8*math.pi**2)) * L
        inv_g2_sq = 1.0/(g2_A**2) + (B2/(8*math.pi**2)) * L
        
        inv_g1_sq = max(inv_g1_sq, 1e-12)
        inv_g2_sq = max(inv_g2_sq, 1e-12)
        
        g1_mZ = 1.0 / math.sqrt(inv_g1_sq)
        g2_mZ = 1.0 / math.sqrt(inv_g2_sq)
        
        e_mZ = g1_mZ * g2_mZ / math.sqrt(g1_mZ**2 + g2_mZ**2)
        return 1.0 / (e_mZ**2 / (4.0 * math.pi))
    
    def compute_sensitivity(self, param_name, param_range):
        alpha_values = []
        for val in param_range:
            test = self.base_values.copy()
            test[param_name] = val
            alpha_values.append(self._compute_alpha(test))
        
        d_alpha = np.gradient(alpha_values, param_range)
        return alpha_values, d_alpha
    
    def rank_parameters(self):
        # Rangos ±10% alrededor de valores calibrados
        params_to_test = {
            'C4_U1': np.linspace(60.0, 75.0, 50),
            'C4_SU2': np.linspace(4.3, 5.3, 50),
            'f_4': np.linspace(0.22, 0.28, 50),
            'A': np.linspace(1.8e16, 2.2e16, 50),
        }
        
        sensitivities = {}
        for param, param_range in params_to_test.items():
            alpha_vals, d_alpha = self.compute_sensitivity(param, param_range)
            p_central = np.mean(param_range)
            alpha_central = np.mean(alpha_vals)
            S = np.mean(np.abs(d_alpha)) * (p_central / alpha_central)
            sensitivities[param] = S
        
        sorted_sens = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
        
        print("\n" + "="*70)
        print("RANKING DE SENSIBILIDAD")
        print("="*70)
        print(f"{'Parámetro':<12} {'Sensibilidad':<15} {'Interpretación'}")
        print("-"*70)
        
        for i, (param, S) in enumerate(sorted_sens):
            if S > 0.1:
                interp = "CRÍTICA - variaciones pequeñas afectan mucho"
            elif S > 0.01:
                interp = "ALTA - debe controlarse bien"
            elif S > 0.001:
                interp = "MEDIA - contribuye a incertidumbre"
            else:
                interp = "BAJA - poco impacto"
            print(f"{param:<12} {S:<15.6f} {interp}")
        
        return sorted_sens


# =============================================================================
# EXPERIMENTOS FÍSICOS (sin cambios respecto a tu versión)
# =============================================================================

class CollapseExperiment:
    """Simulación de experimento de colapso gravitacional"""
    
    def __init__(self):
        self.m = 1e-14  # kg
        self.hbar = 1.054571817e-34
        self.G = 6.674e-11
    
    def measure_collapse_time(self, n_events=1000):
        Delta_x = 100e-9
        Delta_E_grav = self.G * self.m**2 / Delta_x
        
        gamma_TSQVT = 1.0
        rho_particle = 0.95
        tau_TSQVT = self.hbar / (gamma_TSQVT * Delta_E_grav) * (1 - rho_particle)
        tau_DP = self.hbar / Delta_E_grav
        
        sigma_tau = abs(tau_TSQVT) * 0.15
        measured_times = np.random.normal(tau_TSQVT, sigma_tau, n_events)
        
        tau_mean = np.mean(measured_times)
        tau_std = np.std(measured_times) / math.sqrt(n_events)
        
        chi2_TSQVT = np.sum((measured_times - tau_TSQVT)**2) / (sigma_tau**2)
        chi2_DP = np.sum((measured_times - tau_DP)**2) / (sigma_tau**2)
        
        p_TSQVT = 1 - chi2.cdf(chi2_TSQVT, n_events-1)
        p_DP = 1 - chi2.cdf(chi2_DP, n_events-1)
        
        print(f"\n" + "="*70)
        print("EXPERIMENTO DE COLAPSO GRAVITACIONAL")
        print("="*70)
        print(f"\n  Configuración:")
        print(f"    Masa: {self.m:.0e} kg")
        print(f"    Separación: {Delta_x*1e9:.0f} nm")
        print(f"    n_eventos: {n_events}")
        
        print(f"\n  Resultados:")
        print(f"    τ_medido = {tau_mean*1000:.3f} ± {tau_std*1000:.3f} ms")
        print(f"    τ_TSQVT  = {tau_TSQVT*1000:.3f} ms")
        print(f"    τ_DP     = {tau_DP*1000:.3f} ms")
        
        print(f"\n  Estadística:")
        print(f"    χ²_TSQVT = {chi2_TSQVT:.1f} (p = {p_TSQVT:.4f})")
        print(f"    χ²_DP    = {chi2_DP:.1f} (p = {p_DP:.4f})")
        
        if p_TSQVT > 0.05 and p_DP < 0.05:
            print(f"\n  ✓ TSQVT FAVORECIDO sobre Diósi-Penrose")
        elif p_DP > 0.05 and p_TSQVT < 0.05:
            print(f"\n  ✗ Diósi-Penrose favorecido")
        else:
            print(f"\n  ~ Ambas teorías consistentes con datos")
        
        return tau_mean, tau_std, p_TSQVT, p_DP
    
    def detect_chirp_photons(self, n_collapses=10000, E_chirp_keV=1.2):
        rate_signal = 0.01
        rate_background = 0.001
        
        N_signal = n_collapses * rate_signal
        N_background = n_collapses * rate_background
        N_total = N_signal + N_background
        
        significance = N_signal / math.sqrt(N_total) if N_total > 0 else 0
        
        print(f"\n" + "="*70)
        print("DETECCIÓN DE FOTONES CHIRP")
        print("="*70)
        print(f"\n  E_γ = {E_chirp_keV:.2f} keV")
        print(f"  N_colapsos = {n_collapses}")
        print(f"  N_señal = {N_signal:.1f}")
        print(f"  N_fondo = {N_background:.1f}")
        print(f"  Significancia = {significance:.1f}σ")
        
        if significance > 5:
            print(f"\n  ✓ DETECCIÓN DECISIVA")
        elif significance > 3:
            print(f"\n  ⚠ EVIDENCIA MODERADA")
        else:
            print(f"\n  ✗ NO CONCLUSIVO")
        
        return significance


class BECExperiment:
    """Experimento BEC - velocidad del sonido"""
    
    def measure_sound_speed(self, rho_target=2/3, n_measurements=100):
        c_light = 299792458  # m/s exacto
        
        # Predicción TSQVT: c_s → c cuando ρ → ρ_c = 2/3
        c_s_theory = c_light  # En el punto crítico
        sigma_exp = 0.02 * c_s_theory  # 2% incertidumbre experimental
        
        c_s_measured = np.random.normal(c_s_theory, sigma_exp, n_measurements)
        c_s_mean = np.mean(c_s_measured)
        c_s_std = np.std(c_s_measured) / math.sqrt(n_measurements)
        
        # Comparar con c en el medio
        n_medium = 1.00028  # Aire
        c_medium = c_light / n_medium
        
        deviation = abs(c_s_mean - c_medium) / c_s_std
        
        print(f"\n" + "="*70)
        print("EXPERIMENTO BEC - VELOCIDAD DEL SONIDO")
        print("="*70)
        print(f"\n  ρ_target = {rho_target:.4f}")
        print(f"\n  Resultados:")
        print(f"    c_s (medido) = {c_s_mean/1000:.3f} ± {c_s_std/1000:.3f} km/s")
        print(f"    c (medio)    = {c_medium/1000:.3f} km/s")
        print(f"    c_s/c        = {c_s_mean/c_medium:.6f}")
        print(f"    Desviación   = {deviation:.2f}σ")
        
        if deviation < 2:
            print(f"\n  ✓ CONSISTENTE CON c_s = c")
        elif deviation < 5:
            print(f"\n  ⚠ MARGINALMENTE CONSISTENTE")
        else:
            print(f"\n  ✗ INCONSISTENTE")
        
        return c_s_mean, c_s_std, deviation


class MetamaterialTest:
    """Test de coeficiente de Poisson auxético"""
    
    def measure_poisson_ratio(self, material_type='origami'):
        # Medición simulada
        epsilon_axial = 0.01
        epsilon_trans = 0.0052  # Simulado cerca de predicción
        nu_measured = -epsilon_trans / epsilon_axial
        
        # Predicción TSQVT: ν = (1 - 2ρ)/(2 - 2ρ)
        rho_eff = 2/3
        nu_TSQVT = (1 - 2*rho_eff) / (2 - 2*rho_eff)  # = -0.5
        
        error_pct = abs(nu_measured - nu_TSQVT) / abs(nu_TSQVT) * 100
        
        print(f"\n" + "="*70)
        print("TEST METAMATERIAL AUXÉTICO")
        print("="*70)
        print(f"\n  Tipo: {material_type}")
        print(f"  ν_medido = {nu_measured:.3f}")
        print(f"  ν_TSQVT  = {nu_TSQVT:.3f}")
        print(f"  Error    = {error_pct:.1f}%")
        
        if error_pct < 5:
            print(f"\n  ✓ CONSISTENTE")
        elif error_pct < 20:
            print(f"\n  ⚠ MARGINALMENTE CONSISTENTE")
        else:
            print(f"\n  ✗ INCONSISTENTE")
        
        return nu_measured, nu_TSQVT


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("     TSQVT - PREDICCIONES NUMÉRICAS (CALIBRACIÓN INVERSA)")
    print("="*70)
    
    # 1. Análisis Monte Carlo
    analyzer = UncertaintyAnalysis(n_samples=10000)
    results = analyzer.run_full_analysis()
    analyzer.plot_distributions(save=True)
    
    # 2. Análisis de sensibilidad
    sensitivity = SensitivityAnalysis()
    sensitivity.rank_parameters()
    
    # 3. Experimento de colapso
    collapse = CollapseExperiment()
    collapse.measure_collapse_time(n_events=1000)
    collapse.detect_chirp_photons(n_collapses=10000)
    
    # 4. BEC
    bec = BECExperiment()
    bec.measure_sound_speed(rho_target=2/3)
    
    # 5. Metamaterial
    meta = MetamaterialTest()
    meta.measure_poisson_ratio()
    
    print("\n" + "="*70)
    print("ANÁLISIS COMPLETADO")
    print("="*70)
