"""
Plotting Utilities
==================

Visualization functions for TSQVT results.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_rg_running(
    couplings_high: Dict[str, float],
    mu_high: float = 2e16,
    mu_low: float = 91.2,
    n_points: int = 100,
    save_path: Optional[str] = None
):
    """
    Plot RG evolution of gauge couplings.
    
    Parameters
    ----------
    couplings_high : dict
        Couplings at high scale {'alpha1': ..., 'alpha2': ..., 'alpha3': ...}.
    mu_high : float
        High scale in GeV.
    mu_low : float
        Low scale in GeV.
    n_points : int
        Number of points for plotting.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if matplotlib available.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available for plotting")
        return None
    
    from tsqvt.rg import RGRunner
    
    runner = RGRunner(loops=2)
    
    # Generate scale points (log-spaced)
    log_mu = np.linspace(np.log10(mu_low), np.log10(mu_high), n_points)
    mu_vals = 10**log_mu
    
    # Run couplings
    alpha_1 = []
    alpha_2 = []
    alpha_3 = []
    
    for mu in mu_vals:
        a1 = runner.run_alpha(couplings_high.get('alpha1', 1/60), mu_high, mu, 1)
        a2 = runner.run_alpha(couplings_high.get('alpha2', 1/30), mu_high, mu, 2)
        a3 = runner.run_alpha(couplings_high.get('alpha3', 1/10), mu_high, mu, 3)
        
        alpha_1.append(1/a1 if a1 > 0 else np.nan)
        alpha_2.append(1/a2 if a2 > 0 else np.nan)
        alpha_3.append(1/a3 if a3 > 0 else np.nan)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(log_mu, alpha_1, 'b-', label=r'$\alpha_1^{-1}$ (U(1))', linewidth=2)
    ax.plot(log_mu, alpha_2, 'g-', label=r'$\alpha_2^{-1}$ (SU(2))', linewidth=2)
    ax.plot(log_mu, alpha_3, 'r-', label=r'$\alpha_3^{-1}$ (SU(3))', linewidth=2)
    
    ax.set_xlabel(r'$\log_{10}(\mu/\mathrm{GeV})$', fontsize=12)
    ax.set_ylabel(r'$\alpha^{-1}$', fontsize=12)
    ax.set_title('RG Running of Gauge Couplings (TSQVT)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Mark M_Z and M_GUT
    ax.axvline(np.log10(91.2), color='gray', linestyle='--', alpha=0.5, label='$M_Z$')
    ax.axvline(np.log10(2e16), color='gray', linestyle=':', alpha=0.5, label='$M_{GUT}$')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_comparison(
    predictions: Dict[str, float],
    experimental: Dict[str, float],
    uncertainties: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None
):
    """
    Compare TSQVT predictions with experimental values.
    
    Parameters
    ----------
    predictions : dict
        TSQVT predicted values.
    experimental : dict
        Experimental values.
    uncertainties : dict, optional
        Prediction uncertainties.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available for plotting")
        return None
    
    # Get common keys
    keys = list(set(predictions.keys()) & set(experimental.keys()))
    keys = sorted(keys)
    
    if not keys:
        print("No common keys to compare")
        return None
    
    # Compute ratios
    ratios = []
    errors = []
    labels = []
    
    for key in keys:
        pred = predictions[key]
        exp = experimental[key]
        
        if exp != 0:
            ratio = pred / exp
            ratios.append(ratio)
            labels.append(key)
            
            if uncertainties and key in uncertainties:
                errors.append(uncertainties[key] / exp)
            else:
                errors.append(0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(labels))
    width = 0.6
    
    bars = ax.bar(x, ratios, width, yerr=errors, capsize=5, 
                  color='steelblue', edgecolor='navy', alpha=0.8)
    
    # Perfect agreement line
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Perfect agreement')
    
    # 5% bands
    ax.axhspan(0.95, 1.05, color='green', alpha=0.1, label='±5%')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Predicted / Experimental', fontsize=12)
    ax.set_title('TSQVT Predictions vs Experiment', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Set y-axis limits
    ax.set_ylim(0.8, 1.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_collapse_time(
    mass_range: Tuple[float, float] = (1e-16, 1e-12),
    separation: float = 100e-9,
    rho: float = 0.742,
    save_path: Optional[str] = None
):
    """
    Plot collapse time as function of mass.
    
    Parameters
    ----------
    mass_range : tuple
        (min_mass, max_mass) in kg.
    separation : float
        Superposition separation in m.
    rho : float
        Condensation parameter.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available for plotting")
        return None
    
    from tsqvt.experimental import CollapsePredictor
    
    # Generate mass points
    masses = np.logspace(np.log10(mass_range[0]), np.log10(mass_range[1]), 50)
    
    # Compute collapse times
    tau_tsqvt = []
    tau_dp = []
    
    for m in masses:
        pred = CollapsePredictor(mass=m, Delta_x=separation, rho=rho)
        tau_tsqvt.append(pred.tau_tsqvt)
        tau_dp.append(pred.tau_dp)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(masses, tau_tsqvt, 'b-', label='TSQVT', linewidth=2)
    ax.loglog(masses, tau_dp, 'r--', label='Diósi-Penrose', linewidth=2)
    ax.axhline(1e6, color='gray', linestyle=':', label='CSL (~10⁶ s)')
    
    ax.set_xlabel('Mass [kg]', fontsize=12)
    ax.set_ylabel('Collapse time [s]', fontsize=12)
    ax.set_title(f'Objective Collapse Time (Δx = {separation*1e9:.0f} nm)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    
    # Mark experimental regime
    ax.axvspan(1e-15, 1e-13, color='green', alpha=0.1, label='Experimental regime')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_sound_speed(
    rho_range: Tuple[float, float] = (0.1, 0.95),
    save_path: Optional[str] = None
):
    """
    Plot BEC sound speed as function of condensation.
    
    Parameters
    ----------
    rho_range : tuple
        Range of ρ values.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available for plotting")
        return None
    
    from tsqvt.experimental import BECSoundSpeed
    
    bec = BECSoundSpeed()
    profile = bec.profile(rho_range=rho_range)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(profile['rho'], profile['c_s_ratio'], 'b-', linewidth=2)
    ax.axhline(1.0, color='red', linestyle='--', label='c_s = c')
    ax.axvline(2/3, color='green', linestyle=':', label='ρ = 2/3 (critical)')
    
    ax.set_xlabel(r'Condensation parameter $\rho$', fontsize=12)
    ax.set_ylabel(r'$c_s / c$', fontsize=12)
    ax.set_title('Sound Speed in Spectral Medium', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
