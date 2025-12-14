#!/usr/bin/env python3
"""
TSQVT Predictions Pipeline
==========================

Run complete TSQVT predictions for Standard Model parameters.

Usage:
    python run_predictions.py [--output DIR] [--verbose]
    
Examples:
    python run_predictions.py
    python run_predictions.py --output results --verbose
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def main():
    """Main entry point for TSQVT predictions."""
    parser = argparse.ArgumentParser(
        description="Run TSQVT predictions for Standard Model parameters"
    )
    parser.add_argument(
        "--output", 
        default="results", 
        help="Output directory (default: results)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=2e16,
        help="UV cutoff scale in GeV (default: 2e16)"
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=2,
        choices=[1, 2],
        help="RG loop order (default: 2)"
    )
    args = parser.parse_args()
    
    # Banner
    print("=" * 70)
    print("TSQVT: Twistorial Spectral Quantum Vacuum Theory")
    print("=" * 70)
    
    # Import TSQVT modules
    try:
        import tsqvt
        print(f"Version: {tsqvt.__version__}")
    except ImportError as e:
        print(f"Error: Could not import tsqvt: {e}")
        print("Please install with: pip install -e .")
        return 1
    
    from tsqvt.core import SpectralManifold, CondensationField
    from tsqvt.gauge import compute_C4_coefficients, StandardModelGauge
    from tsqvt.rg import RGRunner
    from tsqvt.experimental import CollapsePredictor
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"UV cutoff: {args.cutoff:.2e} GeV")
    print(f"RG loops: {args.loops}")
    
    # Step 1: Create spectral manifold
    print("\n" + "-" * 50)
    print("Step 1: Spectral Manifold")
    print("-" * 50)
    
    manifold = SpectralManifold(
        volume=1.85e-61,
        twist_angle=0.198,
        hodge_numbers=(3, 243)
    )
    
    print(f"  Volume: {manifold.volume:.2e} m^4")
    print(f"  Twist angle: {manifold.twist_angle:.3f} rad")
    print(f"  Generations: {manifold.n_generations}")
    print(f"  Euler characteristic: {manifold.euler_characteristic}")
    
    # Step 2: Condensation field
    print("\n" + "-" * 50)
    print("Step 2: Condensation Field")
    print("-" * 50)
    
    field = CondensationField(vev=0.742)
    
    print(f"  VEV: ⟨ρ⟩ = {field.vev:.3f}")
    print(f"  Critical value: ρ_c = {field.critical_value:.3f}")
    print(f"  Effective cutoff: {field.effective_cutoff:.2e} GeV")
    print(f"  Is critical: {field.is_critical}")
    
    # Step 3: C_4 coefficients
    print("\n" + "-" * 50)
    print("Step 3: Spectral Action Coefficients")
    print("-" * 50)
    
    yukawa = {
        'e': 2.94e-6, 'mu': 6.09e-4, 'tau': 1.03e-2,
        'u': 1.24e-5, 'c': 7.30e-3, 't': 0.994,
        'd': 2.69e-5, 's': 5.35e-4, 'b': 2.40e-2,
    }
    majorana = {'nu1': 1e12, 'nu2': 1e13, 'nu3': 1e14}
    
    C4 = compute_C4_coefficients(yukawa, majorana)
    
    print("  C_4 coefficients:")
    for group, value in C4.items():
        print(f"    C_4^{{{group}}} = {value:.6f}")
    
    # Step 4: Gauge couplings
    print("\n" + "-" * 50)
    print("Step 4: Gauge Couplings")
    print("-" * 50)
    
    sm = StandardModelGauge(
        cutoff=args.cutoff,
        n_generations=manifold.n_generations
    )
    sm.compute()
    
    print("\n  At GUT scale:")
    for group, coupling in sm.couplings_gut.items():
        print(f"    g_{group}(M_GUT) = {coupling.value:.4f}")
    
    print("\n  At M_Z scale (after RG running):")
    for group, coupling in sm.couplings_mz.items():
        print(f"    g_{group}(M_Z) = {coupling.value:.4f}")
    
    # Step 5: Predictions summary
    print("\n" + "-" * 50)
    print("Step 5: Predictions vs Experiment")
    print("-" * 50)
    
    comparison = sm.compare_experiment()
    
    print("\n  {:<20} {:>12} {:>12} {:>10}".format(
        "Observable", "TSQVT", "Experiment", "Error %"
    ))
    print("  " + "-" * 54)
    
    for name, data in comparison.items():
        print("  {:<20} {:>12.4f} {:>12.4f} {:>10.2f}%".format(
            name,
            data['predicted'],
            data['experimental'],
            data['error_percent']
        ))
    
    # Step 6: Experimental predictions
    print("\n" + "-" * 50)
    print("Step 6: Experimental Predictions")
    print("-" * 50)
    
    predictor = CollapsePredictor(
        mass=1e-14,  # kg
        Delta_x=100e-9  # m
    )
    tau = predictor.compute_collapse_time()
    
    print(f"\n  Nanoparticle collapse experiment:")
    print(f"    Mass: {predictor.mass:.2e} kg")
    print(f"    Separation: {predictor.Delta_x:.2e} m")
    print(f"    Predicted collapse time: τ = {tau*1000:.1f} ms")
    
    # Sound speed prediction
    rho_crit = 2/3
    c_s_sq = field.sound_speed_squared(rho_crit)
    
    print(f"\n  BEC sound speed at ρ = {rho_crit:.3f}:")
    print(f"    c_s²/c² = {c_s_sq:.6f}")
    print(f"    c_s/c = {np.sqrt(c_s_sq):.6f}")
    
    # Poisson ratio
    nu = field.poisson_ratio(rho=0.99)
    
    print(f"\n  Auxetic metamaterial at ρ → 1:")
    print(f"    Poisson ratio ν = {nu:.3f}")
    
    # Save results
    print("\n" + "-" * 50)
    print("Saving Results")
    print("-" * 50)
    
    results = {
        'manifold': manifold.to_dict(),
        'condensation': field.to_dict(),
        'C4_coefficients': C4,
        'gauge_couplings': sm.summary(),
        'comparison': comparison,
        'experimental': {
            'collapse_time_ms': tau * 1000,
            'sound_speed_ratio': np.sqrt(c_s_sq),
            'poisson_ratio': nu,
        },
    }
    
    # Save JSON
    json_path = output_dir / 'tsqvt_predictions.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"  Saved: {json_path}")
    
    # Save summary text
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("TSQVT Predictions Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"α⁻¹(M_Z) = {sm.alpha_em_inverse():.2f} (exp: 137.04)\n")
        f.write(f"sin²θ_W = {sm.sin2_theta_w():.4f} (exp: 0.2312)\n")
        f.write(f"α_s(M_Z) = {sm.alpha_s():.4f} (exp: 0.1179)\n")
        f.write(f"Collapse time = {tau*1000:.1f} ms\n")
    print(f"  Saved: {summary_path}")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
