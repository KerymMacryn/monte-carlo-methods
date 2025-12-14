#!/usr/bin/env python3
"""
Generador de figuras TSQVT (versión corregida y más robusta).

Mejoras principales:
- Manejo robusto de matplotlib (fallback y mensaje claro).
- Logging en lugar de prints.
- Protección contra NaN/inf y divisiones por cero al invertir acoplamientos.
- Guardado de ficheros con manejo de errores.
- Parámetros CLI adicionales: --dpi, --npoints, --show, --quiet.
- Límites de ejes más flexibles (cálculo dinámico con fallback).
- Compatibilidad con Python 3.7+ (uso de typing.Optional, typing.List).
"""

from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

# Intentar importar matplotlib; si falla, informar y salir en main
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

# Configuración de logging
logger = logging.getLogger("generate_plots")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def setup_plot_style():
    """Configura estilo de figura con fallback si el estilo no está disponible."""
    if not HAS_MATPLOTLIB:
        return
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        try:
            plt.style.use("seaborn-whitegrid")
        except Exception:
            plt.style.use("classic")
    mpl.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.figsize": (8, 6),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def safe_inverse(arr: Iterable[float]) -> np.ndarray:
    """
    Devuelve 1/arr de forma segura:
    - Filtra valores no finitos.
    - Evita división por cero dejando NaN donde no es posible invertir.
    """
    a = np.asarray(list(arr), dtype=float)
    inv = np.full_like(a, np.nan, dtype=float)
    mask = np.isfinite(a) & (a != 0.0)
    inv[mask] = 1.0 / a[mask]
    return inv


def save_figure(fig, filepath: Path, dpi: Optional[int] = None):
    """Guardar figura con manejo de errores y logging."""
    try:
        fig.savefig(filepath, dpi=dpi)
        logger.info("    Saved: %s", filepath)
    except Exception as exc:
        logger.error("    No se pudo guardar %s: %s", filepath, exc)
        raise
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass


def compute_alpha_running(runner, alpha_init: float, mu_init: float, mu_points: np.ndarray, group: int) -> np.ndarray:
    """
    Ejecuta el running para una lista de escalas mu_points.
    Si runner.run_alpha falla en algún punto, se registra y se coloca NaN.
    """
    out = np.full_like(mu_points, np.nan, dtype=float)
    for i, m in enumerate(mu_points):
        try:
            out[i] = runner.run_alpha(alpha_init, mu_init, m, group)
        except Exception as exc:
            logger.debug("run_alpha fallo en group=%s, mu=%s: %s", group, m, exc)
            out[i] = np.nan
    return out


def set_dynamic_limits(ax, x: np.ndarray, ys: Iterable[np.ndarray], xpad: float = 0.05, ypad: float = 0.05):
    """Ajusta límites de ejes de forma dinámica con protección contra NaN/inf."""
    try:
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)
        yvals = np.hstack([np.asarray(y) for y in ys])
        ymin = np.nanmin(yvals)
        ymax = np.nanmax(yvals)
        if np.isfinite(xmin) and np.isfinite(xmax) and xmin > 0:
            ax.set_xscale("log")
            ax.set_xlim(max(1e-2, xmin * (1 - xpad)), xmax * (1 + xpad))
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin < ymax:
            low = ymin * (1 - ypad) if ymin > 0 else ymin * (1 + ypad)
            ax.set_ylim(low, ymax * (1 + ypad))
    except Exception:
        logger.debug("No se pudieron establecer límites dinámicos; se mantienen los por defecto.")


def plot_rg_running(output_dir: Path, fmt: str = "png", dpi: int = 300, npoints: int = 200):
    """Genera la figura del running de acoplamientos de gauge (1-loop estable)."""
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib no disponible; omitiendo plot_rg_running")
        return

    try:
        from tsqvt.rg import RGRunner
    except Exception as exc:
        logger.error("No se pudo importar RGRunner: %s", exc)
        return

    logger.info(" Generating RG running plot (1-loop)...")

    runner_plot = RGRunner(loops=1)  # <- clave: estable
    mu = np.logspace(2, 16, max(50, int(npoints)))
    alpha_gut = 1.0 / 24.0
    M_GUT = 2e16

    # Vectorizado con try/except por si run_alpha falla en algún punto
    alpha1 = compute_alpha_running(runner_plot, alpha_gut, M_GUT, mu, 1)
    alpha2 = compute_alpha_running(runner_plot, alpha_gut, M_GUT, mu, 2)
    alpha3 = compute_alpha_running(runner_plot, alpha_gut, M_GUT, mu, 3)

    mask = (
        np.isfinite(alpha1) & np.isfinite(alpha2) & np.isfinite(alpha3)
        & (alpha1 > 0) & (alpha2 > 0) & (alpha3 > 0)
    )

    a1i = 1.0 / alpha1[mask]
    a2i = 1.0 / alpha2[mask]
    a3i = 1.0 / alpha3[mask]
    mu_m = mu[mask]

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(mu_m, a1i, "b-", linewidth=2, label=r"$\alpha_1^{-1}$ (U(1))")
    ax.plot(mu_m, a2i, "g-", linewidth=2, label=r"$\alpha_2^{-1}$ (SU(2))")
    ax.plot(mu_m, a3i, "r-", linewidth=2, label=r"$\alpha_3^{-1}$ (SU(3))")

    ax.axvline(91.1876, color="gray", linestyle="--", alpha=0.5)
    ax.text(91.1876 * 1.25, np.nanmin([a1i.min(), a2i.min(), a3i.min()]) + 1.0, r"$M_Z$", fontsize=12)

    ax.axvline(M_GUT, color="gray", linestyle="--", alpha=0.5)
    ax.text(M_GUT * 0.35, np.nanmin([a1i.min(), a2i.min(), a3i.min()]) + 1.0, r"$M_{GUT}$", fontsize=12)

    ax.set_xscale("log")
    ax.set_xlabel(r"Energy scale $\mu$ [GeV]")
    ax.set_ylabel(r"Inverse coupling $\alpha^{-1}$")
    ax.set_title("TSQVT: Gauge Coupling Unification (1-loop)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    filepath = output_dir / ("rg_running." + fmt)
    save_figure(fig, filepath, dpi=dpi)



def plot_predictions_comparison(output_dir: Path, fmt: str = "png", dpi: int = 300):
    """Genera la comparación de predicciones con experimento."""
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib no disponible; omitiendo plot_predictions_comparison")
        return

    try:
        from tsqvt.gauge import StandardModelGauge
    except Exception as exc:
        logger.error("No se pudo importar StandardModelGauge: %s", exc)
        return

    logger.info("  Generating predictions comparison plot...")

    try:
        sm = StandardModelGauge(cutoff=2e16)
        sm.compute()
        comparison = sm.compare_experiment()
    except Exception as exc:
        logger.error("Error al obtener comparaciones desde StandardModelGauge: %s", exc)
        return

    observables = list(comparison.keys())
    predicted = np.array([comparison[k]["predicted"] for k in observables], dtype=float)
    experimental = np.array([comparison[k]["experimental"] for k in observables], dtype=float)
    errors = np.array([comparison[k].get("error_percent", 0.0) for k in observables], dtype=float)

    labels = {
        "alpha_em_inv": r"$\alpha^{-1}$",
        "sin2_theta_w": r"$\sin^2\theta_W$",
        "alpha_s": r"$\alpha_s$",
        "mw_mz_ratio": r"$M_W/M_Z$",
    }
    x_labels = [labels.get(k, k) for k in observables]

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(np.isfinite(experimental) & (experimental != 0.0), predicted / experimental, np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(observables))
    width = 0.6
    colors = ["#2ecc71" if e < 5 else "#f39c12" if e < 10 else "#e74c3c" for e in errors]

    bars = ax1.bar(x, ratios, width, color=colors, edgecolor="black", linewidth=1)
    ax1.axhline(1.0, color="black", linestyle="-", linewidth=2)
    ax1.axhline(0.95, color="gray", linestyle="--", alpha=0.5)
    ax1.axhline(1.05, color="gray", linestyle="--", alpha=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylabel("Prediction / Experiment")
    ax1.set_title("TSQVT Predictions vs Experiment")
    ax1.set_ylim(0.9, 1.1)
    ax1.grid(True, alpha=0.3, axis="y")

    for i, (bar, error) in enumerate(zip(bars, errors)):
        height = bar.get_height()
        if np.isfinite(height):
            ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, "{:.1f}%".format(error), ha="center", va="bottom", fontsize=10)

    ax2.barh(x, errors, color=colors, edgecolor="black", linewidth=1)
    ax2.axvline(5, color="green", linestyle="--", alpha=0.7, label="5% threshold")
    ax2.axvline(10, color="orange", linestyle="--", alpha=0.7, label="10% threshold")

    ax2.set_yticks(x)
    ax2.set_yticklabels(x_labels)
    ax2.set_xlabel("Error (%)")
    ax2.set_title("Prediction Errors")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    filepath = output_dir / ("predictions_comparison." + fmt)
    save_figure(fig, filepath, dpi=dpi)


def plot_collapse_time(output_dir: Path, fmt: str = "png", dpi: int = 300, npoints: int = 50):
    """Genera la figura de tiempos de colapso comparando modelos."""
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib no disponible; omitiendo plot_collapse_time")
        return

    try:
        from tsqvt.experimental import CollapsePredictor
    except Exception as exc:
        logger.error("No se pudo importar CollapsePredictor: %s", exc)
        return

    logger.info("  Generating collapse time plot...")

    masses = np.logspace(-18, -10, npoints)  # kg
    Delta_x = 100e-9  # 100 nm

    tau_tsqvt = []
    for m in masses:
        try:
            predictor = CollapsePredictor(mass=m, Delta_x=Delta_x)
            tau_tsqvt.append(predictor.compute_collapse_time())
        except Exception as exc:
            logger.debug("CollapsePredictor fallo para m=%s: %s", m, exc)
            tau_tsqvt.append(np.nan)
    tau_tsqvt = np.array(tau_tsqvt, dtype=float)

    G = 6.674e-11
    hbar = 1.055e-34
    with np.errstate(divide="ignore", invalid="ignore"):
        tau_dp = np.where(masses != 0.0, hbar * Delta_x / (G * masses**2), np.nan)

    tau_csl = np.ones_like(masses) * 1e6

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.loglog(masses * 1e15, tau_tsqvt, "b-", linewidth=2.5, label="TSQVT")
    ax.loglog(masses * 1e15, tau_dp, "r--", linewidth=2, label="Diósi-Penrose")
    ax.loglog(masses * 1e15, tau_csl, "g:", linewidth=2, label="CSL")

    m_exp = 1e-14
    ax.axvline(m_exp * 1e15, color="gray", linestyle="--", alpha=0.5)
    ax.text(m_exp * 1e15 * 1.3, 1e-2, "{:.0f} fg".format(m_exp * 1e15), fontsize=11)

    ax.axhspan(0.01, 1, alpha=0.2, color="yellow", label="Accessible window")

    ax.set_xlabel("Mass [fg]")
    ax.set_ylabel("Collapse time [s]")
    ax.set_title("Objective Collapse Time (Δx = {:.0f} nm)".format(Delta_x * 1e9))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    ax.set_xlim(1e-3, 1e5)
    ax.set_ylim(1e-6, 1e8)

    filepath = output_dir / ("collapse_time." + fmt)
    save_figure(fig, filepath, dpi=dpi)


def plot_condensation_properties(output_dir: Path, fmt: str = "png", dpi: int = 300):
    """Genera la figura de propiedades de condensación."""
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib no disponible; omitiendo plot_condensation_properties")
        return

    try:
        from tsqvt.core import CondensationField
    except Exception as exc:
        logger.error("No se pudo importar CondensationField: %s", exc)
        return

    logger.info("  Generating condensation properties plot...")

    try:
        field = CondensationField()
    except Exception as exc:
        logger.error("No se pudo instanciar CondensationField: %s", exc)
        return

    rho = np.linspace(0.01, 0.99, 100)

    c_s_sq = np.array([field.sound_speed_squared(r) for r in rho], dtype=float)
    nu = np.array([field.poisson_ratio(r) for r in rho], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(rho, np.sqrt(np.clip(c_s_sq, 0.0, None)), "b-", linewidth=2)
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.axvline(2 / 3, color="red", linestyle="--", alpha=0.7, label=r"$\rho = 2/3$")

    ax1.set_xlabel(r"Condensation parameter $\rho$")
    ax1.set_ylabel(r"Sound speed ratio $c_s/c$")
    ax1.set_title("BEC Sound Speed Prediction")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 2)

    ax2.plot(rho, nu, "g-", linewidth=2)
    ax2.axhline(0, color="gray", linestyle="-", alpha=0.5)
    ax2.axhline(-0.5, color="red", linestyle="--", alpha=0.7, label=r"$\nu = -1/2$ (auxetic)")

    ax2.scatter([0.9, 0.95], [-0.45, -0.52], color="orange", s=100, marker="s", zorder=5, label="Metamaterials")

    ax2.set_xlabel(r"Condensation parameter $\rho$")
    ax2.set_ylabel(r"Poisson ratio $\nu$")
    ax2.set_title("Auxetic Behavior Prediction")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.6, 0.6)

    plt.tight_layout()

    filepath = output_dir / ("condensation_properties." + fmt)
    save_figure(fig, filepath, dpi=dpi)


def main(argv: Optional[List[str]] = None) -> int:
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(description="Generate TSQVT visualization plots")
    parser.add_argument("--output", default="plots", help="Output directory (default: plots)")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Output format")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures")
    parser.add_argument("--npoints", type=int, default=100, help="Number of points for RG running")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging verbosity")
    parser.add_argument("--show", action="store_true", help="Show figures interactively (useful en desarrollo)")
    args = parser.parse_args(argv)

    if args.quiet:
        logger.setLevel(logging.WARNING)

    if not HAS_MATPLOTLIB:
        logger.error("matplotlib es requerido para generar figuras. Instala con: pip install matplotlib")
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir.absolute())
    logger.info("Format: %s", args.format)

    setup_plot_style()

    logger.info("Generating plots...")

    try:
        plot_rg_running(output_dir, fmt=args.format, dpi=args.dpi, npoints=args.npoints)
        plot_predictions_comparison(output_dir, fmt=args.format, dpi=args.dpi)
        plot_collapse_time(output_dir, fmt=args.format, dpi=args.dpi, npoints=min(200, max(10, args.npoints // 2)))
        plot_condensation_properties(output_dir, fmt=args.format, dpi=args.dpi)
    except Exception as exc:
        logger.error("Error durante la generación de figuras: %s", exc)
        return 2

    if args.show and HAS_MATPLOTLIB:
        try:
            plt.show()
        except Exception:
            logger.debug("No se pudo mostrar las figuras interactivamente.")

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
