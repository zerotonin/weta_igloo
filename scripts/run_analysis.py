#!/usr/bin/env python3
"""
 ╔═══════════════════════════════════════════════════════════════════════╗
 ║  ░█▀▄░█░█░█▀█░░░█▀█░█▀█░█▀█░█░░░█░█░█▀▀░▀█▀░█▀▀░░░░░░░░░░░░░░         ║
 ║  ░█▀▄░█░█░█░█░░░█▀█░█░█░█▀█░█░░░░█░░▀▀█░░█░░▀▀█░░░░░░░░░░░░░░         ║
 ║  ░▀░▀░▀▀▀░▀░▀░░░▀░▀░▀░▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀▀▀░░░░░░░░░░░░░░         ║
 ║                                                                       ║
 ║   Full analysis pipeline for wētā burrow thermal model    v0.1.0      ║
 ║   ── load. model. fit. export. ──                                     ║
 ╚═══════════════════════════════════════════════════════════════════════╝

Runs the complete analysis:
  1. Load all experimental data
  2. Validate model on incubator passive control
  3. Fit null + full thermal models per rock
  4. Compute species-level crossover temperature
  5. Run shell thickness sensitivity analysis
  6. Generate all figures (SVG + PNG) and CSV tables

Usage::

    python -m scripts.run_analysis
    python -m scripts.run_analysis --data-dir /path/to/data --output-dir ./results
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── ensure package is importable when running from repo root ─────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from igloo_weta import __version__
from igloo_weta.ingest import load_all, summarise_species
from igloo_weta.fitting import fit_all_rocks, fit_incubator, compute_species_crossover
from igloo_weta.sensitivity import compute_species_rmr, sweep_all_rocks
from igloo_weta.viz import (
    export_results_csv,
    plot_crossover,
    plot_incubator,
    plot_per_rock_fits,
    plot_residuals,
    plot_species_sensitivity,
)


# ┌─────────────────────────────────────────────────────────────────────┐
# │  BANNER                                « first impressions »        │
# └─────────────────────────────────────────────────────────────────────┘
BANNER = r"""
 ┌──────────────────────────────────────────────┐
 │  IGLOO WĒTĀ  v{ver}                          │
 │  Thermal model for burrow heat exchange      │
 │  « model. fit. predict. science. »           │
 └──────────────────────────────────────────────┘
""".format(ver=__version__)


def main() -> None:
    """Parse CLI arguments and execute the full analysis pipeline."""
    parser = argparse.ArgumentParser(
        prog="run_analysis",
        description="Run the full wētā burrow thermal model pipeline.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to data directory (default: ./data/).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Path to output directory (default: ./output/).",
    )
    parser.add_argument(
        "--shell-cm", type=float, default=1.0,
        help="Default stone shell thickness in cm (default: 1.0).",
    )
    args = parser.parse_args()

    print(BANNER)
    t0 = time.time()

    # ── 1. LOAD DATA ─────────────────────────────────────────────────
    print("[1/6] Loading data...")
    ds = load_all(args.data_dir)
    species_stats = summarise_species(ds.weta_morph)
    print(f"       {len(ds.hourly_24h['rock'].unique())} rocks, "
          f"{len(species_stats)} wētā species loaded.")

    # ── 2. INCUBATOR VALIDATION ──────────────────────────────────────
    print("[2/6] Fitting incubator passive control...")
    inc = fit_incubator(ds.incubator)
    print(f"       k = {inc.k_wood:.4f} /h (τ = {1/inc.k_wood:.2f} h), "
          f"R² = {inc.r2:.4f}")

    # ── 3. FIT ALL ROCKS ─────────────────────────────────────────────
    shell_m = args.shell_cm / 100.0
    print(f"[3/6] Fitting thermal models (shell = {args.shell_cm} cm)...")
    results = fit_all_rocks(ds.hourly_24h, ds.rock_phys, shell_m=shell_m)
    n_sig = sum(1 for r in results if r.p_value < 0.05)
    print(f"       {len(results)} rocks fitted, {n_sig} significant (p<0.05).")

    # ── 4. SPECIES CROSSOVER ─────────────────────────────────────────
    print("[4/6] Computing species-level crossover...")
    sp_cross = compute_species_crossover(results)
    print(f"       Raw: {sp_cross['T_cross_raw']:.1f}°C, "
          f"Lag-corrected: {sp_cross['T_cross_corr']:.1f}°C")

    # ── 5. SENSITIVITY ───────────────────────────────────────────────
    print("[5/6] Running sensitivity analysis...")
    sweep = sweep_all_rocks(results)
    sp_rmr = compute_species_rmr(species_stats)
    print(f"       {len(sweep)} rocks × {len(next(iter(sweep.values())))} "
          f"thicknesses, {len(sp_rmr)} species RMR computed.")

    # ── 6. FIGURES + EXPORT ──────────────────────────────────────────
    print(f"[6/6] Generating figures → {args.output_dir}/")
    od = args.output_dir

    plot_incubator(inc, od)
    plot_per_rock_fits(results, od)
    plot_residuals(results, od)
    plot_crossover(results, sp_cross, od)
    plot_species_sensitivity(results, sweep, sp_rmr, od)
    csv_path = export_results_csv(results, od)

    dt = time.time() - t0
    print(f"\n       Done in {dt:.1f}s.  Results in {od}/")
    print(f"       CSV table: {csv_path}")

    # ── summary table to stdout ──────────────────────────────────────
    print()
    print("=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    hdr = (f"{'Rock':>5} {'τ(h)':>6} {'R²null':>7} {'R²full':>7} "
           f"{'p':>8} {'sig':>4} {'ΔT':>7} {'Q(mW)':>8} "
           f"{'Tcross':>7}")
    print(hdr)
    print("-" * 100)
    for r in sorted(results, key=lambda x: x.mean_dT_obs, reverse=True):
        sig = ("***" if r.p_value < 0.001 else "**" if r.p_value < 0.01
               else "*" if r.p_value < 0.05 else "ns")
        tau = f"{r.tau_hours:.2f}" if r.tau_hours < 10 else "→0"
        Q = (f"{r.Q_weta_W * 1000:.0f}" if r.Q_weta_W is not None
             and r.tau_hours < 10 else "—")
        tc = (f"{r.T_cross_corr:.1f}" if abs(r.T_cross_corr) < 30
              else "extrap.")
        print(f"{r.rock:>5d} {tau:>6} {r.r2_null:>7.3f} {r.r2_full:>7.3f} "
              f"{r.p_value:>8.4f} {sig:>4} {r.mean_residual:>+7.3f} "
              f"{Q:>8} {tc:>7}")
    print("-" * 100)
    print(f"Species crossover (lag-corrected): {sp_cross['T_cross_corr']:.1f}°C")


if __name__ == "__main__":
    main()
