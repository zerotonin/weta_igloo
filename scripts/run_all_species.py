#!/usr/bin/env python3
"""
 ╔═══════════════════════════════════════════════════════════════════════╗
 ║  ░█▀▄░█░█░█▀█░░░█▀█░█░░░█░░░░░█▀▀░█▀█░█▀▀░█▀▀░▀█▀░█▀▀░█▀▀░░░  ║
 ║  ░█▀▄░█░█░█░█░░░█▀█░█░░░█░░░░░▀▀█░█▀▀░█▀▀░█░░░░█░░█▀▀░▀▀█░░░  ║
 ║  ░▀░▀░▀▀▀░▀░▀░░░▀░▀░▀▀▀░▀▀▀░░░▀▀▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀░░░  ║
 ║                                                                       ║
 ║   Multi-species wētā thermal analysis                     v0.2.0     ║
 ║   ── H. maori · H. crassidens · H. thoracica ──                     ║
 ╚═══════════════════════════════════════════════════════════════════════╝

Runs the thermal model pipeline for all three wētā species, adapting
to stone (maori) and wood (crassidens, thoracica) gallery types.

Data layout expected::

    data/
    ├── h_maori/           (stone burrows, Rock and Pillars)
    │   ├── 24h_hourly_averages.csv
    │   ├── Rock_data.xlsx
    │   └── ...
    ├── h_crassidens/      (wood galleries)
    │   ├── 24h_hourly_averages.csv
    │   └── ...
    ├── h_thoracica/       (wood galleries)
    │   ├── 24h_hourly_averages.csv
    │   └── ...
    └── Weta_thermoregulation_datasheet.xlsx

Usage::

    python scripts/run_all_species.py
    python scripts/run_all_species.py --data-dir ./data --output-dir ./output
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import f as f_dist

# ── ensure package is importable ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from igloo_weta.constants import (
    STONE, WOOD, AIR, SPECIES, ALLOMETRY, SIM, VIZ,
    H_MAORI, H_THORACICA, H_CRASSIDENS,
)
from igloo_weta.physics import (
    simulate_24h_steady_state, compute_phase_lag, compute_amplitude_ratio,
)

matplotlib.rcParams["svg.fonttype"] = "none"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.linewidth": 0.8})


# ┌─────────────────────────────────────────────────────────────────────┐
# │  COLUMN ADAPTERS                    « speaking every dialect »      │
# └─────────────────────────────────────────────────────────────────────┘

# H. maori uses: Hour, inside_mean, outside_mean, inside_sem, ...
# H. crassidens/thoracica use: hour, temp_in_mean, temp_out_mean, temp_in_sem, ...

MAORI_COLMAP = {}  # already in correct format

TREE_COLMAP = {
    "hour": "Hour",
    "temp_in_mean": "inside_mean",
    "temp_in_median": "inside_median",
    "temp_in_sem": "inside_sem",
    "temp_in_mean_ci_low": "inside_ci_lower_mean",
    "temp_in_mean_ci_up": "inside_ci_upper_mean",
    "temp_out_mean": "outside_mean",
    "temp_out_median": "outside_median",
    "temp_out_sem": "outside_sem",
    "temp_out_mean_ci_low": "outside_ci_lower_mean",
    "temp_out_mean_ci_up": "outside_ci_upper_mean",
    "temp_diff_mean": "diff_mean",
    "temp_diff_sem": "diff_sem",
}


def load_24h(species_dir: Path) -> pd.DataFrame:
    """Load 24-h hourly averages, adapting column names as needed.

    Args:
        species_dir: Path to species data directory.

    Returns:
        DataFrame with standardised column names.
    """
    p = species_dir / "24h_hourly_averages.csv"
    df = pd.read_csv(p)
    if "temp_in_mean" in df.columns:
        df = df.rename(columns=TREE_COLMAP)
    return df


# ┌─────────────────────────────────────────────────────────────────────┐
# │  GALLERY PHYSICS                  « stone vs wood »                │
# └─────────────────────────────────────────────────────────────────────┘

def compute_gallery_C_eff(sp_info, wall_thickness_mm=None):
    """Compute effective heat capacity for a species' gallery.

    Args:
        sp_info: :class:`~constants.SpeciesInfo` instance.
        wall_thickness_mm: Override wall thickness (mm).

    Returns:
        Tuple of (C_eff in J/K, material name string).
    """
    V_cav = sp_info.gallery_volume_cm3 * 1e-6
    SA = sp_info.gallery_SA_cm2 * 1e-4
    wall_m = (wall_thickness_mm or sp_info.gallery_wall_mm) / 1000.0

    if sp_info.wall_material == "stone":
        mat = STONE
    else:
        mat = WOOD

    C_wall = SA * wall_m * mat.rho * mat.c
    C_air = AIR.rho * V_cav * AIR.c
    return C_wall + C_air, mat.name


# ┌─────────────────────────────────────────────────────────────────────┐
# │  SINGLE-SPECIES PIPELINE            « the workhorse »              │
# └─────────────────────────────────────────────────────────────────────┘

def run_species(sp_info, species_dir: Path, output_dir: Path,
                rock_phys_df=None):
    """Run the full thermal model pipeline for one species.

    Args:
        sp_info:       :class:`~constants.SpeciesInfo`.
        species_dir:   Path to species data directory.
        output_dir:    Output directory for figures/CSV.
        rock_phys_df:  Optional photogrammetric data (for H. maori).

    Returns:
        List of result dicts.
    """
    tag = sp_info.name.replace(". ", "_").lower()
    color = VIZ.species_colors.get(sp_info.name, "#888")
    os.makedirs(output_dir, exist_ok=True)

    # ── load data ────────────────────────────────────────────────────
    h24 = load_24h(species_dir)
    C_eff, mat_name = compute_gallery_C_eff(sp_info)

    rmr_10 = ALLOMETRY.a * sp_info.mass_g ** ALLOMETRY.b / ALLOMETRY.Q10 ** 1.5
    rmr_field = ALLOMETRY.a * sp_info.mass_g ** ALLOMETRY.b / ALLOMETRY.Q10 ** 1.2

    print(f"\n  {sp_info.name} ({sp_info.common_name})")
    print(f"  Gallery: {sp_info.wall_material}, V={sp_info.gallery_volume_cm3} cm³, "
          f"SA={sp_info.gallery_SA_cm2} cm², wall={sp_info.gallery_wall_mm} mm")
    print(f"  C_eff = {C_eff:.1f} J/K, material: {mat_name}")
    print(f"  Body mass: {sp_info.mass_g} g, RMR@10°C={rmr_10:.1f} mW")

    # ── per-rock fitting (use photogrammetry for maori if available) ─
    results = []
    for rid in sorted(h24["rock"].unique()):
        sub = h24[h24["rock"] == rid].sort_values("Hour")
        Ti = sub["inside_mean"].values
        To = sub["outside_mean"].values
        ci_lo = sub["inside_ci_lower_mean"].values
        ci_hi = sub["inside_ci_upper_mean"].values

        # For maori with photogrammetry: use per-rock C_eff
        if rock_phys_df is not None and sp_info.wall_material == "stone":
            rrow = rock_phys_df[rock_phys_df["Rock number"] == rid]
            if (len(rrow) > 0
                    and not pd.isna(rrow.iloc[0]["Total Volume (cm3)"])
                    and not pd.isna(rrow.iloc[0]["Total Surface area (cm2)"])):
                sa_m2 = rrow.iloc[0]["Total Surface area (cm2)"] * 1e-4
                wall_m = sp_info.gallery_wall_mm / 1000.0
                C_rock = sa_m2 * wall_m * STONE.rho * STONE.c
                V_cav = rrow.iloc[0]["Total Volume (cm3)"] * 1e-6
                C_rock += AIR.rho * V_cav * AIR.c
            else:
                C_rock = C_eff
        else:
            C_rock = C_eff

        ss_tot = float(np.sum((Ti - np.mean(Ti)) ** 2))
        if ss_tot == 0:
            continue

        # null model
        def cn(k):
            if k <= 0: return 1e10
            p = simulate_24h_steady_state(k, To)
            s = float(np.nansum((p - Ti) ** 2))
            return s if np.isfinite(s) else 1e10

        rn = minimize_scalar(cn, bounds=SIM.k_fit_bounds, method="bounded")
        Tn = simulate_24h_steady_state(rn.x, To)
        ssn = float(np.sum((Ti - Tn) ** 2))

        # full model
        def cf(params):
            k, q = params
            if k <= 0: return 1e10
            p = simulate_24h_steady_state(k, To, Q_norm=q)
            s = float(np.nansum((p - Ti) ** 2))
            return s if np.isfinite(s) else 1e10

        rf = minimize(cf, [rn.x, 0], method="Nelder-Mead",
                      options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 50000})
        kf, qf = rf.x
        if kf > SIM.k_fit_bounds[1] or kf <= 0:
            kf = rn.x
            qf = (np.mean(Ti) - np.mean(To)) * rn.x

        Tf = simulate_24h_steady_state(kf, To, Q_norm=qf)
        ssf = float(np.sum((Ti - Tf) ** 2))
        r2n = 1 - ssn / ss_tot
        r2f = 1 - ssf / ss_tot

        df2 = 24 - 2
        F = ((ssn - ssf) / 1.0) / (ssf / df2) if ssf > 0 else 0
        pv = 1 - f_dist.cdf(max(F, 0), 1, df2)

        res = Ti - Tn
        co = np.polyfit(To, res, 1)

        results.append({
            "rock": rid, "Ti": Ti, "To": To, "Tn": Tn, "Tf": Tf,
            "res": res, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "kf": kf, "qf": qf, "r2n": r2n, "r2f": r2f,
            "F": F, "p": pv, "tau": 1 / kf,
            "dT": float(np.mean(Ti) - np.mean(To)),
            "mr": float(np.mean(res)),
            "Q": qf * C_rock / 3600 * 1000,
            "U": kf * C_rock / 3600 * 1000,
            "Tc": -co[1] / co[0] if co[0] != 0 else np.nan,
            "sig": "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns",
            "C_eff_used": C_rock,
        })

    # ── BH-FDR ───────────────────────────────────────────────────────
    pvals = {r["rock"]: r["p"] for r in results}
    ids = sorted(pvals.keys())
    pv_arr = np.array([pvals[i] for i in ids])
    m = len(pv_arr)
    si = np.argsort(pv_arr); sp_arr = pv_arr[si]; sids = [ids[i] for i in si]
    adj = np.empty(m); adj[-1] = sp_arr[-1]
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i + 1], sp_arr[i] * m / (i + 1))
    adj = np.clip(adj, 0, 1)
    fdr = {sids[i]: {"raw": sp_arr[i], "adj": adj[i], "sig": adj[i] <= 0.05}
           for i in range(m)}

    # ── species crossover ────────────────────────────────────────────
    all_T = np.concatenate([r["To"] for r in results])
    all_r = np.concatenate([r["res"] for r in results])
    cc = np.polyfit(all_T, all_r, 1)
    Tc_all = -cc[1] / cc[0] if cc[0] != 0 else np.nan

    # ── print results ────────────────────────────────────────────────
    print(f"\n  {'Burrow':>7} {'τ(h)':>6} {'R²n':>5} {'R²f':>7} {'p_adj':>10} "
          f"{'FDR':>4} {'dT':>7} {'Q(mW)':>8} {'Q/RMR':>6}")
    print("  " + "-" * 80)
    for r in sorted(results, key=lambda x: x["dT"], reverse=True):
        fdr_s = "***" if fdr[r["rock"]]["sig"] else " ns"
        ratio = r["Q"] / rmr_field if rmr_field > 0 else 0
        print(f"  {r['rock']:>7d} {r['tau']:>6.2f} {r['r2n']:>5.2f} {r['r2f']:>7.4f} "
              f"{fdr[r['rock']]['adj']:>10.2e} {fdr_s:>4} {r['mr']:>+7.3f} "
              f"{r['Q']:>8.1f} {ratio:>+5.1f}×")
    n_fdr = sum(v["sig"] for v in fdr.values())
    n_heat = sum(1 for r in results if r["dT"] > 0 and fdr[r["rock"]]["sig"])
    print(f"  FDR significant: {n_fdr}/{len(results)}, of which {n_heat} heaters")
    if np.isfinite(Tc_all) and abs(Tc_all) < 50:
        print(f"  Species crossover (all burrows): {Tc_all:.1f}°C")

    # ── FIGURES ──────────────────────────────────────────────────────
    hours = np.arange(24)
    n = len(results)
    nc = 3 if n > 4 else 2
    nr = int(np.ceil(n / nc))

    # Fig 1: fits
    fig, axes = plt.subplots(nr, nc, figsize=(5.5 * nc, 4 * nr), sharex=True)
    flat = axes.flatten() if n > 1 else [axes]
    for idx, r in enumerate(sorted(results, key=lambda x: x["rock"])):
        ax = flat[idx]
        ax.fill_between(hours, r["ci_lo"], r["ci_hi"], alpha=0.12, color=color)
        ax.plot(hours, r["To"], "-", lw=1, color="#888", alpha=0.6, label="T_out")
        ax.plot(hours, r["Ti"], "o-", ms=3, lw=0.8, color=color, label="T_in obs")
        ax.plot(hours, r["Tn"], "--", lw=1.5, color="#7f7f7f",
                label=f"Null (R²={r['r2n']:.2f})")
        ax.plot(hours, r["Tf"], "-", lw=1.8, color="#1f77b4",
                label=f"+Wētā (R²={r['r2f']:.3f})")
        fdr_s = " FDR" if fdr[r["rock"]]["sig"] else ""
        ax.set_title(f"Burrow {r['rock']} (τ={r['tau']:.1f}h, Q={r['Q']:.1f}mW, "
                     f"{r['sig']}{fdr_s})", fontsize=9)
        ax.legend(fontsize=5.5, loc="best")
        ax.set_ylabel("T (°C)", fontsize=8)
        ax.grid(True, alpha=0.2)
    for idx in range(n, len(flat)): flat[idx].set_visible(False)
    for ax in flat[max(0, n - nc):n]: ax.set_xlabel("Hour of day")
    fig.suptitle(f"{sp_info.name}: {sp_info.wall_material.title()} Gallery "
                 f"Model Fits", fontsize=12, y=1.01)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(output_dir / f"{tag}_fig1_fits.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: residuals
    fig, axes = plt.subplots(nr, nc, figsize=(5.5 * nc, 4 * nr), sharex=True)
    flat = axes.flatten() if n > 1 else [axes]
    for idx, r in enumerate(sorted(results, key=lambda x: x["rock"])):
        ax = flat[idx]
        ax.fill_between(hours, 0, r["res"], where=r["res"] > 0,
                        color="#d62728", alpha=0.4, label="Heating")
        ax.fill_between(hours, 0, r["res"], where=r["res"] <= 0,
                        color="#1f77b4", alpha=0.4, label="Cooling")
        ax.plot(hours, r["res"], "k-", lw=1)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.axhline(r["mr"], color="#ff7f0e", lw=1.5, ls=":",
                   label=f"Mean: {r['mr']:+.3f}°C")
        ax.set_title(f"Burrow {r['rock']} (τ={r['tau']:.1f}h)", fontsize=9)
        ax.legend(fontsize=6)
        ax.set_ylabel("Residual ΔT (°C)", fontsize=8)
        ax.grid(True, alpha=0.2)
    for idx in range(n, len(flat)): flat[idx].set_visible(False)
    for ax in flat[max(0, n - nc):n]: ax.set_xlabel("Hour of day")
    fig.suptitle(f"{sp_info.name}: Lag-Corrected Wētā Signal", fontsize=12, y=1.01)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(output_dir / f"{tag}_fig2_residuals.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Fig 3: crossover
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.cm.Set1
    ax = axes[0]
    for i, r in enumerate(sorted(results, key=lambda x: x["rock"])):
        c = cmap(i)
        ax.scatter(r["To"], r["res"], s=12, alpha=0.3, color=c)
        co = np.polyfit(r["To"], r["res"], 1)
        xl = np.linspace(r["To"].min(), r["To"].max(), 50)
        ax.plot(xl, np.polyval(co, xl), "-", color=c, lw=1.5, alpha=0.8,
                label=f"B{r['rock']}")
    ax.axhline(0, color="k", lw=0.8)
    if np.isfinite(Tc_all) and abs(Tc_all) < 50:
        ax.axvline(Tc_all, color="#ff7f0e", lw=2, ls="--", alpha=0.7)
        yl = ax.get_ylim()
        ax.fill_between([all_T.min() - 1, Tc_all], 0, yl[1], alpha=0.04, color="red")
        ax.fill_between([Tc_all, all_T.max() + 1], yl[0], 0, alpha=0.04, color="blue")
        ax.set_ylim(yl)
    ax.text(0.03, 0.97, "WĒTĀ HEATS", transform=ax.transAxes, fontsize=8,
            color="#d62728", va="top", fontweight="bold", alpha=0.6)
    ax.text(0.97, 0.03, "WĒTĀ COOLS", transform=ax.transAxes, fontsize=8,
            color="#1f77b4", va="bottom", ha="right", fontweight="bold", alpha=0.6)
    ax.set_xlabel("T_out (°C)"); ax.set_ylabel("Lag-corrected ΔT (°C)")
    ax.set_title("A. Per-burrow"); ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.scatter(all_T, all_r, s=6, alpha=0.1, color=color)
    xsp = np.linspace(all_T.min() - 1, all_T.max() + 1, 100)
    ax.plot(xsp, cc[0] * xsp + cc[1], "-", color=color, lw=2.5,
            label=f"Crossover: {Tc_all:.1f}°C" if abs(Tc_all) < 50 else "Slope ~0")
    if np.isfinite(Tc_all) and abs(Tc_all) < 50:
        ax.plot(Tc_all, 0, "o", ms=12, color="#ff7f0e", markeredgecolor="k", zorder=6)
        yl = ax.get_ylim()
        ax.fill_between([all_T.min() - 1, Tc_all], 0, yl[1], alpha=0.04, color="red")
        ax.fill_between([Tc_all, all_T.max() + 1], yl[0], 0, alpha=0.04, color="blue")
        ax.set_ylim(yl)
    ax.axhline(0, color="k", lw=0.8)
    ax.text(0.03, 0.97, "WĒTĀ HEATS", transform=ax.transAxes, fontsize=8,
            color="#d62728", va="top", fontweight="bold", alpha=0.6)
    ax.text(0.97, 0.03, "WĒTĀ COOLS", transform=ax.transAxes, fontsize=8,
            color="#1f77b4", va="bottom", ha="right", fontweight="bold", alpha=0.6)
    ax.set_xlabel("T_out (°C)"); ax.set_ylabel("ΔT (°C)")
    ax.set_title(f"B. {sp_info.name} crossover"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.suptitle(f"{sp_info.name}: Heating–Cooling Crossover",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(output_dir / f"{tag}_fig3_crossover.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── CSV ──────────────────────────────────────────────────────────
    rows = []
    for r in results:
        rows.append({
            "species": sp_info.name, "burrow": r["rock"],
            "material": sp_info.wall_material,
            "tau_h": round(r["tau"], 2),
            "R2_null": round(r["r2n"], 4), "R2_full": round(r["r2f"], 4),
            "F": round(r["F"], 1), "p_value": f"{r['p']:.2e}",
            "p_adj_BH": f"{fdr[r['rock']]['adj']:.2e}",
            "FDR_sig": fdr[r["rock"]]["sig"],
            "dT_lag_corrected": round(r["mr"], 4),
            "Q_mW": round(r["Q"], 2), "U_mW_K": round(r["U"], 2),
            "Q_over_RMR": round(r["Q"] / rmr_field, 2) if rmr_field > 0 else None,
            "T_crossover": round(r["Tc"], 1) if abs(r["Tc"]) < 50 else None,
        })
    csv_path = output_dir / f"{tag}_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return results, fdr


# ┌─────────────────────────────────────────────────────────────────────┐
# │  MAIN                              « orchestrating the show »      │
# └─────────────────────────────────────────────────────────────────────┘

BANNER = """
 ┌──────────────────────────────────────────────────────────────┐
 │  IGLOO WĒTĀ  v0.2.0  — MULTI-SPECIES PIPELINE               │
 │  « H. maori · H. crassidens · H. thoracica »                │
 └──────────────────────────────────────────────────────────────┘
"""


def main():
    parser = argparse.ArgumentParser(prog="run_all_species")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="output")
    args = parser.parse_args()

    data_root = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parent.parent / "data"
    output_root = Path(args.output_dir)

    print(BANNER)
    t0 = time.time()

    # ── species config ───────────────────────────────────────────────
    species_runs = [
        (H_MAORI, "h_maori", True),
        (H_CRASSIDENS, "h_crassidens", False),
        (H_THORACICA, "h_thoracica", False),
    ]

    all_species_results = {}

    for sp_info, subdir, has_phys in species_runs:
        sp_dir = data_root / subdir
        h24_path = sp_dir / "24h_hourly_averages.csv"
        if not h24_path.is_file():
            print(f"\n  SKIP {sp_info.name}: {h24_path} not found")
            continue

        out_dir = output_root / subdir
        rock_phys = None
        if has_phys:
            phys_path = sp_dir / "Rock_data.xlsx"
            if phys_path.is_file():
                rock_phys = pd.read_excel(phys_path, sheet_name="Sheet1")

        results, fdr = run_species(sp_info, sp_dir, out_dir, rock_phys)
        all_species_results[sp_info.name] = (results, fdr, sp_info)

    # ── cross-species summary ────────────────────────────────────────
    print(f"\n{'='*100}")
    print("THREE-SPECIES COMPARISON")
    print(f"{'='*100}")
    print(f"  {'Species':<20} {'Material':>8} {'N':>3} {'FDR sig':>8} {'Heaters':>8} "
          f"{'Q range(mW)':>14} {'Q/RMR range':>14}")
    print("  " + "-" * 90)

    for sp_name, (res, fdr, sp_info) in all_species_results.items():
        n = len(res)
        n_sig = sum(v["sig"] for v in fdr.values())
        n_heat = sum(1 for r in res if r["dT"] > 0 and fdr[r["rock"]]["sig"])
        rmr = ALLOMETRY.a * sp_info.mass_g ** ALLOMETRY.b / ALLOMETRY.Q10 ** 1.2
        Qs = [r["Q"] for r in res]
        ratios = [r["Q"] / rmr for r in res if r["Q"] > 0]

        Q_range = f"{min(Qs):.0f} – {max(Qs):.0f}"
        r_range = (f"{min(ratios):.1f}–{max(ratios):.1f}×" if ratios else "—")

        print(f"  {sp_name:<20} {sp_info.wall_material:>8} {n:>3} "
              f"{n_sig}/{n:>6} {n_heat:>7}h "
              f"{Q_range:>14} {r_range:>14}")

    dt = time.time() - t0
    print(f"\n  Done in {dt:.1f}s. Output in {output_root}/")


if __name__ == "__main__":
    main()
