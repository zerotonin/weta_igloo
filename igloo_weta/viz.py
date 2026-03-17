 #!/usr/bin/env python3
"""
 ┌─────────────────────────────────────────────────────────────────────┐
 │  VIZ                                    « painting the picture »    │
 └─────────────────────────────────────────────────────────────────────┘

Publication-quality figure generation for the wētā burrow thermal model.
Every figure is exported as SVG (editable text), PNG, and accompanying
CSV data tables.

SVG output uses editable fonts — open in Inkscape or Illustrator to
tweak labels without re-running the pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import constants as C
from .fitting import IncubatorResult, RockResult
from .sensitivity import SpeciesRMR, ThicknessPoint

# ── matplotlib config for editable SVG text ──────────────────────────
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as <text>, not paths


# ┌─────────────────────────────────────────────────────────────────────┐
# │  HELPERS                                « utility belt »            │
# └─────────────────────────────────────────────────────────────────────┘
def _setup_style() -> None:
    """Apply consistent rcParams for all figures."""
    plt.rcParams.update({
        "font.family": C.SVG_FONT_FAMILY,
        "font.size": 10,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })


def _save(fig: plt.Figure, stem: str, output_dir: str) -> list[str]:
    """Save figure as SVG, PNG, and return list of written paths.

    Args:
        fig:        Matplotlib figure.
        stem:       Filename stem (no extension).
        output_dir: Target directory.

    Returns:
        List of absolute paths written.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for ext in ("svg", "png"):
        p = os.path.join(output_dir, f"{stem}.{ext}")
        dpi = C.FIG_DPI if ext == "png" else None
        fig.savefig(p, dpi=dpi, bbox_inches="tight", format=ext)
        paths.append(p)
    return paths


def _save_csv(df: pd.DataFrame, stem: str, output_dir: str) -> str:
    """Write a DataFrame as CSV and return the path.

    Args:
        df:         DataFrame to export.
        stem:       Filename stem.
        output_dir: Target directory.

    Returns:
        Absolute path of the written CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    p = os.path.join(output_dir, f"{stem}.csv")
    df.to_csv(p, index=False)
    return p


# ┌─────────────────────────────────────────────────────────────────────┐
# │  FIGURE 1: INCUBATOR VALIDATION      « the passive sanity check »   │
# └─────────────────────────────────────────────────────────────────────┘
def plot_incubator(
    inc: IncubatorResult,
    output_dir: str,
) -> list[str]:
    """Plot incubator passive control: observed vs predicted.

    Args:
        inc:        :class:`~fitting.IncubatorResult`.
        output_dir: Output directory for files.

    Returns:
        List of written file paths (SVG, PNG, CSV).
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(inc.hours, inc.T_out, "k-", lw=1.2, label="T outside", alpha=0.7)
    ax.plot(inc.hours, inc.T_in, "o", ms=2, color="#d62728",
            label="T inside (obs)", alpha=0.6)
    ax.plot(inc.hours, inc.T_pred, "-", lw=1.5, color="#1f77b4",
            label=f"Null model (k={inc.k_wood:.3f}/h, R²={inc.r2:.3f})")

    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Incubator Validation: Passive Heat Exchange in Wood Burrow")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    paths = _save(fig, "fig1_incubator_validation", output_dir)
    plt.close(fig)

    # CSV companion
    df = pd.DataFrame({
        "elapsed_hour": inc.hours,
        "T_outside_C": inc.T_out,
        "T_inside_observed_C": inc.T_in,
        "T_inside_predicted_C": inc.T_pred,
    })
    paths.append(_save_csv(df, "fig1_incubator_data", output_dir))
    return paths


# ┌─────────────────────────────────────────────────────────────────────┐
# │  FIGURE 2: PER-ROCK 24H FITS          « the diurnal gallery »       │
# └─────────────────────────────────────────────────────────────────────┘
def plot_per_rock_fits(
    results: list[RockResult],
    output_dir: str,
) -> list[str]:
    """Plot 24-h model fits for each rock (null + full model).

    Args:
        results:    List of :class:`~fitting.RockResult`.
        output_dir: Output directory.

    Returns:
        Written file paths.
    """
    _setup_style()
    n = len(results)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))
    hours = np.arange(24)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.8 * n_rows),
                             sharex=True)
    flat = axes.flatten()

    for idx, r in enumerate(sorted(results, key=lambda x: x.rock)):
        ax = flat[idx]
        ax.fill_between(hours, r.T_in_ci_lo, r.T_in_ci_hi,
                        alpha=0.12, color="#d62728")
        ax.plot(hours, r.T_out_obs, "-", lw=1, color="#888", alpha=0.6,
                label="T_out")
        ax.plot(hours, r.T_in_obs, "o-", ms=3, lw=0.8, color="#d62728",
                label="T_in obs")
        ax.plot(hours, r.T_pred_null, "--", lw=1.5, color="#7f7f7f",
                label=f"Null (R²={r.r2_null:.2f})")
        ax.plot(hours, r.T_pred_full, "-", lw=1.8, color="#1f77b4",
                label=f"+Wētā (R²={r.r2_full:.2f})")

        sig = ("***" if r.p_value < 0.001 else "**" if r.p_value < 0.01
               else "*" if r.p_value < 0.05 else "ns")
        tau_s = f"τ={r.tau_hours:.1f}h" if r.tau_hours < 10 else "τ→0"
        ax.set_title(f"Rock {r.rock} ({tau_s}, {sig})", fontsize=9)
        ax.legend(fontsize=5.5, loc="best")
        ax.set_ylabel("T (°C)", fontsize=8)
        ax.grid(True, alpha=0.2)

    for idx in range(n, len(flat)):
        flat[idx].set_visible(False)
    for ax in flat[max(0, n - n_cols):n]:
        ax.set_xlabel("Hour of day")

    fig.suptitle(
        "Burrow Temperature: Passive Lag (Null) vs Lag + Wētā Heat (Full)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    paths = _save(fig, "fig2_per_rock_fits", output_dir)
    plt.close(fig)
    return paths


# ┌─────────────────────────────────────────────────────────────────────┐
# │  FIGURE 3: RESIDUALS                 « the weta signal »            │
# └─────────────────────────────────────────────────────────────────────┘
def plot_residuals(
    results: list[RockResult],
    output_dir: str,
) -> list[str]:
    """Plot lag-corrected residuals (T_obs − T_null) per rock.

    Args:
        results:    List of :class:`~fitting.RockResult`.
        output_dir: Output directory.

    Returns:
        Written file paths.
    """
    _setup_style()
    n = len(results)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))
    hours = np.arange(24)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.8 * n_rows),
                             sharex=True)
    flat = axes.flatten()

    for idx, r in enumerate(sorted(results, key=lambda x: x.rock)):
        ax = flat[idx]
        res = r.residual_null
        ax.fill_between(hours, 0, res, where=res > 0,
                        color="#d62728", alpha=0.4, label="Heating")
        ax.fill_between(hours, 0, res, where=res <= 0,
                        color="#1f77b4", alpha=0.4, label="Cooling")
        ax.plot(hours, res, "k-", lw=1)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.axhline(np.mean(res), color="#ff7f0e", lw=1.5, ls=":",
                   label=f"Mean: {np.mean(res):+.3f}°C")

        tau_s = f"τ={r.tau_hours:.1f}h" if r.tau_hours < 10 else "τ→0"
        ax.set_title(f"Rock {r.rock} ({tau_s})", fontsize=9)
        ax.legend(fontsize=6)
        ax.set_ylabel("Residual ΔT (°C)", fontsize=8)
        ax.grid(True, alpha=0.2)

    for idx in range(n, len(flat)):
        flat[idx].set_visible(False)
    for ax in flat[max(0, n - n_cols):n]:
        ax.set_xlabel("Hour of day")

    fig.suptitle(
        "Lag-Corrected Residuals: Wētā Signal After Removing Thermal Inertia",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    paths = _save(fig, "fig3_residuals", output_dir)
    plt.close(fig)
    return paths


# ┌─────────────────────────────────────────────────────────────────────┐
# │  FIGURE 4: CROSSOVER                 « heating or cooling? »        │
# └─────────────────────────────────────────────────────────────────────┘
def plot_crossover(
    results: list[RockResult],
    species_cross: dict,
    output_dir: str,
) -> list[str]:
    """Plot the heating–cooling crossover analysis (4 panels).

    Args:
        results:       List of :class:`~fitting.RockResult`.
        species_cross: Output of :func:`~fitting.compute_species_crossover`.
        output_dir:    Output directory.

    Returns:
        Written file paths.
    """
    _setup_style()
    from matplotlib.gridspec import GridSpec

    cmap = plt.cm.tab10
    Tc_raw = species_cross["T_cross_raw"]
    Tc_corr = species_cross["T_cross_corr"]

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ── A: Raw ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    for i, r in enumerate(sorted(results, key=lambda x: x.rock)):
        c = cmap(i)
        dT = r.T_in_obs - r.T_out_obs
        ax.scatter(r.T_out_obs, dT, s=10, alpha=0.3, color=c)
        co = np.polyfit(r.T_out_obs, dT, 1)
        xl = np.linspace(r.T_out_obs.min(), r.T_out_obs.max(), 50)
        st = ":" if r.rock in C.EXCLUDE_ROCK_IDS else "-"
        ax.plot(xl, np.polyval(co, xl), st, color=c, lw=1.3, alpha=0.8,
                label=f"R{r.rock}")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(Tc_raw, color="#ff7f0e", lw=2, ls="--", alpha=0.7)
    ax.set_xlabel("T_out (°C)")
    ax.set_ylabel("Raw ΔT (°C)")
    ax.set_title("A. Uncorrected")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.2)

    # ── B: Corrected ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    for i, r in enumerate(sorted(results, key=lambda x: x.rock)):
        c = cmap(i)
        ax.scatter(r.T_out_obs, r.residual_null, s=10, alpha=0.3, color=c)
        co = np.polyfit(r.T_out_obs, r.residual_null, 1)
        xl = np.linspace(r.T_out_obs.min(), r.T_out_obs.max(), 50)
        st = ":" if r.rock in C.EXCLUDE_ROCK_IDS else "-"
        ax.plot(xl, np.polyval(co, xl), st, color=c, lw=1.3, alpha=0.8,
                label=f"R{r.rock}")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(Tc_corr, color="#ff7f0e", lw=2, ls="--", alpha=0.7)
    ax.set_xlabel("T_out (°C)")
    ax.set_ylabel("Lag-corrected ΔT (°C)")
    ax.set_title("B. Thermal inertia removed")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.2)

    # ── C: Species pooled ────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    T_p = species_cross["T_pool"]
    ax.scatter(T_p, species_cross["dT_raw_pool"], s=5, alpha=0.06, color="#888")
    ax.scatter(T_p, species_cross["dT_corr_pool"], s=5, alpha=0.06,
               color="#1f77b4")
    xsp = np.linspace(T_p.min() - 1, T_p.max() + 1, 100)
    ax.plot(xsp,
            species_cross["slope_raw"] * xsp + species_cross["intercept_raw"],
            "--", color="#888", lw=2, label=f"Raw: {Tc_raw:.1f}°C")
    ax.plot(xsp,
            species_cross["slope_corr"] * xsp + species_cross["intercept_corr"],
            "-", color="#1f77b4", lw=2.5, label=f"Corrected: {Tc_corr:.1f}°C")
    ax.plot(Tc_corr, 0, "o", ms=12, color="#ff7f0e", markeredgecolor="k",
            zorder=6)
    ax.axhline(0, color="k", lw=0.8)

    # ── quadrant shading: heats left-above, cools right-below ────────
    ylims_c = ax.get_ylim()
    ax.fill_between(
        [xsp[0], Tc_corr], 0, ylims_c[1], alpha=0.04, color="red", zorder=0,
    )
    ax.fill_between(
        [Tc_corr, xsp[-1]], ylims_c[0], 0, alpha=0.04, color="blue", zorder=0,
    )
    ax.set_ylim(ylims_c)
    ax.text(0.03, 0.97, "WĒTĀ HEATS", transform=ax.transAxes, fontsize=8,
            color="#d62728", va="top", fontweight="bold", alpha=0.6)
    ax.text(0.97, 0.03, "WĒTĀ COOLS", transform=ax.transAxes, fontsize=8,
            color="#1f77b4", va="bottom", ha="right", fontweight="bold", alpha=0.6)

    ax.set_xlabel("T_out (°C)")
    ax.set_ylabel("ΔT (°C)")
    ax.set_title("C. Species-level crossover (excl. R22)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── D: Q vs T_out ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    heaters = [r for r in results
               if r.mean_dT_obs > 0 and r.has_phys
               and r.rock not in C.EXCLUDE_ROCK_IDS and r.tau_hours < 10]
    T_range = np.linspace(4, 20, 100)
    for i, r in enumerate(heaters):
        dT_pred = r.slope_corr * T_range + r.intercept_corr
        Q_mW = r.U_fit_W_K * dT_pred * 1000 if r.U_fit_W_K else np.zeros_like(T_range)
        ax.plot(T_range, Q_mW, "-", color=cmap(i), lw=1.5,
                label=f"R{r.rock}", alpha=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(Tc_corr, color="#d62728", lw=1, ls="--", alpha=0.5)

    # ── quadrant shading: heats left-above, cools right-below ────────
    ylims_d = ax.get_ylim()
    ax.fill_between(
        [T_range[0], Tc_corr], 0, ylims_d[1],
        alpha=0.04, color="red", zorder=0,
    )
    ax.fill_between(
        [Tc_corr, T_range[-1]], ylims_d[0], 0,
        alpha=0.04, color="blue", zorder=0,
    )
    ax.set_ylim(ylims_d)
    ax.text(0.03, 0.97, "WĒTĀ HEATS", transform=ax.transAxes, fontsize=8,
            color="#d62728", va="top", fontweight="bold", alpha=0.6)
    ax.text(0.97, 0.03, "WĒTĀ COOLS", transform=ax.transAxes, fontsize=8,
            color="#1f77b4", va="bottom", ha="right", fontweight="bold", alpha=0.6)

    ax.set_xlabel("T_out (°C)")
    ax.set_ylabel("Q_wētā (mW)")
    ax.set_title("D. Predicted metabolic output vs T_out")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    fig.suptitle("Heating–Cooling Crossover: Corrected for Thermal Inertia",
                 fontsize=13, fontweight="bold", y=1.01)
    paths = _save(fig, "fig4_crossover", output_dir)
    plt.close(fig)
    return paths


# ┌─────────────────────────────────────────────────────────────────────┐
# │  FIGURE 5: SPECIES SENSITIVITY     « how thick is the igloo? »      │
# └─────────────────────────────────────────────────────────────────────┘
def plot_species_sensitivity(
    results: list[RockResult],
    sweep_data: dict[int, list[ThicknessPoint]],
    species_rmr: dict[str, SpeciesRMR],
    output_dir: str,
) -> list[str]:
    """Plot shell thickness sensitivity with species RMR bands.

    Top row: Q vs thickness.  Bottom row: Q/RMR ratio (log scale).
    One column per species.

    Args:
        results:     List of :class:`~fitting.RockResult`.
        sweep_data:  Output of :func:`~sensitivity.sweep_all_rocks`.
        species_rmr: Output of :func:`~sensitivity.compute_species_rmr`.
        output_dir:  Output directory.

    Returns:
        Written file paths.
    """
    _setup_style()
    from matplotlib.gridspec import GridSpec

    heater_ids = [r.rock for r in results
                  if r.mean_dT_obs > 0 and r.has_phys
                  and r.rock not in C.EXCLUDE_ROCK_IDS
                  and r.tau_hours < 10 and r.rock in sweep_data]

    thick_fine = np.linspace(0.1, 5.0, 200)
    rock_cm = plt.cm.Set1
    species_order = ["H. maori", "H. thoracica", "H. crassidens"]

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    for col, sp in enumerate(species_order):
        if sp not in species_rmr:
            continue
        info = species_rmr[sp]

        # ── top row: Q vs thickness ──────────────────────────────────
        ax = fig.add_subplot(gs[0, col])
        thermo_lo = info.rmr_10 * 5
        thermo_hi = info.rmr_10 * 20

        ax.axhspan(info.rmr_5, info.rmr_15, alpha=0.12, color=info.color)
        ax.axhline(info.rmr_10, color=info.color, lw=2, ls=":", alpha=0.8,
                   label=f"RMR@10°C: {info.rmr_10:.1f}mW")
        ax.axhspan(thermo_lo, thermo_hi, alpha=0.06, color="purple")

        for i, rid in enumerate(heater_ids):
            Q_at_1 = next(
                (p.Q_mW for p in sweep_data[rid] if p.thickness_cm == 1.0), 0
            )
            Q_fine = Q_at_1 * thick_fine
            rr = next(r for r in results if r.rock == rid)
            sig = "**" if rr.p_value < 0.01 else "*" if rr.p_value < 0.05 else ""
            ax.plot(thick_fine, Q_fine, "-", color=rock_cm(i), lw=2, alpha=0.8,
                    label=f"R{rid} {sig}")

        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("Shell thickness (cm)")
        ax.set_ylabel("Q (mW)")
        ax.set_title(f"{sp} ({info.mass_g:.1f} g)", fontsize=11,
                     fontweight="bold", color=info.color)
        ax.legend(fontsize=6.5, loc="upper left")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, 5)
        ax.set_ylim(-20, thermo_hi * 1.3)

        # ── bottom row: Q/RMR ratio ─────────────────────────────────
        ax = fig.add_subplot(gs[1, col])
        for i, rid in enumerate(heater_ids):
            Q_at_1 = next(
                (p.Q_mW for p in sweep_data[rid] if p.thickness_cm == 1.0), 0
            )
            ratio_fine = (Q_at_1 * thick_fine) / info.rmr_10
            rr = next(r for r in results if r.rock == rid)
            sig = "**" if rr.p_value < 0.01 else "*" if rr.p_value < 0.05 else ""
            ax.plot(thick_fine, ratio_fine, "-", color=rock_cm(i), lw=2,
                    alpha=0.8, label=f"R{rid} {sig}")

        ax.axhline(1, color="k", lw=1.5, ls="-", alpha=0.5, label="1× RMR")
        ax.axhspan(5, 20, alpha=0.08, color="purple", label="Thermogenesis (5–20×)")
        ax.set_xlabel("Shell thickness (cm)")
        ax.set_ylabel("Q_model / RMR")
        ax.set_title(f"{sp}: multiple of resting metabolism", fontsize=10)
        ax.legend(fontsize=6.5, loc="upper left")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, 5)
        ax.set_yscale("log")
        ax.set_ylim(0.3, 150)

    fig.suptitle(
        "Shell Thickness Sensitivity: Model Heat Output vs Species Metabolic Rate\n"
        "(RMR: 10.5·M⁰·⁷⁵ mW at 25°C, Q₁₀=2.5)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    paths = _save(fig, "fig5_species_sensitivity", output_dir)
    plt.close(fig)
    return paths


# ┌─────────────────────────────────────────────────────────────────────┐
# │  RESULTS TABLE                       « the final scoreboard »       │
# └─────────────────────────────────────────────────────────────────────┘
def export_results_csv(
    results: list[RockResult],
    output_dir: str,
) -> str:
    """Export per-rock results as a CSV table.

    Args:
        results:    List of :class:`~fitting.RockResult`.
        output_dir: Output directory.

    Returns:
        Path to the written CSV.
    """
    rows = []
    for r in sorted(results, key=lambda x: x.rock):
        rows.append({
            "rock": r.rock,
            "tau_hours": r.tau_hours if r.tau_hours < 10 else None,
            "k_fit_per_hour": r.k_fit if r.k_fit < 10 else None,
            "R2_null": round(r.r2_null, 4),
            "R2_full": round(r.r2_full, 4),
            "F_statistic": round(r.F_stat, 2) if not np.isnan(r.F_stat) else None,
            "p_value": round(r.p_value, 5),
            "mean_dT_raw_C": round(r.mean_dT_obs, 4),
            "mean_dT_lag_corrected_C": round(r.mean_residual, 4),
            "Q_weta_mW": (round(r.Q_weta_W * 1000, 1)
                          if r.Q_weta_W is not None and r.tau_hours < 10
                          else None),
            "U_mW_K": (round(r.U_fit_W_K * 1000, 1)
                       if r.U_fit_W_K is not None else None),
            "T_crossover_raw_C": (round(r.T_cross_raw, 1)
                                  if abs(r.T_cross_raw) < 30 else None),
            "T_crossover_corrected_C": (round(r.T_cross_corr, 1)
                                        if abs(r.T_cross_corr) < 30 else None),
            "cavity_volume_cm3": (round(r.phys.V_cavity_m3 * 1e6, 1)
                                  if r.phys else None),
            "inner_SA_cm2": (round(r.phys.SA_m2 * 1e4, 1)
                             if r.phys else None),
        })
    return _save_csv(pd.DataFrame(rows), "results_table", output_dir)