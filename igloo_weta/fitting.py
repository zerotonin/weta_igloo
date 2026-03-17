#!/usr/bin/env python3
"""
 ┌─────────────────────────────────────────────────────────────────────┐
 │  FITTING                              « dialling in the numbers »   │
 └─────────────────────────────────────────────────────────────────────┘

Parameter estimation for the burrow thermal model.  Fits the thermal
rate constant *k* and normalised heat input *q* from 24-h diurnal
temperature recordings, then converts to physical units using burrow
geometry.

Two models are fitted per rock:

1. **Null model** (passive thermal lag, Q_weta = 0):
   ``dT_in/dt = k · (T_out − T_in)``

2. **Full model** (lag + wētā heat):
   ``dT_in/dt = k · (T_out − T_in) + q``

An F-test determines whether the wētā heat term significantly improves
the fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import f as f_dist

from . import constants as C
from .physics import (
    BurrowPhysics,
    compute_amplitude_ratio,
    compute_burrow_physics,
    compute_phase_lag,
    simulate_24h_steady_state,
    simulate_burrow_temperature,
)


# ┌─────────────────────────────────────────────────────────────────────┐
# │  RESULT CONTAINER                         « the whole dossier »    │
# └─────────────────────────────────────────────────────────────────────┘
@dataclass
class RockResult:
    """Full analysis result for a single rock burrow.

    Attributes:
        rock:              Rock identifier number.
        has_phys:          Whether photogrammetric geometry is available.
        T_in_obs:          Observed inside temperature (24 h).
        T_out_obs:         Observed outside temperature (24 h).
        T_in_ci_lo:        Lower 95 % CI on inside temperature.
        T_in_ci_hi:        Upper 95 % CI on inside temperature.
        mean_dT_obs:       Mean observed ΔT = T_in − T_out (°C).
        phase_lag_obs:     Observed phase lag (hours).
        amp_ratio_obs:     Observed amplitude ratio.
        k_null:            Fitted k from null model (1/h).
        r2_null:           R² of null model.
        T_pred_null:       Predicted T_in from null model (24 h).
        phase_null:        Phase lag of null prediction.
        amp_null:          Amplitude ratio of null prediction.
        k_fit:             Fitted k from full model (1/h).
        q_fit:             Fitted normalised heat q (°C/h).
        r2_full:           R² of full model.
        T_pred_full:       Predicted T_in from full model (24 h).
        F_stat:            F-statistic for model comparison.
        p_value:           p-value of F-test.
        residual_null:     Lag-corrected residual = T_obs − T_null.
        mean_residual:     Mean of residual_null.
        tau_hours:         Thermal time constant 1/k (hours).
        phys:              :class:`BurrowPhysics` or ``None``.
        U_fit_W_K:         Thermal conductance U (W/K) or ``None``.
        Q_weta_W:          Metabolic heat output (W) or ``None``.
        T_cross_raw:       Raw crossover temperature (°C).
        T_cross_corr:      Lag-corrected crossover temperature (°C).
        slope_corr:        Slope of corrected ΔT vs T_out regression.
        intercept_corr:    Intercept of corrected regression.
    """

    rock: int
    has_phys: bool
    # ── observed ──
    T_in_obs: np.ndarray
    T_out_obs: np.ndarray
    T_in_ci_lo: np.ndarray
    T_in_ci_hi: np.ndarray
    mean_dT_obs: float
    phase_lag_obs: int
    amp_ratio_obs: float
    # ── null model ──
    k_null: float
    r2_null: float
    T_pred_null: np.ndarray
    phase_null: int
    amp_null: float
    # ── full model ──
    k_fit: float
    q_fit: float
    r2_full: float
    T_pred_full: np.ndarray
    F_stat: float
    p_value: float
    # ── residuals ──
    residual_null: np.ndarray
    mean_residual: float
    # ── derived ──
    tau_hours: float
    phys: Optional[BurrowPhysics]
    U_fit_W_K: Optional[float]
    Q_weta_W: Optional[float]
    # ── crossover ──
    T_cross_raw: float
    T_cross_corr: float
    slope_corr: float
    intercept_corr: float


# ┌─────────────────────────────────────────────────────────────────────┐
# │  INCUBATOR VALIDATION               « the passive control »       │
# └─────────────────────────────────────────────────────────────────────┘
@dataclass
class IncubatorResult:
    """Result from fitting the passive incubator control experiment.

    Attributes:
        k_wood:   Fitted thermal rate constant for wood burrow (1/h).
        T_pred:   Predicted inside temperature series.
        T_in:     Observed inside temperature series.
        T_out:    Observed outside temperature series.
        hours:    Elapsed hour array.
        r2:       Coefficient of determination.
    """

    k_wood: float
    T_pred: np.ndarray
    T_in: np.ndarray
    T_out: np.ndarray
    hours: np.ndarray
    r2: float


def fit_incubator(incubator_df: pd.DataFrame) -> IncubatorResult:
    """Fit thermal rate constant from the passive incubator experiment.

    No wētā were present — validates the modelling framework on wood
    burrows where Q_weta is known to be zero.

    Args:
        incubator_df: Output of :func:`~igloo_weta.ingest.load_incubator`.

    Returns:
        :class:`IncubatorResult` with fitted k, predictions, and R².
    """
    t_in = incubator_df["temperature_in_C_mean"].values
    t_out = incubator_df["temperature_out_C_mean"].values
    hours = incubator_df["elapsed_hour"].values

    mask = ~(np.isnan(t_in) | np.isnan(t_out))
    t_in, t_out, hours = t_in[mask], t_out[mask], hours[mask]

    def _cost(k: float) -> float:
        pred = simulate_burrow_temperature(k, t_out, t_in[0])
        sse = float(np.nansum((pred - t_in) ** 2))
        return sse if np.isfinite(sse) else 1e20

    res = minimize_scalar(_cost, bounds=(0.01, 5.0), method="bounded")
    k = res.x
    T_pred = simulate_burrow_temperature(k, t_out, t_in[0])

    ss_res = float(np.nansum((t_in - T_pred) ** 2))
    ss_tot = float(np.nansum((t_in - np.nanmean(t_in)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return IncubatorResult(
        k_wood=k, T_pred=T_pred, T_in=t_in, T_out=t_out, hours=hours, r2=r2
    )


# ┌─────────────────────────────────────────────────────────────────────┐
# │  PER-ROCK FITTING                       « the main event »         │
# └─────────────────────────────────────────────────────────────────────┘
def fit_single_rock(
    rock_id: int,
    hourly_24h: pd.DataFrame,
    rock_phys_df: pd.DataFrame,
    shell_m: float = C.SHELL_THICKNESS_M,
) -> Optional[RockResult]:
    """Fit thermal model for a single rock burrow.

    Fits both the null (passive) and full (passive + wētā) models,
    performs an F-test, computes lag-corrected residuals and crossover
    temperatures.

    Args:
        rock_id:       Numeric rock identifier.
        hourly_24h:    Full 24-h hourly average DataFrame (all rocks).
        rock_phys_df:  Photogrammetric geometry DataFrame.
        shell_m:       Stone shell thickness in metres.

    Returns:
        :class:`RockResult` or ``None`` if insufficient data.
    """
    sub = hourly_24h[hourly_24h["rock"] == rock_id].sort_values("Hour")
    T_in_obs = sub["inside_mean"].values
    T_out_obs = sub["outside_mean"].values
    T_in_ci_lo = sub["inside_ci_lower_mean"].values
    T_in_ci_hi = sub["inside_ci_upper_mean"].values

    if len(T_in_obs) < 24:
        return None

    # ── geometry ─────────────────────────────────────────────────────
    rrow = rock_phys_df[rock_phys_df["Rock number"] == rock_id]
    has_phys = (
        len(rrow) > 0
        and not pd.isna(rrow.iloc[0]["Total Volume (cm3)"])
        and not pd.isna(rrow.iloc[0]["Total Surface area (cm2)"])
    )
    phys = None
    if has_phys:
        phys = compute_burrow_physics(
            rrow.iloc[0]["Total Volume (cm3)"],
            rrow.iloc[0]["Total Surface area (cm2)"],
            shell_m=shell_m,
        )

    # ── diagnostics ──────────────────────────────────────────────────
    phase_lag_obs, _ = compute_phase_lag(T_in_obs, T_out_obs)
    amp_ratio_obs = compute_amplitude_ratio(T_in_obs, T_out_obs)
    mean_dT_obs = float(np.mean(T_in_obs) - np.mean(T_out_obs))
    ss_tot = float(np.sum((T_in_obs - np.mean(T_in_obs)) ** 2))

    # ── null model ───────────────────────────────────────────────────
    def _cost_null(k: float) -> float:
        if k <= 0:
            return 1e10
        pred = simulate_24h_steady_state(k, T_out_obs)
        sse = float(np.nansum((pred - T_in_obs) ** 2))
        return sse if np.isfinite(sse) else 1e10

    res_null = minimize_scalar(_cost_null, bounds=C.K_FIT_BOUNDS, method="bounded")
    k_null = res_null.x
    T_pred_null = simulate_24h_steady_state(k_null, T_out_obs)
    ss_null = float(np.sum((T_in_obs - T_pred_null) ** 2))
    r2_null = 1.0 - ss_null / ss_tot if ss_tot > 0 else 0.0
    phase_null, _ = compute_phase_lag(T_pred_null, T_out_obs)
    amp_null = compute_amplitude_ratio(T_pred_null, T_out_obs)

    # ── full model ───────────────────────────────────────────────────
    def _cost_full(params: np.ndarray) -> float:
        k, q = params
        if k <= 0:
            return 1e10
        pred = simulate_24h_steady_state(k, T_out_obs, Q_norm=q)
        sse = float(np.nansum((pred - T_in_obs) ** 2))
        return sse if np.isfinite(sse) else 1e10

    res_full = minimize(
        _cost_full,
        [k_null, 0.0],
        method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 50000},
    )
    k_fit, q_fit = res_full.x

    # guard degenerate fits
    if k_fit > C.K_FIT_BOUNDS[1] or k_fit <= 0:
        k_fit = k_null
        q_fit = mean_dT_obs * k_null

    T_pred_full = simulate_24h_steady_state(k_fit, T_out_obs, Q_norm=q_fit)
    ss_full = float(np.sum((T_in_obs - T_pred_full) ** 2))
    r2_full = 1.0 - ss_full / ss_tot if ss_tot > 0 else 0.0

    # ── F-test ───────────────────────────────────────────────────────
    df2 = 24 - 2
    if ss_full > 0 and df2 > 0:
        F_stat = ((ss_null - ss_full) / 1.0) / (ss_full / df2)
        p_value = 1.0 - f_dist.cdf(max(F_stat, 0), 1, df2)
    else:
        F_stat, p_value = np.nan, np.nan

    # ── residuals ────────────────────────────────────────────────────
    residual_null = T_in_obs - T_pred_null
    mean_residual = float(np.mean(residual_null))

    # ── physical units ───────────────────────────────────────────────
    tau_hours = 1.0 / k_fit if k_fit > 0 else np.nan
    U_fit, Q_weta = None, None
    if phys is not None:
        U_fit = k_fit * phys.C_eff / 3600.0
        Q_weta = q_fit * phys.C_eff / 3600.0

    # ── crossover temperatures ───────────────────────────────────────
    coeffs_corr = np.polyfit(T_out_obs, residual_null, 1)
    a_c, b_c = coeffs_corr
    T_cross_corr = -b_c / a_c if a_c != 0 else np.nan

    coeffs_raw = np.polyfit(T_out_obs, T_in_obs - T_out_obs, 1)
    a_r, b_r = coeffs_raw
    T_cross_raw = -b_r / a_r if a_r != 0 else np.nan

    return RockResult(
        rock=rock_id,
        has_phys=has_phys,
        T_in_obs=T_in_obs,
        T_out_obs=T_out_obs,
        T_in_ci_lo=T_in_ci_lo,
        T_in_ci_hi=T_in_ci_hi,
        mean_dT_obs=mean_dT_obs,
        phase_lag_obs=phase_lag_obs,
        amp_ratio_obs=amp_ratio_obs,
        k_null=k_null,
        r2_null=r2_null,
        T_pred_null=T_pred_null,
        phase_null=phase_null,
        amp_null=amp_null,
        k_fit=k_fit,
        q_fit=q_fit,
        r2_full=r2_full,
        T_pred_full=T_pred_full,
        F_stat=F_stat,
        p_value=p_value,
        residual_null=residual_null,
        mean_residual=mean_residual,
        tau_hours=tau_hours,
        phys=phys,
        U_fit_W_K=U_fit,
        Q_weta_W=Q_weta,
        T_cross_raw=T_cross_raw,
        T_cross_corr=T_cross_corr,
        slope_corr=float(a_c),
        intercept_corr=float(b_c),
    )


def fit_all_rocks(
    hourly_24h: pd.DataFrame,
    rock_phys_df: pd.DataFrame,
    shell_m: float = C.SHELL_THICKNESS_M,
) -> list[RockResult]:
    """Fit thermal models for every rock in the dataset.

    Args:
        hourly_24h:    24-h hourly averages (all rocks).
        rock_phys_df:  Photogrammetric geometry.
        shell_m:       Stone shell thickness in metres.

    Returns:
        List of :class:`RockResult`, one per rock with sufficient data.
    """
    rock_ids = sorted(hourly_24h["rock"].unique())
    results = []
    for rid in rock_ids:
        r = fit_single_rock(rid, hourly_24h, rock_phys_df, shell_m=shell_m)
        if r is not None:
            results.append(r)
    return results


def compute_species_crossover(
    results: list[RockResult],
    exclude: Optional[list[int]] = None,
) -> dict:
    """Compute the species-level crossover temperature.

    Pools lag-corrected residuals across rocks (excluding specified IDs)
    and fits a linear regression of ΔT_corrected vs T_out.

    Args:
        results: List of :class:`RockResult`.
        exclude: Rock IDs to exclude (default: :data:`constants.EXCLUDE_ROCK_IDS`).

    Returns:
        Dict with keys ``T_cross_raw``, ``T_cross_corr``, ``slope_raw``,
        ``slope_corr``, ``intercept_raw``, ``intercept_corr``.
    """
    if exclude is None:
        exclude = C.EXCLUDE_ROCK_IDS

    T_pool, dT_raw_pool, dT_corr_pool = [], [], []
    for r in results:
        if r.rock in exclude:
            continue
        T_pool.extend(r.T_out_obs)
        dT_raw_pool.extend(r.T_in_obs - r.T_out_obs)
        dT_corr_pool.extend(r.residual_null)

    T_arr = np.array(T_pool)
    raw_arr = np.array(dT_raw_pool)
    corr_arr = np.array(dT_corr_pool)

    c_raw = np.polyfit(T_arr, raw_arr, 1)
    c_corr = np.polyfit(T_arr, corr_arr, 1)

    return {
        "T_cross_raw": -c_raw[1] / c_raw[0],
        "T_cross_corr": -c_corr[1] / c_corr[0],
        "slope_raw": c_raw[0],
        "slope_corr": c_corr[0],
        "intercept_raw": c_raw[1],
        "intercept_corr": c_corr[1],
        "T_pool": T_arr,
        "dT_raw_pool": raw_arr,
        "dT_corr_pool": corr_arr,
    }
