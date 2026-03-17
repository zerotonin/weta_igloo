#!/usr/bin/env python3
"""
 ┌─────────────────────────────────────────────────────────────────────┐
 │  PHYSICS                             « the laws of thermodynamics » │
 └─────────────────────────────────────────────────────────────────────┘

Lumped-parameter thermal model for stone-shingle wētā burrows.

The governing ODE::

    C_eff · dT_in/dt  =  U · (T_out(t) − T_in(t))  +  Q_weta

where *C_eff* is the effective heat capacity of the burrow (stone shell +
enclosed air), *U* is the overall thermal conductance, and *Q_weta* is
the metabolic heat production of the wētā.

Defining  k = U / C_eff  (thermal rate constant, 1/s)::

    dT_in/dt  =  k · (T_out − T_in)  +  Q_weta / C_eff

The **null model** sets Q_weta = 0.  Any systematic positive residual
(T_in_observed > T_in_null) implies active heating by the wētā.

Analogous to the IGLOO model (Giraldo et al. 2019, Sci Rep 9:3974),
which used the same heat conduction framework for *Drosophila* body
temperature in thermal gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import constants as C


# ┌─────────────────────────────────────────────────────────────────────┐
# │  BURROW GEOMETRY                    « measuring the stone igloo »   │
# └─────────────────────────────────────────────────────────────────────┘
@dataclass
class BurrowPhysics:
    """Physical properties of a stone-shingle burrow cavity.

    All quantities in SI units unless noted.

    Attributes:
        C_eff:       Effective heat capacity [J/K].
        C_stone:     Stone shell heat capacity [J/K].
        C_air:       Enclosed air heat capacity [J/K].
        V_cavity_m3: Air cavity volume [m³].
        SA_m2:       Inner wall surface area [m²].
        M_stone_kg:  Stone shell mass [kg].
        V_shell_m3:  Stone shell volume [m³].
        shell_m:     Shell thickness used [m].
    """

    C_eff: float
    C_stone: float
    C_air: float
    V_cavity_m3: float
    SA_m2: float
    M_stone_kg: float
    V_shell_m3: float
    shell_m: float


def compute_burrow_physics(
    volume_cm3: float,
    surface_area_cm2: float,
    shell_m: float = C.SHELL_THICKNESS_M,
) -> BurrowPhysics:
    """Compute thermal properties from foam-cast photogrammetry.

    The photogrammetry scans measure the **foam cast** of the cavity, so
    ``volume_cm3`` is the air space and ``surface_area_cm2`` is the inner
    wall area.  The stone shell of thickness ``shell_m`` surrounds this
    cavity.

    Args:
        volume_cm3:       Cavity volume from foam cast in cm³.
        surface_area_cm2: Inner surface area from foam cast in cm².
        shell_m:          Stone shell thickness in metres.

    Returns:
        Populated :class:`BurrowPhysics` instance.
    """
    V_cav = volume_cm3 * 1e-6
    SA = surface_area_cm2 * 1e-4

    V_shell = SA * shell_m
    M_stone = V_shell * C.RHO_STONE

    C_stone = M_stone * C.C_STONE
    C_air = C.RHO_AIR * V_cav * C.C_AIR
    C_eff = C_stone + C_air

    return BurrowPhysics(
        C_eff=C_eff,
        C_stone=C_stone,
        C_air=C_air,
        V_cavity_m3=V_cav,
        SA_m2=SA,
        M_stone_kg=M_stone,
        V_shell_m3=V_shell,
        shell_m=shell_m,
    )


# ┌─────────────────────────────────────────────────────────────────────┐
# │  THERMAL PENETRATION                    « how deep does it go? »   │
# └─────────────────────────────────────────────────────────────────────┘
def thermal_penetration_depth(time_s: float) -> float:
    """Compute thermal penetration depth into stone.

    Uses the standard diffusion scaling δ = √(α·t) where α is the
    thermal diffusivity of stone.

    Args:
        time_s: Time horizon in seconds.

    Returns:
        Penetration depth in metres.
    """
    alpha = C.K_THERMAL_STONE / (C.RHO_STONE * C.C_STONE)
    return np.sqrt(alpha * time_s)


# ┌─────────────────────────────────────────────────────────────────────┐
# │  ODE INTEGRATION                   « forward Euler, old reliable » │
# └─────────────────────────────────────────────────────────────────────┘
def simulate_burrow_temperature(
    k: float,
    T_out_series: np.ndarray,
    T_in_initial: float,
    Q_norm: float = 0.0,
    dt: float = 1.0,
) -> np.ndarray:
    """Simulate burrow interior temperature using forward Euler.

    Integrates::

        T_in[i+1] = T_in[i] + k·(T_out[i] − T_in[i])·dt + Q_norm·dt

    Args:
        k:             Thermal rate constant [1/hour].
        T_out_series:  Outside temperature time series [°C].
        T_in_initial:  Initial inside temperature [°C].
        Q_norm:        Normalised heat input Q_weta/C_eff [°C/hour].
        dt:            Timestep in hours.

    Returns:
        Array of simulated inside temperatures, same length as
        ``T_out_series``.
    """
    n = len(T_out_series)
    T_in = np.empty(n)
    T_in[0] = T_in_initial

    for i in range(1, n):
        dT = k * (T_out_series[i - 1] - T_in[i - 1]) * dt + Q_norm * dt
        T_in[i] = T_in[i - 1] + dT
        if not np.isfinite(T_in[i]):
            T_in[i:] = np.nan
            break

    return T_in


def simulate_24h_steady_state(
    k: float,
    T_out_24h: np.ndarray,
    Q_norm: float = 0.0,
    n_cycles: int = C.N_CYCLES,
    substeps: int = C.SUBSTEPS,
) -> np.ndarray:
    """Run the 24-h cycle to steady-state oscillation.

    Uses sub-hour Euler steps for accuracy, then samples at hourly
    resolution.

    Args:
        k:         Thermal rate constant [1/hour].
        T_out_24h: Mean outside temperature for each hour (length 24).
        Q_norm:    Normalised heat input [°C/hour].
        n_cycles:  Warm-up cycles before recording.
        substeps:  Euler sub-steps per hour.

    Returns:
        Steady-state inside temperature array of length 24.
    """
    dt = 1.0 / substeps
    hours_fine = np.linspace(0, 24, 24 * substeps, endpoint=False)
    T_out_fine = np.interp(hours_fine, np.arange(24), T_out_24h, period=24)

    T = np.mean(T_out_24h)
    n_fine = len(T_out_fine)

    # ── warm-up ──
    for _ in range(n_cycles):
        for i in range(n_fine):
            T = T + k * (T_out_fine[i] - T) * dt + Q_norm * dt
        if not np.isfinite(T):
            return np.full(24, np.nan)

    # ── record final cycle at hourly resolution ──
    T_hourly = np.empty(24)
    for h in range(24):
        T_hourly[h] = T
        for s in range(substeps):
            idx = h * substeps + s
            T = T + k * (T_out_fine[idx] - T) * dt + Q_norm * dt

    return T_hourly


# ┌─────────────────────────────────────────────────────────────────────┐
# │  DIAGNOSTICS                        « phase lag & amplitude »      │
# └─────────────────────────────────────────────────────────────────────┘
def compute_phase_lag(T_in: np.ndarray, T_out: np.ndarray) -> tuple[int, float]:
    """Compute phase lag via circular cross-correlation.

    Args:
        T_in:  Inside temperature array (length 24).
        T_out: Outside temperature array (length 24).

    Returns:
        Tuple of (lag_hours, max_correlation).  Positive lag means
        T_in lags behind T_out.
    """
    T_in_n = T_in - np.mean(T_in)
    T_out_n = T_out - np.mean(T_out)

    corrs = np.array(
        [np.corrcoef(T_in_n, np.roll(T_out_n, lag))[0, 1] for lag in range(24)]
    )
    best = int(np.argmax(corrs))
    if best > 12:
        best -= 24
    return best, float(np.max(corrs))


def compute_amplitude_ratio(T_in: np.ndarray, T_out: np.ndarray) -> float:
    """Ratio of diurnal temperature amplitudes (inside / outside).

    Args:
        T_in:  Inside temperature array.
        T_out: Outside temperature array.

    Returns:
        Amplitude ratio.  Values < 1 indicate damping by thermal mass.
    """
    amp_in = (np.max(T_in) - np.min(T_in)) / 2.0
    amp_out = (np.max(T_out) - np.min(T_out)) / 2.0
    return amp_in / amp_out if amp_out > 0 else np.nan
