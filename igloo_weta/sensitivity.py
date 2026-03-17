#!/usr/bin/env python3
"""
 ┌─────────────────────────────────────────────────────────────────────┐
 │  SENSITIVITY                       « wiggling the knobs »           │
 └─────────────────────────────────────────────────────────────────────┘

Shell thickness sensitivity analysis and allometric metabolic rate
estimation for wētā species.

Key insight: the fitted parameters *k* and *q* (in °C/h) are determined
entirely from the temperature dynamics and do **not** depend on shell
thickness.  Only the conversion to physical units (Watts, mW/K) scales
linearly with the assumed shell thickness.  The crossover temperature
is also thickness-independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import constants as C
from .fitting import RockResult


# ┌─────────────────────────────────────────────────────────────────────┐
# │  ALLOMETRIC METABOLIC RATE             « scaling the fire within »  │
# └─────────────────────────────────────────────────────────────────────┘
def rmr_at_temperature(mass_g: float, temp_c: float) -> float:
    """Estimate resting metabolic rate at a given temperature.

    Uses standard insect allometric scaling::

        RMR_25 = a · M^b    (mW, grams)

    corrected to ``temp_c`` via Q₁₀::

        RMR_T = RMR_25 / Q10^((25 − T) / 10)

    Args:
        mass_g: Body mass in grams.
        temp_c: Ambient temperature in °C.

    Returns:
        Estimated RMR in milliwatts.
    """
    rmr_25 = C.ALLOMETRIC_A * mass_g ** C.ALLOMETRIC_B
    correction = C.Q10 ** ((C.T_REF_C - temp_c) / 10.0)
    return rmr_25 / correction


@dataclass
class SpeciesRMR:
    """Metabolic rate estimates for one wētā species.

    Attributes:
        species:   Species name.
        mass_g:    Mean body mass (g).
        rmr_5:     RMR at 5 °C (mW).
        rmr_10:    RMR at 10 °C (mW).
        rmr_15:    RMR at 15 °C (mW).
        rmr_25:    RMR at 25 °C (mW).
        color:     Default plot colour.
    """

    species: str
    mass_g: float
    rmr_5: float
    rmr_10: float
    rmr_15: float
    rmr_25: float
    color: str


def compute_species_rmr(species_stats: dict) -> dict[str, SpeciesRMR]:
    """Compute RMR estimates for each wētā species.

    Args:
        species_stats: Output of :func:`~igloo_weta.ingest.summarise_species`.

    Returns:
        Dict keyed by species name, values are :class:`SpeciesRMR`.
    """
    colors = {
        "H. maori": "#d62728",
        "H. thoracica": "#ff7f0e",
        "H. crassidens": "#2ca02c",
    }
    out = {}
    for sp, s in species_stats.items():
        m = s["mean"]
        out[sp] = SpeciesRMR(
            species=sp,
            mass_g=m,
            rmr_5=rmr_at_temperature(m, 5.0),
            rmr_10=rmr_at_temperature(m, 10.0),
            rmr_15=rmr_at_temperature(m, 15.0),
            rmr_25=rmr_at_temperature(m, 25.0),
            color=colors.get(sp, "#888888"),
        )
    return out


# ┌─────────────────────────────────────────────────────────────────────┐
# │  SHELL THICKNESS SWEEP               « how thick is the igloo? »    │
# └─────────────────────────────────────────────────────────────────────┘
@dataclass
class ThicknessPoint:
    """Q and U estimates at one shell thickness for one rock.

    Attributes:
        thickness_cm: Shell thickness in cm.
        C_eff:        Effective heat capacity (J/K).
        U_mW_K:       Thermal conductance (mW/K).
        Q_mW:         Metabolic heat estimate (mW).
        M_stone_g:    Stone shell mass (g).
    """

    thickness_cm: float
    C_eff: float
    U_mW_K: float
    Q_mW: float
    M_stone_g: float


def sweep_shell_thickness(
    result: RockResult,
    thicknesses_cm: Optional[list[float]] = None,
) -> list[ThicknessPoint]:
    """Compute Q_weta and U at multiple shell thicknesses for one rock.

    Because *k* and *q* are thickness-independent, ``Q = q · C_eff / 3600``
    scales linearly.  No re-fitting is needed.

    Args:
        result:          :class:`RockResult` from :func:`~fitting.fit_single_rock`.
        thicknesses_cm:  List of thicknesses to evaluate (default from constants).

    Returns:
        List of :class:`ThicknessPoint`, one per thickness.
    """
    if thicknesses_cm is None:
        thicknesses_cm = C.SHELL_THICKNESSES_CM

    if result.phys is None:
        return []

    SA_m2 = result.phys.SA_m2
    V_cav = result.phys.V_cavity_m3

    points = []
    for t_cm in thicknesses_cm:
        t_m = t_cm / 100.0
        V_shell = SA_m2 * t_m
        M_stone = V_shell * C.RHO_STONE
        C_stone = M_stone * C.C_STONE
        C_air = C.RHO_AIR * V_cav * C.C_AIR
        C_eff = C_stone + C_air

        U = result.k_fit * C_eff / 3600.0
        Q = result.q_fit * C_eff / 3600.0

        points.append(
            ThicknessPoint(
                thickness_cm=t_cm,
                C_eff=C_eff,
                U_mW_K=U * 1000.0,
                Q_mW=Q * 1000.0,
                M_stone_g=M_stone * 1000.0,
            )
        )
    return points


def sweep_all_rocks(
    results: list[RockResult],
    thicknesses_cm: Optional[list[float]] = None,
) -> dict[int, list[ThicknessPoint]]:
    """Run shell thickness sweep for every rock with geometry.

    Args:
        results:         List of :class:`RockResult`.
        thicknesses_cm:  Thicknesses to evaluate.

    Returns:
        Dict keyed by rock ID, values are lists of :class:`ThicknessPoint`.
    """
    out = {}
    for r in results:
        pts = sweep_shell_thickness(r, thicknesses_cm)
        if pts:
            out[r.rock] = pts
    return out
