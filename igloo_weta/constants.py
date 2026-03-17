#!/usr/bin/env python3
"""
 ┌─────────────────────────────────────────────────────────────────────┐
 │  CONSTANTS                             « the numbers that matter »  │
 └─────────────────────────────────────────────────────────────────────┘

Physical constants, material properties, and default configuration for
the wētā burrow thermal model.  Import what you need::

    from igloo_weta.constants import RHO_STONE, C_STONE, SHELL_THICKNESS_M
"""

# ── thermodynamic properties ─────────────────────────────────────────
RHO_STONE: float = 2650.0
"""Density of greywacke/schist stone in kg/m³."""

C_STONE: float = 840.0
"""Specific heat capacity of stone in J/(kg·K)."""

RHO_AIR: float = 1.225
"""Density of air at ~15 °C in kg/m³."""

C_AIR: float = 1005.0
"""Specific heat capacity of air at constant pressure in J/(kg·K)."""

K_THERMAL_STONE: float = 2.5
"""Thermal conductivity of stone in W/(m·K).  Used for penetration depth."""

# ── default burrow geometry ──────────────────────────────────────────
SHELL_THICKNESS_M: float = 0.01
"""Default stone shell thickness in metres (1 cm)."""

SHELL_THICKNESSES_CM: list = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
"""Sweep values for sensitivity analysis in centimetres."""

# ── simulation defaults ──────────────────────────────────────────────
N_CYCLES: int = 25
"""Number of 24-h warm-up cycles before recording the steady state."""

SUBSTEPS: int = 20
"""Euler sub-steps per hour for integration accuracy."""

K_FIT_BOUNDS: tuple = (0.005, 10.0)
"""Bounds on the thermal rate constant *k* during optimisation (1/h)."""

# ── allometric metabolic scaling ─────────────────────────────────────
ALLOMETRIC_A: float = 10.5
"""Pre-factor for insect RMR allometry: RMR₂₅ = a · M^b  (mW, g)."""

ALLOMETRIC_B: float = 0.75
"""Mass exponent for insect RMR allometry."""

Q10: float = 2.5
"""Temperature coefficient for metabolic rate correction."""

T_REF_C: float = 25.0
"""Reference temperature for allometric RMR (°C)."""

# ── visualisation ────────────────────────────────────────────────────
FIG_DPI: int = 200
"""Default raster resolution for PNG export."""

SVG_FONT_FAMILY: str = "sans-serif"
"""Font family embedded in SVG output (editable in Inkscape/Illustrator)."""

# ── rock 22 exclusion flag ───────────────────────────────────────────
EXCLUDE_ROCK_IDS: list = [22]
"""Rock IDs excluded from species-level pooling (saturated humidity)."""
