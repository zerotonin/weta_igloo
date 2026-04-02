#!/usr/bin/env python3
"""
 ┌─────────────────────────────────────────────────────────────────────┐
 │  CONSTANTS                             « the numbers that matter »  │
 └─────────────────────────────────────────────────────────────────────┘

Every assumption, physical constant, material property, species
measurement, and default configuration in one file.  When a reviewer
asks "where did that number come from?" — point them here.

All values carry their source and plausible range so sensitivity
analyses can sweep them systematically.

Import what you need::

    from igloo_weta.constants import STONE, WOOD, SPECIES, SIM

Or grab a specific value::

    from igloo_weta.constants import STONE
    print(STONE.rho)          # 2650.0 kg/m³
    print(STONE.rho_range)    # (2500.0, 2800.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  MATERIAL PROPERTIES                   « what stuff is made of »    ║
# ╚═══════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class MaterialProperties:
    """Thermophysical properties of a burrow wall material.

    Attributes:
        name:           Human-readable material name.
        rho:            Density [kg/m³].
        rho_range:      Plausible density range [kg/m³].
        c:              Specific heat capacity [J/(kg·K)].
        c_range:        Plausible specific heat range [J/(kg·K)].
        k_thermal:      Thermal conductivity [W/(m·K)].
        k_thermal_range: Plausible conductivity range [W/(m·K)].
        source:         Literature reference for these values.
    """

    name: str
    rho: float
    rho_range: tuple[float, float]
    c: float
    c_range: tuple[float, float]
    k_thermal: float
    k_thermal_range: tuple[float, float]
    source: str

    @property
    def alpha(self) -> float:
        """Thermal diffusivity α = k / (ρ·c) in m²/s."""
        return self.k_thermal / (self.rho * self.c)


STONE = MaterialProperties(
    name="Greywacke / schist (Otago)",
    rho=2650.0,
    rho_range=(2500.0, 2800.0),
    c=840.0,
    c_range=(780.0, 900.0),
    k_thermal=2.5,
    k_thermal_range=(1.5, 3.5),
    source=(
        "Greywacke typical values. Robertson (1988) Engineering Geology "
        "of the NZ Greywacke; Clauser & Huenges (1995) Thermal "
        "conductivity of rocks and minerals, AGU Ref Shelf 3."
    ),
)
"""Properties of Otago greywacke/schist stone for rock burrows."""

WOOD = MaterialProperties(
    name="Native hardwood (mahoe / ngaio / manuka)",
    rho=550.0,
    rho_range=(400.0, 700.0),
    c=1700.0,
    c_range=(1400.0, 2000.0),
    k_thermal=0.15,
    k_thermal_range=(0.10, 0.25),
    source=(
        "Wood handbook (Forest Products Lab, USDA 2010). NZ native "
        "hardwoods: mahoe (Melicytus ramiflorus) ~500 kg/m³, ngaio "
        "(Myoporum laetum) ~550 kg/m³, manuka (Leptospermum) ~600 kg/m³. "
        "Thermal conductivity across grain 0.10–0.25 W/(m·K)."
    ),
)
"""Properties of native NZ hardwood for tree gallery burrows."""


# ┌─────────────────────────────────────────────────────────────────────┐
# │  AIR                                 « the stuff in the burrow »   │
# └─────────────────────────────────────────────────────────────────────┘

@dataclass(frozen=True)
class AirProperties:
    """Thermophysical properties of air at field-relevant temperatures.

    Attributes:
        rho:    Density at ~15 °C [kg/m³].
        c:      Specific heat at constant pressure [J/(kg·K)].
        source: Reference.
    """

    rho: float
    c: float
    source: str


AIR = AirProperties(
    rho=1.225,
    c=1005.0,
    source="Standard atmosphere at 15 °C, 1 atm.",
)
"""Air properties at typical field temperature."""


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  SPECIES MORPHOMETRICS              « the animals themselves »      ║
# ╚═══════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class SpeciesInfo:
    """Morphometric and ecological data for one wētā species.

    Attributes:
        name:             Binomial species name.
        common_name:      Common name.
        body_length_mm:   Typical adult body length [mm].
        body_length_range: Range of body lengths [mm].
        body_width_mm:    Typical adult body width [mm].
        body_width_range: Range of body widths [mm].
        mass_g:           Mean adult body mass [g] (from our data).
        mass_range_g:     Range of masses in our dataset [g].
        mass_n:           Sample size for mass measurements.
        burrow_type:      ``"stone"`` or ``"wood"``.
        gallery_entrance_mm: Gallery entrance diameter [mm].
        gallery_entrance_range: Entrance diameter range [mm].
        gallery_volume_cm3: Estimated single-occupant gallery volume [cm³].
        gallery_volume_range: Volume range [cm³].
        gallery_SA_cm2:   Estimated inner surface area [cm²].
        gallery_SA_range: Surface area range [cm²].
        gallery_wall_mm:  Estimated wall thickness [mm].
        gallery_wall_range: Wall thickness range [mm].
        gallery_source:   Reference for gallery dimensions.
        wall_material:    Reference to material: ``STONE`` or ``WOOD``.
        field_site:       Where the temperature data were collected.
        notes:            Additional ecological notes.
    """

    name: str
    common_name: str
    # ── body dimensions ──────────────────────────────────────────────
    body_length_mm: float
    body_length_range: tuple[float, float]
    body_width_mm: float
    body_width_range: tuple[float, float]
    mass_g: float
    mass_range_g: tuple[float, float]
    mass_n: int
    # ── burrow / gallery ─────────────────────────────────────────────
    burrow_type: str
    gallery_entrance_mm: float
    gallery_entrance_range: tuple[float, float]
    gallery_volume_cm3: float
    gallery_volume_range: tuple[float, float]
    gallery_SA_cm2: float
    gallery_SA_range: tuple[float, float]
    gallery_wall_mm: float
    gallery_wall_range: tuple[float, float]
    gallery_source: str
    wall_material: str  # "stone" or "wood"
    # ── field context ────────────────────────────────────────────────
    field_site: str
    notes: str


H_MAORI = SpeciesInfo(
    name="Hemideina maori",
    common_name="Mountain stone wētā",
    body_length_mm=45.0,
    body_length_range=(30.0, 55.0),
    body_width_mm=14.0,
    body_width_range=(10.0, 18.0),
    mass_g=6.27,
    mass_range_g=(3.09, 8.56),
    mass_n=17,
    burrow_type="stone",
    gallery_entrance_mm=20.0,
    gallery_entrance_range=(12.0, 30.0),
    gallery_volume_cm3=500.0,
    gallery_volume_range=(188.0, 3071.0),
    gallery_SA_cm2=1200.0,
    gallery_SA_range=(663.0, 2664.0),
    gallery_wall_mm=10.0,
    gallery_wall_range=(5.0, 30.0),
    gallery_source=(
        "Photogrammetric foam casts of Rock and Pillars stone burrows "
        "(this study). Volume and SA are the cavity dimensions. Wall "
        "thickness estimated as 1 cm stone shingle."
    ),
    wall_material="stone",
    field_site="Rock and Pillars, Otago, NZ (~1428 m asl)",
    notes=(
        "Lives above the treeline in stone shingle crevices and "
        "cavities. Abandoned arboreal life millions of years ago. "
        "Rock burrows range from small single-animal crevices to "
        "large multi-chamber shingle piles."
    ),
)
"""*H. maori* — the alpine species with stone burrows (this study)."""

H_THORACICA = SpeciesInfo(
    name="Hemideina thoracica",
    common_name="Auckland tree wētā",
    body_length_mm=40.0,
    body_length_range=(30.0, 45.0),
    body_width_mm=12.0,
    body_width_range=(9.0, 15.0),
    mass_g=3.32,
    mass_range_g=(2.16, 4.50),
    mass_n=9,
    burrow_type="wood",
    gallery_entrance_mm=15.0,
    gallery_entrance_range=(12.0, 18.0),
    gallery_volume_cm3=45.0,
    gallery_volume_range=(35.0, 55.0),
    gallery_SA_cm2=75.0,
    gallery_SA_range=(60.0, 90.0),
    gallery_wall_mm=12.0,
    gallery_wall_range=(10.0, 20.0),
    gallery_source=(
        "Measured from custom wētā motels used in this study (photo "
        "documentation with ruler). Two-chamber Forstner-routed design: "
        "cavity ~75×30×20 mm total (~45 cm³), wall ~10–12 mm sides, "
        "~15 mm top/bottom. Back board ~15–20 mm. Entrance hole ~15 mm Ø. "
        "Mounted on native trees in bush. iButton DS1921/DS1922 loggers "
        "placed inside cavity (T_in) and on outer bark (T_out)."
    ),
    wall_material="wood",
    field_site="North Island, NZ (lowland–mid-elevation forest)",
    notes=(
        "Data collected from custom wētā motels, NOT natural galleries. "
        "Motel design: two-chamber cavity routed with Forstner bit, "
        "perspex viewing window, mounted on tree trunk. Readily occupied "
        "by wētā within weeks of deployment. Polygynandrous with harem "
        "formation. Natural galleries in trees bored by puriri moth "
        "(Aenetus virescens) and kanuka beetle (Ochrocydus huttoni) "
        "larvae would have different thermal properties."
    ),
)
"""*H. thoracica* — lowland tree galleries, North Island."""

H_CRASSIDENS = SpeciesInfo(
    name="Hemideina crassidens",
    common_name="Wellington tree wētā",
    body_length_mm=65.0,
    body_length_range=(50.0, 75.0),
    body_width_mm=15.0,
    body_width_range=(12.0, 18.0),
    mass_g=4.89,
    mass_range_g=(3.85, 6.13),
    mass_n=8,
    burrow_type="wood",
    gallery_entrance_mm=15.0,
    gallery_entrance_range=(12.0, 18.0),
    gallery_volume_cm3=45.0,
    gallery_volume_range=(35.0, 55.0),
    gallery_SA_cm2=75.0,
    gallery_SA_range=(60.0, 90.0),
    gallery_wall_mm=12.0,
    gallery_wall_range=(10.0, 20.0),
    gallery_source=(
        "Measured from custom wētā motels used in this study (photo "
        "documentation with ruler). Same motel design as H. thoracica: "
        "two-chamber Forstner-routed cavity ~75×30×20 mm (~45 cm³), "
        "wall ~10–12 mm sides, ~15 mm top/bottom. Entrance ~15 mm Ø. "
        "iButton DS1921/DS1922 paired loggers: inside cavity + outer bark. "
        "Literature natural gallery reference: Kelly (2008) artificial "
        "galleries 53.8 cm³ for ~3 adults on Maud Island; entrance "
        "~20 mm (Moller 1985)."
    ),
    wall_material="wood",
    field_site="Southern North Island / NW South Island, NZ",
    notes=(
        "Data collected from custom wētā motels (same design as "
        "H. thoracica motels). Largest tree wētā. Polygynous — males "
        "guard females in galleries. Can also use rock refuges where "
        "trees unavailable. Gallery entrance kept clean; entered "
        "head-first, exited in reverse with tibial spines outward."
    ),
)
"""*H. crassidens* — larger tree galleries, southern NI / NW SI."""

SPECIES: dict[str, SpeciesInfo] = {
    "H. maori": H_MAORI,
    "H. thoracica": H_THORACICA,
    "H. crassidens": H_CRASSIDENS,
}
"""All species in a lookup dict keyed by binomial name."""


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  ALLOMETRIC METABOLIC SCALING       « the fire within »             ║
# ╚═══════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class AllometryParams:
    """Parameters for insect resting metabolic rate allometry.

    The standard model::

        RMR_25 = a · M^b   (mW, grams)
        RMR_T  = RMR_25 / Q10^((25 − T) / 10)

    Attributes:
        a:        Pre-factor [mW / g^b].
        b:        Mass scaling exponent.
        Q10:      Temperature coefficient.
        T_ref_C:  Reference temperature [°C].
        a_range:  Plausible range for a.
        b_range:  Plausible range for b.
        Q10_range: Plausible range for Q10.
        source:   Literature reference.
    """

    a: float
    b: float
    Q10: float
    T_ref_C: float
    a_range: tuple[float, float]
    b_range: tuple[float, float]
    Q10_range: tuple[float, float]
    source: str


ALLOMETRY = AllometryParams(
    a=10.5,
    b=0.75,
    Q10=2.5,
    T_ref_C=25.0,
    a_range=(8.0, 14.0),
    b_range=(0.67, 0.82),
    Q10_range=(2.0, 3.0),
    source=(
        "Lighton (2008) Measuring Metabolic Rates, OUP. Insect RMR "
        "compilation. Q10 range from Chown & Nicolson (2004) Insect "
        "Physiological Ecology, OUP. Exponent 0.75: Kleiber's law; "
        "range 0.67–0.82 from Glazier (2005) Biol Rev 80:611."
    ),
)
"""Allometric metabolic rate parameters with literature ranges."""


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  SIMULATION DEFAULTS                « knobs for the model »         ║
# ╚═══════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class SimulationConfig:
    """Default parameters for the ODE integration and fitting.

    Attributes:
        n_cycles:            Warm-up cycles before steady state.
        substeps:            Euler sub-steps per hour.
        k_fit_bounds:        Bounds on k during optimisation (1/h).
        shell_thickness_m:   Default stone shell thickness [m].
        shell_sweep_cm:      Thicknesses for sensitivity sweep [cm].
        wood_wall_sweep_cm:  Wall thicknesses for wood gallery sweep [cm].
    """

    n_cycles: int
    substeps: int
    k_fit_bounds: tuple[float, float]
    shell_thickness_m: float
    shell_sweep_cm: tuple[float, ...]
    wood_wall_sweep_cm: tuple[float, ...]


SIM = SimulationConfig(
    n_cycles=25,
    substeps=20,
    k_fit_bounds=(0.005, 10.0),
    shell_thickness_m=0.01,
    shell_sweep_cm=(0.5, 1.0, 1.5, 2.0, 3.0, 5.0),
    wood_wall_sweep_cm=(1.0, 1.5, 2.0, 2.5, 3.0, 5.0),
)
"""Default simulation and fitting configuration."""


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  VISUALISATION                      « making it pretty »            ║
# ╚═══════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class VizConfig:
    """Plot styling defaults.

    Attributes:
        dpi:              Raster resolution for PNG.
        svg_font_family:  Font family for editable SVG text.
        species_colors:   Colour per species for consistent plots.
    """

    dpi: int
    svg_font_family: str
    species_colors: dict[str, str]


VIZ = VizConfig(
    dpi=200,
    svg_font_family="sans-serif",
    species_colors={
        "H. maori": "#d62728",
        "H. thoracica": "#ff7f0e",
        "H. crassidens": "#2ca02c",
    },
)
"""Visualisation defaults."""


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  EXCLUSIONS & FLAGS                 « the fine print »              ║
# ╚═══════════════════════════════════════════════════════════════════════╝

EXCLUDE_ROCK_IDS: list[int] = [22]
"""Rock IDs excluded from species-level pooling.

Rock 22 is excluded because its humidity was at saturation (~100%),
indicating potential evaporative cooling artefacts that confound the
thermal model.
"""


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  BACKWARD COMPATIBILITY             « don't break old imports »     ║
# ╚═══════════════════════════════════════════════════════════════════════╝
# These module-level aliases keep existing code working while we
# migrate to the structured dataclasses above.

RHO_STONE: float = STONE.rho
C_STONE: float = STONE.c
K_THERMAL_STONE: float = STONE.k_thermal
RHO_AIR: float = AIR.rho
C_AIR: float = AIR.c
SHELL_THICKNESS_M: float = SIM.shell_thickness_m
SHELL_THICKNESSES_CM: list = list(SIM.shell_sweep_cm)
N_CYCLES: int = SIM.n_cycles
SUBSTEPS: int = SIM.substeps
K_FIT_BOUNDS: tuple = SIM.k_fit_bounds
ALLOMETRIC_A: float = ALLOMETRY.a
ALLOMETRIC_B: float = ALLOMETRY.b
Q10: float = ALLOMETRY.Q10
T_REF_C: float = ALLOMETRY.T_ref_C
FIG_DPI: int = VIZ.dpi
SVG_FONT_FAMILY: str = VIZ.svg_font_family
