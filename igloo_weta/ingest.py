#!/usr/bin/env python3
"""
 ┌─────────────────────────────────────────────────────────────────────┐
 │  INGEST                                « loading the raw signal »   │
 └─────────────────────────────────────────────────────────────────────┘

Data loading, validation, and cleaning for field temperature recordings,
photogrammetric burrow geometry, and wētā morphometrics.

All loaders return pandas DataFrames with consistent column names.
Paths default to the ``data/`` directory adjacent to the package root.

Example::

    from igloo_weta.ingest import load_all
    ds = load_all("/path/to/data")
    print(ds.hourly_24h.columns)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ┌─────────────────────────────────────────────────────────────────────┐
# │  DATA BUNDLE                          « one object to rule them »   │
# └─────────────────────────────────────────────────────────────────────┘
@dataclass
class DataBundle:
    """Container for all experimental datasets.

    Attributes:
        hourly_24h:  Mean 24-h diurnal cycle per rock (hourly bins).
        incubator:   Full-duration hourly aggregates from incubator control.
        overall:     Per-rock summary statistics across all days.
        daily:       Day-by-day averages per rock.
        rock_phys:   Photogrammetric cavity geometry (foam cast measurements).
        weta_morph:  Wētā morphometrics and species assignments.
    """

    hourly_24h: pd.DataFrame
    incubator: pd.DataFrame
    overall: pd.DataFrame
    daily: pd.DataFrame
    rock_phys: pd.DataFrame
    weta_morph: pd.DataFrame


# ┌─────────────────────────────────────────────────────────────────────┐
# │  PATH RESOLUTION                       « finding the goods »        │
# └─────────────────────────────────────────────────────────────────────┘
def _default_data_dir() -> Path:
    """Return ``<package_root>/../data`` as the default data directory."""
    return Path(__file__).resolve().parent.parent / "data"


def _resolve(data_dir: Optional[str], filename: str) -> Path:
    """Build and validate a file path inside the data directory.

    Args:
        data_dir: Override data directory.  ``None`` → default.
        filename: Filename to look up.

    Returns:
        Resolved absolute ``Path``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    d = Path(data_dir) if data_dir else _default_data_dir()
    p = d / filename
    if not p.is_file():
        raise FileNotFoundError(f"Expected data file not found: {p}")
    return p


# ┌─────────────────────────────────────────────────────────────────────┐
# │  LOADERS                             « reading the bit stream »     │
# └─────────────────────────────────────────────────────────────────────┘
def load_hourly_24h(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Load the 24-hour hourly average temperatures per rock.

    Args:
        data_dir: Path to data directory.  ``None`` uses the default.

    Returns:
        DataFrame with columns including ``Hour``, ``rock``,
        ``inside_mean``, ``outside_mean``, ``diff_mean``, etc.
    """
    p = _resolve(data_dir, "24h_hourly_averages.csv")
    return pd.read_csv(p)


def load_incubator(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Load the incubator passive-control hourly time series.

    Args:
        data_dir: Path to data directory.

    Returns:
        DataFrame indexed by ``elapsed_hour`` with inside/outside
        temperature means, SEMs, and confidence intervals.
    """
    p = _resolve(data_dir, "full_duration_hourly_aggregates.csv")
    return pd.read_csv(p)


def load_overall(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Load per-rock overall summary statistics.

    Args:
        data_dir: Path to data directory.

    Returns:
        DataFrame with one row per rock: mean temperatures,
        confidence intervals, humidity.
    """
    p = _resolve(data_dir, "full_duration_overall_stats.csv")
    return pd.read_csv(p)


def load_daily(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Load day-by-day temperature averages per rock.

    Args:
        data_dir: Path to data directory.

    Returns:
        DataFrame with ``day``, ``rock``, and temperature columns.
    """
    p = _resolve(data_dir, "total_duration_averages.csv")
    return pd.read_csv(p)


def load_rock_physics(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Load photogrammetric burrow geometry from foam casts.

    The ``Total Volume`` and ``Total Surface area`` columns describe the
    **air cavity** (foam imprint), not the stone.  Stone shell properties
    are computed downstream in :mod:`igloo_weta.physics`.

    Args:
        data_dir: Path to data directory.

    Returns:
        DataFrame with ``Rock number``, ``Total Volume (cm3)``,
        ``Total Surface area (cm2)``, and chunk data.
    """
    p = _resolve(data_dir, "Rock_data.xlsx")
    df = pd.read_excel(p, sheet_name="Sheet1")
    return df


def load_weta_morphometrics(
    data_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Load wētā body measurements and assign species from ID prefixes.

    Species are inferred from the ``Weta number`` prefix:
    ``HM`` → *H. maori*, ``Hthora`` → *H. thoracica*,
    ``Hcrass`` → *H. crassidens*.

    Args:
        data_dir: Path to data directory.

    Returns:
        DataFrame with added ``species`` column.
    """
    p = _resolve(data_dir, "Weta_thermoregulation_datasheet.xlsx")
    df = pd.read_excel(p, sheet_name="Sheet1")

    def _assign(row: pd.Series) -> str:
        wn = str(row["Weta number"]).lower()
        if wn.startswith("hm"):
            return "H. maori"
        elif wn.startswith("hthora"):
            return "H. thoracica"
        elif wn.startswith("hcrass"):
            return "H. crassidens"
        return "unknown"

    df["species"] = df.apply(_assign, axis=1)
    return df


def summarise_species(morph_df: pd.DataFrame) -> dict:
    """Compute per-species weight statistics from morphometric data.

    Args:
        morph_df: Output of :func:`load_weta_morphometrics`.

    Returns:
        Dict keyed by species name, each containing ``n``, ``mean``,
        ``std``, ``min``, ``max``, ``median``, and ``weights`` (array).
    """
    stats = {}
    for sp in ["H. maori", "H. thoracica", "H. crassidens"]:
        w = morph_df.loc[morph_df["species"] == sp, "Weight (g)"].dropna()
        if len(w) == 0:
            warnings.warn(f"No weight data for {sp}")
            continue
        stats[sp] = {
            "n": len(w),
            "mean": float(w.mean()),
            "std": float(w.std()),
            "min": float(w.min()),
            "max": float(w.max()),
            "median": float(w.median()),
            "weights": w.values.copy(),
        }
    return stats


# ┌─────────────────────────────────────────────────────────────────────┐
# │  BUNDLE LOADER                           « the whole enchilada »    │
# └─────────────────────────────────────────────────────────────────────┘
def load_all(data_dir: Optional[str] = None) -> DataBundle:
    """Load every dataset and return a single :class:`DataBundle`.

    Args:
        data_dir: Override for the data directory path.

    Returns:
        Populated :class:`DataBundle` ready for analysis.
    """
    return DataBundle(
        hourly_24h=load_hourly_24h(data_dir),
        incubator=load_incubator(data_dir),
        overall=load_overall(data_dir),
        daily=load_daily(data_dir),
        rock_phys=load_rock_physics(data_dir),
        weta_morph=load_weta_morphometrics(data_dir),
    )
