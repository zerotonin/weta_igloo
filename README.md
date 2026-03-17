# 🏔️ igloo_weta

```
 ╔═══════════════════════════════════════════════════════════════╗
 ║  IGLOO WĒTĀ — Thermal model for burrow heat exchange         ║
 ║  « model. fit. predict. science. »                           ║
 ╚═══════════════════════════════════════════════════════════════╝
```

A lumped-parameter heat conduction model for stone-shingle wētā burrows,
extending the [IGLOO framework](https://github.com/zerotonin/igloo)
(Giraldo et al. 2019, *Sci Rep* 9:3974) to estimate active
thermoregulation in ectothermic insects.

## What it does

Alpine wētā (*Hemideina maori*) shelter in stone-shingle burrows at
~1400 m altitude on New Zealand's Rock and Pillars range.  Temperature
loggers inside and outside these burrows reveal that burrow interiors are
systematically warmer than a passive heat-exchange model predicts.

This package:

1. **Fits a null model** of passive thermal lag from burrow geometry
   (photogrammetric foam casts → cavity volume + surface area → stone
   shell thermal mass).
2. **Tests for active heating** by fitting an additional metabolic heat
   term and evaluating significance via F-test.
3. **Computes the thermoregulatory crossover** — the outside temperature
   at which wētā switch from heating to cooling their burrows (~10.6 °C).
4. **Runs sensitivity analysis** on stone shell thickness, with species-
   specific metabolic rate validation from allometric scaling.

## The model

The governing ODE:

```
C_eff · dT_in/dt  =  U · (T_out(t) − T_in(t))  +  Q_weta
```

where:
- **C_eff** = effective heat capacity of burrow (stone shell + air)
- **U** = overall thermal conductance (W/K)
- **Q_weta** = metabolic heat production (W); zero in the null model

Defining `k = U / C_eff`:

```
dT_in/dt  =  k · (T_out − T_in)  +  Q_weta / C_eff
```

The null model sets `Q_weta = 0`.  Any systematic positive residual
(T_in_observed > T_in_null) implies active heating.

## Installation

```bash
git clone https://github.com/zerotonin/igloo_weta.git
cd igloo_weta
pip install -e .
```

For documentation building:
```bash
pip install -e ".[docs]"
```

### Dependencies

Only standard scientific Python — no exotic deps:

- numpy ≥ 1.24
- pandas ≥ 2.0
- scipy ≥ 1.11
- matplotlib ≥ 3.7
- openpyxl ≥ 3.1

## Usage

### Full pipeline

```bash
python scripts/run_analysis.py
```

With custom paths:
```bash
python scripts/run_analysis.py --data-dir ./data --output-dir ./results --shell-cm 1.0
```

### Output

All figures are exported as **SVG** (editable text for Inkscape/Illustrator),
**PNG** (raster), and **CSV** (data tables).

```
output/
├── fig1_incubator_validation.svg   # passive control validation
├── fig1_incubator_validation.png
├── fig1_incubator_data.csv
├── fig2_per_rock_fits.svg          # 24-h model fits per rock
├── fig3_residuals.svg              # lag-corrected wētā signal
├── fig4_crossover.svg              # heating–cooling crossover
├── fig5_species_sensitivity.svg    # shell thickness × species RMR
└── results_table.csv               # numerical results
```

### As a library

```python
from igloo_weta.ingest import load_all, summarise_species
from igloo_weta.fitting import fit_all_rocks, compute_species_crossover
from igloo_weta.sensitivity import compute_species_rmr, sweep_all_rocks

ds = load_all("./data")
results = fit_all_rocks(ds.hourly_24h, ds.rock_phys, shell_m=0.01)
crossover = compute_species_crossover(results)
print(f"Crossover: {crossover['T_cross_corr']:.1f}°C")
```

## Project structure

```
igloo_weta/
├── igloo_weta/           # the package
│   ├── constants.py      # physical constants, defaults
│   ├── ingest.py         # data loading & validation
│   ├── physics.py        # thermal ODE, geometry
│   ├── fitting.py        # parameter estimation, F-test
│   ├── sensitivity.py    # shell thickness, allometric RMR
│   └── viz.py            # SVG/PNG/CSV figure export
├── scripts/
│   └── run_analysis.py   # full pipeline entry point
├── data/                 # experimental data files
├── docs/                 # Sphinx documentation source
└── output/               # generated figures (gitignored)
```

## Key findings

| Metric | Value |
|--------|-------|
| Species-level crossover temperature | **10.6 °C** |
| Strongest heater (Rock 18) | Q ≈ 131 mW at 1 cm shell, p = 0.003 |
| Q / RMR ratio | **12–20× resting metabolism** |
| Biological interpretation | Active thermogenesis, comparable to bumblebees |

## Data

The `data/` directory contains:

- **24h_hourly_averages.csv** — mean diurnal temperature cycle per rock
- **full_duration_hourly_aggregates.csv** — incubator passive control
- **full_duration_overall_stats.csv** — per-rock summary statistics
- **total_duration_averages.csv** — day-by-day averages
- **Rock_data.xlsx** — photogrammetric cavity geometry (foam casts)
- **Weta_thermoregulation_datasheet.xlsx** — wētā morphometrics

## Documentation

API docs are auto-built via Sphinx and deployed to GitHub Pages on push
to `main`.  Build locally:

```bash
cd docs
sphinx-build -b html . _build/html
```

## References

- Giraldo D, Adden A, Kuhlemann I, Gras H, Geurten BRH (2019).
  Correcting locomotion dependent observation biases in thermal
  preference of *Drosophila*. *Scientific Reports* 9:3974.
  [doi:10.1038/s41598-019-40459-z](https://doi.org/10.1038/s41598-019-40459-z)

## License

MIT — see [LICENSE](LICENSE).

## Author

**Bart R.H. Geurten**
Department of Zoology, University of Otago, Dunedin, New Zealand
[bgeurte@gwdg.de](mailto:bgeurte@gwdg.de)
