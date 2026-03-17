"""
 ╔═══════════════════════════════════════════════════════════════════════╗
 ║  ░▀█▀░█▀▀░█░░░█▀█░█▀█░░░█░█░█▀▀░▀█▀░█▀█░░░░░░░░░░░░░░░░░░░░░░░  ║
 ║  ░░█░░█░█░█░░░█░█░█░█░░░█▄█░█▀▀░░█░░█▀█░░░░░░░░░░░░░░░░░░░░░░░  ║
 ║  ░▀▀▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀░░░▀░▀░▀▀▀░░▀░░▀░▀░░░░░░░░░░░░░░░░░░░░░░░  ║
 ║                                                                       ║
 ║   Thermal model for wētā burrow heat exchange              v0.1.0    ║
 ║   ── model. fit. predict. science. ──                                 ║
 ╚═══════════════════════════════════════════════════════════════════════╝

A lumped-parameter heat conduction model for stone-shingle wētā burrows.
Extends the IGLOO framework (Giraldo et al. 2019) to estimate active
thermoregulation in ectothermic insects from field temperature recordings
and photogrammetric burrow geometry.

Modules:
    constants   Physical constants and default configuration.
    ingest      Data loading, validation, and cleaning.
    physics     Thermal ODE, geometry, penetration depth.
    fitting     Parameter estimation, null model, F-test.
    sensitivity Shell thickness sweep, allometric RMR.
    viz         Publication figures: SVG, PNG, CSV export.
"""

__version__ = "0.1.0"
__author__ = "Bart R.H. Geurten"
