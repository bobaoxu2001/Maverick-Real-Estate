# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0-professional] - 2026-02-26

### Added

- Professional data scale configuration with presets:
  - `quick`
  - `professional`
  - `institutional`
- Environment-driven fetch limits and demo sizing:
  - `DATA_SCALE`
  - `LIMIT_PLUTO`
  - `LIMIT_ROLLING_SALES`
  - `LIMIT_DOB_PERMITS`
  - `LIMIT_HPD_VIOLATIONS`
  - `DEMO_SAMPLE_SIZE`
  - `MACRO_START_DATE`
- New CLI options in `run_pipeline.py`:
  - `--data-scale`
  - `--demo-size`
  - `--limit-pluto`
  - `--limit-sales`
  - `--limit-permits`
  - `--limit-violations`
- Automated model reporting framework:
  - `src/reporting/model_reports.py`
  - Outputs timestamped model artifacts:
    - `model_metrics.csv`
    - `model_results.json`
    - `report.html`
    - PNG diagnostics charts
- Dashboard page for diagnostics:
  - `Model Diagnostics Gallery`
- Project configuration template:
  - `.env.example`
- Data dictionary for analytical transparency:
  - `docs/data_dictionary.md`

### Changed

- Hedonic valuation modeling upgraded with **Stacked Ensemble**:
  - Ridge + Gradient Boosting + Random Forest base learners
  - Linear meta-learner for blended prediction
- Pipeline now persists staging tables for reliable step-by-step execution:
  - `staging_pluto.parquet`
  - `staging_sales.parquet`
  - `staging_permits.parquet`
  - `staging_violations.parquet`
- Dashboard data harmonization improved:
  - Handles `yearbuilt` vs `year_built`
  - Handles `numfloors` vs `num_floors`
  - Safer plotting behavior under sparse/real-world schema variations
- README upgraded with professional enhancement highlights and operational usage.

### Fixed

- Fixed `step_features` early-return behavior that previously bypassed feature engineering.
- Fixed macroeconomic file-path logic that used a truthy `Path` object instead of `.exists()`.
- Fixed distress training flow to incorporate violations signals when available.
- Fixed hedonic model artifact serialization to include GB/stacked pipelines.
- Reduced target-encoding leakage risk by using out-of-fold encoding strategy.
- Added divide-by-zero guard in portfolio diversification metric calculation.

### Quality / Operational Improvements

- Better reproducibility through explicit environment controls.
- Stronger stakeholder-ready deliverables via auto-generated model reporting artifacts.
- More robust standalone step execution across acquisition/ELT/feature/model workflow.

---

## Author

**Allen Xu**
