# Runnable mission cases

This directory contains the machine-readable versions of the three mission
profiles defined in the repository [README](../README.md). The coupled-stress
notebook reads these cases when it produces one optimized design per mission.

## Run the current workflow

From the repository root, install dependencies with
`pip install -r requirements.txt`, then run all cells in
`coupled_stress_optimization.ipynb`. It reads
`gauntlet/runnable_cases.json` and writes the following files under `outputs/`:

- `coupled_stress_output_summary.csv` — the submission-ready table containing
  mission conditions, all 21 design variables, and achieved metrics.
- `coupled_stress_candidates.json` — the same selected designs in a
  machine-readable candidate envelope.
- `coupled_stress_diagnostics.json` — surrogate-validation and optimization
  diagnostics.
- `coupled_stress_validation.png` — a compact validation and search-history
  figure.

The notebook uses a surrogate-only stress check because this repository does
not provide a callable FE solver. Its conservative stress upper bound is used
for optimizer selection; the CSV reports the predicted mean hotspot stress.

## Candidate JSON format

`coupled_stress_candidates.json` contains one candidate for every supplied
case. Each candidate echoes the mission from `runnable_cases.json`, includes
the 21 design variables, and reports the achieved metrics:

```json
{
  "schema_version": 1,
  "candidates": [
    {
      "case_id": "high_speed_dash",
      "mission": {
        "ld_target": 6.0,
        "payload_volume_min_m3": 0.75,
        "fuel_volume_min_m3": 0.45,
        "altitude_kft": 15.0,
        "kcas_kt": 120.0,
        "aoa_deg": 1.0,
        "stress_max_mpa": 335.0
      },
      "design": { "...": "21 design variables" },
      "metrics": {
        "empty_mass_kg": 0.0,
        "ld": 0.0,
        "payload_volume_m3": 0.0,
        "fuel_volume_m3": 0.0,
        "max_hotspot_stress_mpa": 0.0
      }
    }
  ]
}
```

The CSV is the preferred file for reviewing or submitting results: it includes
the mission targets and conditions beside the selected design and its achieved
metrics. Volumes in both outputs are expressed in m³; stress is in MPa.

## Sensitivity utility

`robustness_check.py` is retained for command-line optimizers that accept
`--cases` and `--output` arguments. It creates baseline and independent ±10%
payload/fuel-target variants, then reports stress deltas and centered
sensitivities. It relies on the gauntlet scoring helper, so it is not part of
the notebook execution path.
