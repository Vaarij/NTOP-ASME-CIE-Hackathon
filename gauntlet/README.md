# BWB optimizer gauntlet

Run an optimizer target with the documented contract:

```bash
python3 gauntlet/score_candidates.py --target sample_1/main.py
```

The scorer starts the target from its own directory as:

```bash
python3 main.py --cases /absolute/path/to/gauntlet/runnable_cases.json --output /temporary/candidates.json
```

`runnable_cases.json` contains the three original README missions. The target
must write this exact JSON shape:

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
      "design": {
        "C2/C1": 0.70,
        "C3/C1": 0.23,
        "C4/C1": 0.075,
        "B1/C1": 0.15,
        "B2/C1": 0.12,
        "B3/C1": 0.52,
        "X3/C1": 0.575,
        "S1": 50.0,
        "S3": 30.0,
        "C1": 3000.0,
        "Skin Thickness": 0.003,
        "Front Spar Chord %": 0.25,
        "Rear Spar Chord %": 0.65,
        "Spar Thickness": 0.004,
        "# of Ribs": 8,
        "Rib Thickness": 0.006,
        "Wingbox Cutout": 0.03,
        "# of Fuselage Ribs": 7,
        "# of Fuselage Spars": 6,
        "Fuselage Struct Thickness": 0.010,
        "Fuselage Struct Width": 0.006
      },
      "metrics": {
        "empty_mass_kg": 50.0,
        "ld": 6.0,
        "payload_volume_m3": 0.75,
        "fuel_volume_m3": 0.45,
        "max_hotspot_stress_mpa": 335.0
      }
    }
  ]
}
```

The target must return one record for every supplied case and echo each mission
exactly. `output_schema.json` documents the envelope; the scorer is stricter and
also enforces the documented variable bounds and integer axes.

The loss follows `README.md`: `0.4 * mass / 50` plus weighted fractional
shortfalls for L/D, fuel volume, and payload volume. Stress above 335 MPa is a
hard failure with infinite loss. The two records in
`negative_stress_fixtures.json` are regression fixtures for that rule, not
missions sent to an optimizer.

## Volume-target stress sensitivity

Run an optimizer against a baseline plus independent ±10% payload- and
fuel-volume target perturbations:

```bash
python3 gauntlet/robustness_check.py --target path/to/main.py --percent 10 \
  --report outputs/volume_stress_sensitivity.json
```

The command runs five scenarios for each original mission: baseline, payload
−/+10%, and fuel −/+10%. It reports one-sided stress deltas and centered
sensitivities in MPa/m³. A scenario over 335 MPa remains visible in the report;
the command is a sensitivity measurement, not a pass/fail robustness gate.

## Hybrid sklearn verification

The maintained target is `hybrid_sklearn_optimization.py`. Its default search is
fast enough for repeated gauntlet checks; use a larger global DE budget only for
an offline final run:

```bash
python3 hybrid_sklearn_optimization.py --cases gauntlet/runnable_cases.json \
  --output outputs/hybrid_candidates.json --diagnostics outputs/hybrid_diagnostics.json \
  --de-maxiter 100
```

Diagnostics distinguish an optimizer-feasible result from a stress-safe
database fallback. The fallback preserves the strict output contract but is not
evidence that every mission threshold was met.

Verify the hybrid sklearn optimizer across three deterministic root seeds while
recording both stress-sensitivity consistency and whether selected candidates
are database seeds or off-grid search results:

```bash
python3 gauntlet/verify_hybrid_sklearn.py
```

The verifier writes a JSON summary and a scenario-level CSV under `outputs/`.
It reports provenance rather than rejecting database-seed selections, since the
optimizer deliberately uses a hybrid seed-plus-continuous-search strategy.

`sample_1/main.py` is only a protocol smoke-test: its metrics are synthetic and
must be replaced by a real evaluated optimizer output.
