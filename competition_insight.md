# Competition insight and run guide

This repository supports an inverse-design workflow for the BWB structural
optimization competition. The main deliverable is the end-to-end notebook
[`coupled_stress_optimization.ipynb`](coupled_stress_optimization.ipynb), which
turns the supplied mission profiles into one candidate design per mission.

## 1. Environment setup and running the notebook

The notebook was tested with **Python 3.12**. From the repository root, create
and activate a Python environment, then install the project dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Open `coupled_stress_optimization.ipynb` in Jupyter and run all cells from top
to bottom. The notebook must be launched from the repository root because it
uses paths relative to that directory.

```bash
jupyter notebook coupled_stress_optimization.ipynb
```

### Files and directories to preserve

Keep the following paths together when copying, zipping, or submitting the
project:

- `coupled_stress_optimization.ipynb` — the inverse-design implementation.
- `requirements.txt` — Python dependencies needed to run the notebook.
- `data/bwb_structures_dataset.csv` — structural training data used to fit the
  mass, volume, and stress surrogates.
- `models/ld_surrogate/` — the provided L/D predictor imported by the notebook.
- `gauntlet/runnable_cases.json` — the three machine-readable mission inputs.
- `ntop_model/` — the supplied parameterized nTop model; it is part of the
  competition material even though the notebook does not call nTop directly.

The notebook creates `outputs/` automatically if it is missing. Existing files
inside that directory can be replaced when the notebook is rerun, so keep any
prior results separately if they need to be retained.

## 2. The gauntlet

The `gauntlet/` directory is a lightweight evaluation harness for exercising an
optimizer against a consistent set of mission profiles. It exists to make the
input/output contract explicit and to support repeatable checks outside an
interactive notebook.

### Mission cases

`gauntlet/runnable_cases.json` contains the three competition missions:

| Case | L/D target | Payload target | Fuel target | Flight condition |
|---|---:|---:|---:|---|
| High Speed Dash | 6.0 | 0.75 m³ | 0.45 m³ | 15 kft, 120 kt KCAS, 1.0° AOA |
| Max Endurance | 10.0 | 0.80 m³ | 0.45 m³ | 15 kft, 45 kt KCAS, 8.0° AOA |
| Max Capacity | 15.0 | 1.00 m³ | 0.65 m³ | 5 kft, 220 kt KCAS, 4.5° AOA |

All three cases use a 335 MPa maximum-hotspot-stress limit. Stress is the hard
constraint; L/D and the two volume values are soft targets used in the loss.

### Robustness check

`gauntlet/robustness_check.py` is intended for a command-line optimizer that
accepts `--cases` and `--output` arguments. For each base mission it creates
five scenarios: the baseline, payload target −10% and +10%, and fuel target
−10% and +10%. It then measures the change in optimizer-reported stress and
calculates centered stress sensitivities in MPa/m³.

The robustness utility does **not necessarily work with this version of the
optimizer**. `coupled_stress_optimization.ipynb` is an interactive notebook,
not a command-line target, and the current checkout does not include the
gauntlet scoring helper that `robustness_check.py` imports. The main notebook
still runs its three supplied missions directly and produces the required
submission outputs.

## 3. What the notebook does end to end

The notebook runs the complete inverse-design loop:

1. It loads and cleans the supplied structural dataset, separates training,
   calibration, and test rows, and defines the valid 21-variable design space.
2. It trains bootstrap ensembles for empty mass, payload volume, and fuel
   volume. Each ensemble combines ExtraTrees and histogram gradient-boosting
   regressors so the spread across members can be used as an uncertainty signal.
3. It trains overlapping low- and high-stress experts, then blends them with a
   calibrated classifier. A one-sided stress upper bound is calibrated on held-
   out data and used as the conservative stress screen.
4. For each mission in `runnable_cases.json`, it evaluates design candidates
   with the structural surrogates and provided L/D predictor, then searches the
   bounded design space using differential evolution and short perturbation
   checks.
5. It selects the lowest-loss candidate that passes the conservative stress
   screen. If none does, it re-evaluates database designs for that mission and
   records the fallback path explicitly.
6. It validates the candidate contract and variable rules, then writes the
   submission summary and supporting diagnostics.

The run produces these artifacts in `outputs/`:

- `coupled_stress_output_summary.csv` — one submission-ready row per mission,
  including all 21 design variables, mission conditions, targets, and achieved
  mass, L/D, volume, and stress metrics.
- `coupled_stress_candidates.json` — the machine-readable candidate envelope.
- `coupled_stress_diagnostics.json` — stress calibration, validation, search,
  and selection diagnostics.
- `coupled_stress_validation.png` — held-out stress, sampling-weight, and
  search-history visualizations.

This is end to end because the notebook begins with the supplied training data
and mission definitions, trains the needed surrogate models, runs optimization,
checks the produced designs, and exports the results needed for competition
review. It does not replace a finite-element verification run: the structural
metrics and stress screen are surrogate predictions, which should be stated
clearly in any final technical summary.
