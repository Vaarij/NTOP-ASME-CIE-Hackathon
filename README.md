# AI-Driven Multidisciplinary Design Optimization — BWB Inverse Design

**ASME IDETC/CIE Student Hackathon — nTop × MIT DeCoDE Lab**

This repository bundles the **data and forward models** for the Blended-Wing-Body
(BWB) structural inverse-design challenge: given a mission profile and structural
constraints, find the **lightest internal structure and external planform** whose
stress stays under the allowable, while getting as close as possible to a target
lift-to-drag ratio and the payload/fuel volume minima.

It provides the three ingredients a participant builds an inverse pipeline on top of:

| Provided artifact | Where it lives here |
|---|---|
| **Structures dataset** (13,720 FE designs) | [`data/bwb_structures_dataset.csv`](data/bwb_structures_dataset.csv) |
| **Forward surrogate** — integrated `L/D` from geometry + flight | [`models/ld_surrogate/`](models/ld_surrogate/) |
| **Parameterized nTop implicit model** | [`ntop_model/`](ntop_model/) *(BlendedNet++StructuresVisualizeShare.ntop)* |

---

## The challenge

### Background

Engineers are often forced into an impossible decision early in the design
process: iterate quickly with low-fidelity models, or commit to a high-fidelity
model that is difficult to update. This leads to the **lock-in trap**, where teams
are forced to pursue a sub-optimal design chosen early in the process because
changes to downstream CAD and FEA models are too costly.

New technologies like AI and implicit modeling are enabling a **code-first**
approach to engineering that connects system requirements directly to geometries.
Rapid performance evaluations from surrogate physics models, along with
dynamically parameterizable implicit models, eliminate geometric failure
bottlenecks — allowing for true multidisciplinary design analysis and optimization
(MDAO) and agile inverse design workflows.

### Objective

Develop an **automated inverse design process** that takes high-level mission
profile targets and structural constraints as inputs and directly outputs both the
optimal **external planform geometry** and the **internal structural
configuration** for a BWB aircraft.

### Task

Construct an optimization framework — multi-objective genetic algorithms, Bayesian
optimization, machine-learning-driven inverse loops, or anything else — that
navigates this coupled, multidisciplinary design space. The inverse pipeline must
accept a target mission and system payload envelope and seamlessly map them to
concurrent external and internal geometric definitions.

---

## Framework inputs and outputs

### Inputs — the mission profile and constraints

| Input | Symbol | Description | Type |
|---|---|---|---|
| Minimum aerodynamic performance | `L/D_min` | Target lift-to-drag ratio | **soft target** |
| Minimum payload volume | `V_payload_min` | Volumetric target | **soft target** |
| Minimum fuel volume | `V_fuel_min` | Volumetric target | **soft target** |
| Maximum hotspot stress | `Stress_max` | Structural safety threshold | **hard constraint** |

**Only `Stress_max` is a hard constraint.** A design whose maximum hotspot stress
exceeds the allowable is **invalid** — it is discarded for that case no matter how
good the rest of it looks. `L/D_min`, `V_payload_min`, and `V_fuel_min` are
targets: get as close as you can and report the shortfall. Missing one degrades
the score; it never invalidates the design. This is deliberate, since it lets you
trade a small volume or `L/D` miss against a real mass saving, which is exactly
the multidisciplinary trade the challenge is about.

Flight conditions:

| Parameter | Description | Unit | Min | Max |
|---|---|---|---:|---:|
| `Altitude` | altitude | kft | 0.0015 | 18 |
| `KCAS` | calibrated airspeed | kt | 33 | 250 |
| `AOA` | angle of attack | deg | -8 | 16 |

### Outputs — the generated design vector

**21 variables: 10 planform + 11 structural.** These are exactly the geometry and
structure columns of the dataset below — the pipeline emits a full row of them per
mission profile.

- **Planform (10, via BlendedNet++):** root chord `C1`, chord-length ratios
  (`C2/C1`, `C3/C1`, `C4/C1`), span-fraction parameters (`B1/C1`, `B2/C1`,
  `B3/C1`), sweep angles (`S1`, `S3`), and the outboard break streamwise fraction
  (`X3/C1`).
- **Structures (11, via the nTop implicit model):** skin thickness, placement and
  thickness of spars and wingbox, and the count and thickness of ribs and spars
  for both the wing and the fuselage.

Bounds for every variable are tabulated under [The dataset](#the-dataset).

---

## Test cases

The pipeline is evaluated against three mission profiles:

| Case | Condition | `L/D_target` | `V_payload_min` (m³) | `V_fuel_min` (m³) | Altitude (kft) | KCAS (kt) | AOA (deg) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | **High Speed Dash** | 6.0 | 0.75 | 0.45 | 15 | 120 | 1.0 |
| 2 | **Max Endurance** | 10.0 | 0.80 | 0.45 | 15 | 45 | 8.0 |
| 3 | **Max Capacity** | 15.0 | 1.00 | 0.65 | 5 | 220 | 4.5 |

`Stress_max` is the **335 MPa** structural allowable for all three cases, and it
is the one threshold that **must** be met — the `L/D` and volume figures above are
targets to approach, not pass/fail gates.

**Units.** Volumes above are m³; the CSV stores `Payload Volume` and `Fuel Volume`
in mm³, so divide by `1e9` before comparing. All three targets are inside what
the dataset spans (payload 0.066–2.083 m³, fuel 0.017–0.900 m³).


---

## Visualizations

**Parameterized internal structure** — the nTop implicit model adapts rib/spar
frequencies and shell thicknesses as the external mold line changes:

![Parameterized BWB structure](assets/structure_param.gif)

**Finite-element stress field** — each design is solved under the mapped
aerodynamic + inertial loads; the label is a hotspot stress read off this field:

![BWB stress field](assets/stress.gif)

---

## Repository layout

```
bwb-inverse-design/
├── data/
│   └── bwb_structures_dataset.csv      # 13,720 designs × (24 params + 4 outputs)
├── models/
│   └── ld_surrogate/                   # geometry + flight -> CL, CD, L/D  (pure NumPy)
│       ├── predict_ld.py               #   -> start here: predict_ld(geom, alt, kcas, aoa)
│       ├── regressor.py, reg_full.json, reg_feasible.json
│       ├── flight_conversion.py, aero_design_space.json
│       └── README.md
├── ntop_model/                         # parameterized nTop implicit model (+ free-license info)
│   └── BlendedNet++StructuresVisualizeShare.ntop
├── assets/                             # visualizations used in this README
├── requirements.txt
└── README.md
```

---

## The dataset

`data/bwb_structures_dataset.csv` — **13,720** finite-element designs, one per row.
Each row is a full BWB design (external planform + internal structure) at a flight
condition, with the resulting mass, volumes, and maximum hotspot stress. **28 columns = 24
input parameters + 4 outputs.**

### Inputs (24)

The 24 parameters are 10 planform-geometry variables, 3 flight-condition variables,
and 11 internal-structure variables. Ranges below are the actual min/max in the
shipped data.

**Geometry — planform (nTop ratios)**

| Parameter | Description | Unit | Min | Max |
|---|---|---|---:|---:|
| `C2/C1` | chord ratio (station 2 / root) | – | 0.55 | 0.85 |
| `C3/C1` | chord ratio (station 3 / root) | – | 0.18 | 0.28 |
| `C4/C1` | chord ratio (tip / root) | – | 0.06 | 0.09 |
| `B1/C1` | span-fraction parameter 1 | – | 0.1 | 0.2 |
| `B2/C1` | span-fraction parameter 2 | – | 0.05 | 0.2 |
| `B3/C1` | span-fraction parameter 3 | – | 0.35 | 0.7 |
| `X3/C1` | outboard break streamwise fraction | – | 0.5 | 0.65 |
| `S1` | inboard sweep angle | deg | 40 | 60 |
| `S3` | outboard sweep angle | deg | 20 | 40 |
| `C1` | root chord (scale) | mm | 2500 | 4000 |

**Flight condition (mission)**

| Parameter | Description | Unit | Min | Max |
|---|---|---|---:|---:|
| `Altitude` | altitude | kft | 0.0015 | 18 |
| `KCAS` | calibrated airspeed | kt | 33 | 250 |
| `AOA` | angle of attack | deg | -8 | 16 |

**Structure (nTop implicit)**

| Parameter | Description | Unit | Min | Max |
|---|---|---|---:|---:|
| `Skin Thickness` | skin shell thickness | m | 0.0003 | 0.0050|
| `Front Spar Chord %` | front spar chordwise position | frac | 0.18 | 0.35 |
| `Rear Spar Chord %` | rear spar chordwise position | frac | 0.55 | 0.75 |
| `Spar Thickness` | spar shell thickness | m | 0.00098 | 0.0080 |
| `# of Ribs` | wing rib count (integer) | – | 3 | 14 |
| `Rib Thickness` | rib shell thickness | m | 0.0015 | 0.015 |
| `Wingbox Cutout` | wingbox cutout fraction | frac | 0.01 | 0.05 |
| `# of Fuselage Ribs` | fuselage rib count (**odd** integer) | – | 3 | 11 |
| `# of Fuselage Spars` | fuselage spar count (integer) | – | 3 | 12 |
| `Fuselage Struct Thickness` | fuselage member thickness | m | 0.002 | 0.025 |
| `Fuselage Struct Width` | fuselage member width | m | 0.0010 | 0.015 |

**Integer axes.** `# of Ribs` and `# of Fuselage Spars` take every integer in their
range; `# of Fuselage Ribs` is sampled **odd-only** — `{3, 5, 7, 9, 11}`. An
optimizer proposing an even fuselage rib count is off-distribution.

### Outputs (4)

| Column | Meaning | Unit |
|---|---|---|
| `Aircraft Empty Weight` | structural mass to minimize | kg *(19.5 – 1639)* |
| `Payload Volume` | internal payload volume achieved | mm³ *(÷1e9 → m³)* |
| `Fuel Volume` | internal fuel volume achieved | mm³ *(÷1e9 → m³)* |
| `Max Hotspot Stress` | **Maximum hotspot stress** (feasibility label) | MPa |

**About the `Max Hotspot Stress` label.** The raw FE maximum is dominated by mesh
singularities (unbounded, mesh-dependent). The label here is instead a **maximum
hotspot stress** — the field averaged over a fixed 5 mm physical ball, then the
max of that locally-averaged field, worst of the components — which is
singularity-robust and mesh-independent. The structural
allowable is **335 MPa** (aluminium 7075-T6 yield / 1.5 safety factor); ~41 % of
the dataset lies above it on purpose, giving a constraint-boundary-rich sample.

All 28 columns are fully numeric across all 13,720 rows.

---

## The forward model

### L/D surrogate — `models/ld_surrogate/`
Torch-free MLP: `(9 planform vars + C1) + (altitude, KCAS, AoA) → CL, CD, L/D`.
Fast enough to sit inside an optimizer population as a single matmul. The root
chord `C1` enters aerodynamics **only through Reynolds number**, so it acts as a
near-pure scale knob (volume/mass up, L/D barely moved) — the multidisciplinary
coupling the challenge is about. See [`models/ld_surrogate/README.md`](models/ld_surrogate/README.md).

```bash
python models/ld_surrogate/predict_ld.py --demo
```

---

## Quickstart

```bash
python -m pip install -r requirements.txt

# 1. integrated L/D from a mission profile
python models/ld_surrogate/predict_ld.py --demo

# 2. load the dataset
python -c "import pandas as pd; d=pd.read_csv('data/bwb_structures_dataset.csv'); print(d.shape); print(d.columns.tolist())"
```

A minimal inverse-design loop then: propose a design vector → predict `L/D`
(L/D surrogate) and `mass / volumes / stress` (fit your own surrogates on the CSV,
or use the dataset directly) → **reject anything with `stress > 335 MPa`**, then
among the survivors minimize mass while penalizing the shortfall against
`L/D_target` and the volume minima. The exact objective this is scored on is in
[The scoring metric](#the-scoring-metric).

## The nTop model

[`ntop_model/`](ntop_model/) holds the parameterized BWB implicit model
(`BlendedNet++*.ntop`) — the generator behind the structural dataset, whose
internal rib/spar layout and shell thicknesses adapt automatically to the external
mold line (see the first visualization above). You only need it to regenerate
geometry or run new structural cases; the dataset and surrogates here are
self-contained.

Running the `.ntop` file requires **nTop**. Students and educators can request a
**free license** through the nTop education program — **<https://www.ntop.com/education/>**.
See [`ntop_model/README.md`](ntop_model/README.md) for details.

---

## Scope, assumptions, and constraints

**Out of scope**

- **External physics solvers.** Do not integrate external CFD (Fluent, OpenFOAM)
  or FEA (Ansys, Abaqus). All physical and structural metrics must be evaluated
  strictly using the dataset and forward surrogate models provided here.
- **Dynamic aeroelasticity.** Flutter and time-dependent aerodynamic/structural
  interactions are ignored for the scope of this hackathon.

**Assumptions**

- Parameter ranges for both planform geometry and structural configuration are the
  ones specified above and in the provided JSON files.
- There are known limitations in the accuracy of the CFD and FEA simulations used
  to generate the datasets — your approach should **assume the data is valid**.
  The dataset will be regenerated in the future, so please **do not publish work
  based on this version**.

---

## Submission requirements

Teams submit a zipped repository containing:

| Deliverable | Detail |
|---|---|
| **Inverse pipeline implementation** | Well-documented Python scripts or Jupyter notebooks implementing the optimization / ML algorithm that maps the specified inputs to the outputs. *If the organizers cannot validate your code, the results are considered invalid.* |
| **Output summary file** | CSV or JSON of the generated optimal design variables (10 planform + 11 structural) evaluated against all 3 test-case mission profiles, together with the achieved mass, `L/D`, payload/fuel volumes, and maximum hotspot stress per case — so the stress constraint can be checked and any `L/D`/volume shortfall scored. |
| **Technical summary** | Max 3-page PDF explaining the optimization strategy, how volumetric and stress constraints were handled, and visualizations of the trade-off spaces (e.g. Pareto front of mass vs. aerodynamic efficiency). |
| **Reproducibility guide** | A `README.md` outlining software dependencies (`requirements.txt` or environment files) and exact execution commands. |

## Evaluation criteria (100 pts)

| Criterion | Points | What it measures |
|---|---:|---|
| Design optimization performance | 40 | Meeting the 335 MPa structural allowable (**required** — a design over it is invalid and scores nothing for that case), then minimizing overall structural mass and how closely `L/D_target` and the payload/fuel volume minima are approached |
| Methodological innovation | 20 | Rigor, algorithmic efficiency, and sophistication of the inverse loop or ML approach |
| Multidisciplinary parameter coupling | 20 | How intelligently the pipeline balances trade-offs between external aerodynamics and internal implicit structural properties |
| Code cleanliness & reproducibility | 20 | Compliance with submission instructions, readability of code, and clarity of the final technical presentation |

### The scoring metric

Design optimization performance is scored with the loss function below — the same
objective you should be minimizing inside your inverse loop. Full write-up:
[`assets/optimization_loss_function2.pdf`](assets/optimization_loss_function2.pdf).

Because the terms carry different units (kg vs. m³ vs. dimensionless `L/D`), every
variable is **normalized against its target or a baseline reference** before the
40/20/20/20 weighting is applied — otherwise the weights are skewed by unit choice.

$$
\mathcal{L} = 0.4\left(\frac{M}{M_{ref}}\right)
+ 0.2\,\mathrm{ReLU}\left(\frac{LD_{target} - LD}{LD_{target}}\right)
+ 0.2\,\mathrm{ReLU}\left(\frac{V_{f,target} - V_f}{V_{f,target}}\right)
+ 0.2\,\mathrm{ReLU}\left(\frac{V_{p,target} - V_p}{V_{p,target}}\right)
+ \mathcal{P}_{stress}
$$

**Breakdown**

- **Mass ($M$) — weight 0.4.** Minimized directly. Dividing by a reference mass
  $M_{ref}$ keeps it dimensionless so it scales cleanly against the other terms;
  the term shrinks as mass drops.
- **Target metrics ($LD$, $V_f$, $V_p$) — weight 0.2 each.** Scored one-sided with
  $\mathrm{ReLU}(x) = \max(0, x)$:
  - **Exceed** the target → the bracket is negative, ReLU returns **0**, no penalty
    (over-delivering earns nothing extra).
  - **Below** the target → the bracket is positive and you take a penalty
    **proportional to the fractional shortfall**, weighted by 0.2.
- **Stress hard constraint ($\mathcal{P}_{stress}$).** Exceeding the allowable makes
  the design entirely invalid, so it acts as a step penalty:

$$
\mathcal{P}_{stress} =
\begin{cases}
0 & \text{if } \sigma \le \sigma_{max} \\
\infty & \text{if } \sigma > \sigma_{max}
\end{cases}
$$

with $\sigma_{max} = 335$ MPa. If your optimizer cannot handle infinity, substitute a
large penalty barrier such as $\lambda\,\mathrm{ReLU}(\sigma - \sigma_{max})$ with
$\lambda$ set arbitrarily high (e.g. $10^6$).

---

## Requirements

`numpy`, `scipy`, `pandas` — pure Python, no GPU. That's all the L/D surrogate and
dataset need.

## Attribution

BWB geometry, aerodynamic surrogates, and structural dataset from the
**BlendedNet++** effort. Challenge: **nTop / MIT DeCoDE Lab**.

**Point of contact:** Matthew Mueller — `matthewmueller@ntop.com`
