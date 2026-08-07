# FiLMNet — field-resolved surface aerodynamics

FiLMNet is the **field** aerodynamic surrogate from BlendedNet++. Where the L/D
surrogate returns a single integrated `(CL, CD)`, FiLMNet predicts the **local
surface aerodynamic distribution** — pressure and skin-friction — at every point
on the aircraft skin, conditioned on the planform geometry and flight state.

```
3D surface coordinate (x, y, z)  +  condition vector (geometry + flight)
        ->  FiLMNet  ->  local surface aerodynamic field  (pressure / Cf vector)
```

It is a **coordinate-MLP with FiLM conditioning**: a modulation network turns the
condition vector into per-layer scale/shift (γ, β) parameters that modulate an MLP
evaluated at each surface coordinate. This lets one network represent the field
over the whole design space (see `film_model_v1.py`).

## Files

| File | Purpose |
|---|---|
| `film_model_v1.py` | The `FiLMNet` architecture (`FiLMModulation` + `ModulatedMLP`). |
| `checkpoints/film_best.pth` | Trained weights. Load with `strict=True`. |
| `norm_stats.json` | Input/output normalisation (flight, shape, coordinate, and output mean/std). |
| `filmnet_point_map_export.py` | End-to-end exporter: geometry → surface point cloud (via OpenVSP) → FiLMNet inference → CSV maps `X,Y,Z,VX,VY,VZ`. |
| `filmnet_direct_stl.py` | Same inference, but reads a surface STL you already have (no OpenVSP call). |

## Load the model

The shipped checkpoint is `cond_dim=13`, `coord_dim=6`, `output_dim=3`,
`hidden_dim=256`, `num_layers=4`, `extra_layers=3`. The bundled scripts **infer
these dims from the checkpoint tensor shapes**, so they stay correct if the model
is retrained — mirror that rather than hard-coding:

```python
import torch, json
from film_model_v1 import FiLMNet

sd = torch.load("checkpoints/film_best.pth", map_location="cpu")
sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd

cfg = dict(
    cond_dim   = sd["modulation_net.fc.0.weight"].shape[1],   # 13 = 10 shape + 3 flight
    hidden_dim = sd["mlp.layers.0.weight"].shape[0],          # 256
    coord_dim  = sd["mlp.layers.0.weight"].shape[1],          # 6 = point(3) + normal(3)
    output_dim = 3,
    num_layers = 4,
    extra_layers = 3,
)
model = FiLMNet(**cfg)
model.load_state_dict(sd, strict=True)
model.eval()

norm = json.load(open("norm_stats.json"))                     # normalisation stats
```

- **Condition vector (13)** = normalised `flight` (3: Re, Mach, α) + normalised
  `shape` (10 geometry). Built by `build_cond_vector(...)` in the exporter.
- **Coordinate input (6)** = normalised surface point `(x,y,z)` concatenated with
  the (negated) surface unit normal `(nx,ny,nz)`.
- **Output (3)** = the surface field, de-normalised with `output_mean/std`.

`norm_stats.json` carries `flight_mean/std` (3), `shape_mean/std` (10),
`coord_min/max` (3) and `output_mean/std` (3). Normalise inputs and de-normalise the
output with these — the exporter scripts show the exact recipe.

Pressure is written as a **directional vector** using the surface normals:
`VX = -p·nx`, `VY = -p·ny`, `VZ = -p·nz`.

## Running the point-map exporter

`filmnet_point_map_export.py` needs **OpenVSP** (for surface tessellation) in
addition to the Python requirements:

1. Install OpenVSP: <https://openvsp.org/download.php>
2. Put its Python modules on `PYTHONPATH` if not automatic:
   ```bash
   export PYTHONPATH=/opt/OpenVSP/python:/opt/OpenVSP/python/openvsp:$PYTHONPATH
   ```
3. Run:
   ```bash
   python filmnet_point_map_export.py \
     --altitude_kft 25 --length_m 5 --kcas 150 --alpha_deg 2 \
     --B1 150 --B2 125 --B3 525 --C2 700 --C3 230 --C4 75 --S1 50 --S3 30 --X3 575
   ```
   (`X3` accepts either the fraction `0.575` or the CSV-style `575`, auto-scaled by /1000.)

Input ranges accepted by the exporter: `altitude_kft` [0, 40], `length_m` [0.1, 10],
`kcas` [25, 250], `alpha_deg` [−8, 16], `B1` [100, 200], `B2` [50, 200],
`B3` [350, 700], `C2` [550, 850], `C3` [180, 280], `C4` [60, 90], `S1` [40, 60],
`S3` [20, 40], `X3` [0.50, 0.65].

## When to use which aero model

- **L/D surrogate** (`../ld_surrogate/`) — fast, torch-free, integrated `CL/CD/LD`.
  Use it *inside* the optimization loop to hit `L/D_target`.
- **FiLMNet** (here) — heavier (PyTorch), returns the full surface field. Use it to
  inspect / post-process the aerodynamic loading of a chosen design, or as the
  source of the surface pressure loads that drive the structural FE dataset.
