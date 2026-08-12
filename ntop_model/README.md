# Parameterized nTop Model

This folder holds the **parameterized BWB implicit model** (`BlendedNet++*.ntop`) —
the generator behind the structural dataset. Its internal rib/spar layout and shell
thicknesses adapt automatically to the external mold line, so a single file spans
the whole design space (see the parameterized-structure animation in the top-level
[`README`](../README.md#visualizations)).

> The `.ntop` file is provided in this folder. You only need it if you want to
> **regenerate geometry or run new structural cases** — the [dataset](../data/) and
> [forward surrogates](../models/) in this repo are self-contained and need no
> nTop install.

## Running the model — request a free nTop license

Opening and running the `.ntop` file requires **nTop**. Students and educators can
request a **free license** through the nTop education program:

### → https://www.ntop.com/education/

1. Go to the link above and request an education (student/academic) license.
2. Install nTop and activate it with the license you receive.
3. Open the `.ntop` file in this folder to inspect or drive the parameterized model.

The design variables exposed by the model (and their ranges) are the geometry and
structure parameters documented in the top-level
[README parameter tables](../README.md#inputs-24).
