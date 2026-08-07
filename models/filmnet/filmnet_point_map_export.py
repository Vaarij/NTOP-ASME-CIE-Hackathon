#!/usr/bin/env python3
import argparse
import csv
import json
import threading
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from flightcondition import FlightCondition, unit

from film_model_v1 import FiLMNet  # noqa: E402


try:
    from openvsp import vsp as vsp_api
except Exception:
    try:
        import vsp as vsp_api
    except Exception as exc:
        raise ImportError(
            "OpenVSP python API not found. Run in your OpenVSP-enabled environment (e.g. aero_demo310)."
        ) from exc


# Keep ranges identical to combined_bwb_app_v5.py
RANGES = {
    "altitude_kft": (0.0, 40.0),
    "length_m": (0.1, 10.0),
    "kcas": (25.0, 250.0),
    "alpha_deg": (-8.0, 16.0),
    "B1": (100.0, 200.0),
    "B2": (50.0, 200.0),
    "B3": (350.0, 700.0),
    "C2": (550.0, 850.0),
    "C3": (180.0, 280.0),
    "C4": (60.0, 90.0),
    "S1": (40.0, 60.0),
    "S3": (20.0, 40.0),
    "X3": (0.50, 0.65),
}

C1_FIXED = 1000.0
OPENVSP_UNIT_DIVISOR = 1000.0
VSP_LOCK = threading.Lock()
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_VSP3 = SCRIPT_DIR / "bwb.vsp3"
DEFAULT_NORM = SCRIPT_DIR / "norm_stats.json"
DEFAULT_WEIGHTS = SCRIPT_DIR / "checkpoints" / "film_best.pth"


def check_range(name: str, value: float):
    lo, hi = RANGES[name]
    if value < lo or value > hi:
        raise ValueError(f"{name}={value} outside allowed range [{lo}, {hi}]")


def normalize_x3_value(x3: float) -> float:
    # Accept either demo-style X3 (0.50-0.65) or CSV-style X3 (500-650).
    if x3 > 1.5:
        return x3 / C1_FIXED
    return x3


def load_norm_stats(path: Path):
    with path.open("r") as f:
        raw = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}


def infer_arch(sd: dict):
    cond_dim = int(sd["modulation_net.fc.0.weight"].shape[1])
    hidden_dim = int(sd["mlp.layers.0.weight"].shape[0])
    coord_dim = int(sd["mlp.layers.0.weight"].shape[1])
    output_dim = int(sd["mlp.output_layer.weight"].shape[0])

    layer_ids = {
        int(k.split(".")[2])
        for k in sd.keys()
        if k.startswith("mlp.layers.") and k.endswith(".weight")
    }
    num_layers = (max(layer_ids) + 1) + 1 if layer_ids else 4

    extra_ids = {
        int(k.split(".")[2])
        for k in sd.keys()
        if k.startswith("mlp.extra.") and k.endswith(".weight")
    }
    extra_layers = (max(extra_ids) + 1) if extra_ids else 0

    return {
        "cond_dim": cond_dim,
        "coord_dim": coord_dim,
        "output_dim": output_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "extra_layers": extra_layers,
    }


def compute_robust_normals(mesh: pv.PolyData) -> pv.PolyData:
    mesh = mesh.triangulate() if not mesh.is_all_triangles else mesh
    mesh = mesh.compute_normals(
        point_normals=True,
        auto_orient_normals=True,
        consistent_normals=True,
    )
    if "Normals" in mesh.point_data:
        nrm = mesh.point_data["Normals"]
        pts = mesh.points
        if np.mean(np.sum((pts - pts.mean(0)) * nrm, axis=1)) < 0:
            mesh.point_data["Normals"] = -nrm
    return mesh


def export_stl_from_params(vsp3_path: Path, out_stl: Path, geom: dict):
    with VSP_LOCK:
        vsp_api.ClearVSPModel()
        vsp_api.ReadVSPFile(str(vsp3_path.resolve()))
        upc = vsp_api.GetUserParmContainer()
        for pid in vsp_api.FindContainerParmIDs(upc):
            pname = vsp_api.GetParmName(pid)
            if pname in geom:
                vsp_api.SetParmValUpdate(pid, float(geom[pname]))
        vsp_api.Update()
        ok = vsp_api.ExportFile(str(out_stl.resolve()), vsp_api.SET_ALL, vsp_api.EXPORT_STL)
    if not ok:
        raise RuntimeError(f"Failed to export STL: {out_stl}")


def build_cond_vector(stats: dict, re_val: float, mach: float, alpha_deg: float, geom: dict):
    flight_vals = np.array([re_val, mach, alpha_deg], dtype=np.float32)
    geom_vals = np.array(
        [
            geom["B1"], geom["B2"], geom["B3"],
            geom["C1"], geom["C2"], geom["C3"], geom["C4"],
            geom["S1"], geom["S3"], geom["X3"],
        ],
        dtype=np.float32,
    )

    f_mu, f_sd = stats["flight_mean"], stats["flight_std"]
    g_mu, g_sd = stats["shape_mean"], stats["shape_std"]

    f_norm = (flight_vals - f_mu) / (f_sd + 1e-12)
    g_norm = (geom_vals - g_mu) / (g_sd + 1e-12)
    return np.concatenate([f_norm, g_norm], axis=0).astype(np.float32)


def run_filmnet_on_mesh(model: FiLMNet, stats: dict, mesh: pv.PolyData, cond_vec: np.ndarray, device: torch.device):
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    mesh = compute_robust_normals(mesh)

    pts = mesh.points.astype(np.float32)
    nrm = mesh.point_data["Normals"].astype(np.float32)

    pts_scaled = pts / float(OPENVSP_UNIT_DIVISOR)
    coord_min = stats["coord_min"]
    coord_max = stats["coord_max"]
    p_norm = 2.0 * (pts_scaled - coord_min) / (coord_max - coord_min + 1e-12) - 1.0

    # Respect model coordinate dimension inferred from checkpoint.
    coord_dim = int(model.mlp.layers[0].in_features)
    if coord_dim == 6:
        coords = np.concatenate([p_norm, -nrm], axis=1).astype(np.float32)
    elif coord_dim == 3:
        coords = p_norm.astype(np.float32)
    else:
        raise ValueError(f"Unsupported coord_dim={coord_dim}; expected 3 or 6")

    coords_t = torch.from_numpy(coords).to(device)
    cond_t_base = torch.from_numpy(cond_vec).to(device)

    preds = []
    chunk_size = 50000
    with torch.no_grad():
        for s in range(0, len(coords), chunk_size):
            e = min(s + chunk_size, len(coords))
            c = coords_t[s:e]
            cond_t = cond_t_base.unsqueeze(0).expand(e - s, -1)
            out = model(c, cond_t)
            preds.append(out.detach().cpu().numpy())

    pred_norm = np.concatenate(preds, axis=0)
    o_mu, o_sd = stats["output_mean"], stats["output_std"]
    pred_phys = pred_norm * o_sd + o_mu

    cp = pred_phys[:, 0]
    cfx = pred_phys[:, 1]
    cfz = pred_phys[:, 2]
    return pts_scaled, nrm, cp, cfx, cfz


def write_point_map_csv(path: Path, xyz: np.ndarray, vxyz: np.ndarray):
    out = np.concatenate([xyz, vxyz], axis=1)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y", "Z", "VX", "VY", "VZ"])
        writer.writerows(out.tolist())


def main():
    parser = argparse.ArgumentParser(
        description="FilmNet-only point-map exporter (pressure, friction-x, friction-z)"
    )
    parser.add_argument("--altitude_kft", type=float, required=True)
    parser.add_argument("--length_m", type=float, required=True)
    parser.add_argument("--kcas", type=float, required=True)
    parser.add_argument("--alpha_deg", type=float, required=True)

    parser.add_argument("--B1", type=float, required=True)
    parser.add_argument("--B2", type=float, required=True)
    parser.add_argument("--B3", type=float, required=True)
    parser.add_argument("--C2", type=float, required=True)
    parser.add_argument("--C3", type=float, required=True)
    parser.add_argument("--C4", type=float, required=True)
    parser.add_argument("--S1", type=float, required=True)
    parser.add_argument("--S3", type=float, required=True)
    parser.add_argument("--X3", type=float, required=True)

    parser.add_argument("--vsp3", type=str, default=str(DEFAULT_VSP3))
    parser.add_argument("--norm", type=str, default=str(DEFAULT_NORM))
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--outdir", type=str, default=str(Path(__file__).resolve().parent / "outputs"))
    parser.add_argument("--tag", type=str, default="case")

    args = parser.parse_args()

    # Range checks to match demo sliders.
    check_range("altitude_kft", args.altitude_kft)
    check_range("length_m", args.length_m)
    check_range("kcas", args.kcas)
    check_range("alpha_deg", args.alpha_deg)
    for k in ["B1", "B2", "B3", "C2", "C3", "C4", "S1", "S3"]:
        check_range(k, float(getattr(args, k)))
    x3_norm = normalize_x3_value(float(args.X3))
    check_range("X3", x3_norm)

    vsp3_path = Path(args.vsp3)
    norm_path = Path(args.norm)
    weights_path = Path(args.weights)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not vsp3_path.exists():
        raise FileNotFoundError(f"Missing vsp3: {vsp3_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing norm stats: {norm_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing FilmNet weights: {weights_path}")

    # Build flight condition exactly like demo app.
    fc = FlightCondition(
        h=args.altitude_kft * unit("kft"),
        CAS=args.kcas * unit("knots"),
        L=args.length_m * unit("m"),
    )
    re_val = float(fc.Re)
    mach = float(fc.M)

    geom = {
        "B1": args.B1,
        "B2": args.B2,
        "B3": args.B3,
        "C1": C1_FIXED,
        "C2": args.C2,
        "C3": args.C3,
        "C4": args.C4,
        "S1": args.S1,
        "S3": args.S3,
        "X3": float(args.X3),  # Use raw value for VSP API; build_cond_vector normalizes it
    }

    # 1) Build STL from given geometric parameters.
    stl_path = outdir / f"{args.tag}.stl"
    export_stl_from_params(vsp3_path, stl_path, geom)

    # 2) Load FiLMNet.
    stats = load_norm_stats(norm_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sd = torch.load(weights_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    arch = infer_arch(sd)

    model = FiLMNet(
        cond_dim=arch["cond_dim"],
        coord_dim=arch["coord_dim"],
        output_dim=arch["output_dim"],
        hidden_dim=arch["hidden_dim"],
        num_layers=arch["num_layers"],
        extra_layers=arch["extra_layers"],
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    mesh = pv.read(str(stl_path))

    # Keep geometry export on raw X3, but condition FilmNet on normalized X3.
    geom_for_model = {**geom, "X3": x3_norm}

    xyz, normals, cp, cfx, cfz = run_filmnet_on_mesh(
        model=model,
        stats=stats,
        mesh=mesh,
        cond_vec=build_cond_vector(stats, re_val, mach, args.alpha_deg, geom_for_model),
        device=device,
    )

    # 3) Export separate, non-summed vector maps in required format.
    # Pressure force is physically directional: Fp = -p * n.
    v_pressure = -cp[:, None] * normals
    v_fric_x = np.stack([cfx, np.zeros_like(cfx), np.zeros_like(cfx)], axis=1)
    v_fric_z = np.stack([np.zeros_like(cfz), np.zeros_like(cfz), cfz], axis=1)
    v_sum = v_pressure + v_fric_x + v_fric_z

    pressure_csv = outdir / f"{args.tag}_pressure_point_map.csv"
    friction_x_csv = outdir / f"{args.tag}_friction_x_point_map.csv"
    friction_z_csv = outdir / f"{args.tag}_friction_z_point_map.csv"
    sum_point_csv = outdir / f"{args.tag}_sum_force_point_map.csv"

    write_point_map_csv(pressure_csv, xyz, v_pressure)
    write_point_map_csv(friction_x_csv, xyz, v_fric_x)
    write_point_map_csv(friction_z_csv, xyz, v_fric_z)
    write_point_map_csv(sum_point_csv, xyz, v_sum)

    print("Done. Outputs:")
    print(f" - STL: {stl_path}")
    print(f" - Pressure CSV: {pressure_csv}")
    print(f" - Friction X CSV: {friction_x_csv}")
    print(f" - Friction Z CSV: {friction_z_csv}")
    print(f" - Summed force (point-wise) CSV: {sum_point_csv}")


if __name__ == "__main__":
    main()
