#!/usr/bin/env python3
"""Scikit-learn hybrid feasibility-first optimizer with the gauntlet CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from scipy.special import ndtr
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "models" / "ld_surrogate"))
from predict_ld import predict_ld_batch  # noqa: E402


RNG_SEED = 20260819
STRESS_FINAL = 335.0
MIN_FEAS_PROB = 0.80
TOP_K = 10
DE_POPSIZE = 8
# A Sobol population is the practical gauntlet default (hundreds of global
# evaluations, versus the old one-generation local search). Callers can raise
# this, e.g. --de-maxiter 100, for an offline final optimization.
DE_MAXITER = 0
BO_SAMPLES = 256
CONSTRAINT_WEIGHT = 1e6
DESIGN_COLUMNS = [
    "C2/C1", "C3/C1", "C4/C1", "B1/C1", "B2/C1", "B3/C1", "X3/C1", "S1", "S3", "C1",
    "Skin Thickness", "Front Spar Chord %", "Rear Spar Chord %", "Spar Thickness", "# of Ribs",
    "Rib Thickness", "Wingbox Cutout", "# of Fuselage Ribs", "# of Fuselage Spars",
    "Fuselage Struct Thickness", "Fuselage Struct Width",
]
FLIGHT_COLUMNS = ["Altitude", "KCAS", "AOA"]
FEATURE_COLUMNS = DESIGN_COLUMNS + FLIGHT_COLUMNS
TARGET_COLUMNS = ["Aircraft Empty Weight", "Payload Volume", "Fuel Volume", "Max Hotspot Stress"]
BOUNDS = np.array([
    [.55, .85], [.18, .28], [.06, .09], [.1, .2], [.05, .2], [.35, .7], [.5, .65], [40, 60], [20, 40],
    [2500, 4000], [.0003, .005], [.18, .35], [.55, .75], [.00098, .008], [3, 14], [.0015, .015], [.01, .05],
    [3, 11], [3, 12], [.002, .025], [.001, .015],
], dtype=float)
LO, HI = BOUNDS[:, 0], BOUNDS[:, 1]


def normalize(designs: np.ndarray) -> np.ndarray:
    return (np.asarray(designs) - LO) / (HI - LO)


def repair_design(designs: np.ndarray) -> np.ndarray:
    result = np.clip(np.asarray(designs, dtype=float).copy(), LO, HI)
    result[..., 14] = np.rint(result[..., 14])
    result[..., 17] = 2 * np.rint((result[..., 17] - 3) / 2) + 3
    result[..., 18] = np.rint(result[..., 18])
    return result


def mission_for_model(mission: dict[str, float], name: str = "mission") -> dict[str, float | str]:
    return {
        "mission": name,
        "altitude": float(mission["altitude_kft"]),
        "kcas": float(mission["kcas_kt"]),
        "aoa": float(mission["aoa_deg"]),
        "ld_target": float(mission["ld_target"]),
        "payload_target": float(mission["payload_volume_min_m3"]),
        "fuel_target": float(mission["fuel_volume_min_m3"]),
        "stress_limit": float(mission["stress_max_mpa"]),
    }


class HybridOptimizer:
    def __init__(self, seed: int = RNG_SEED) -> None:
        self.seed = seed
        data = pd.read_csv(ROOT / "data" / "bwb_structures_dataset.csv")
        self.data = data.loc[data["Max Hotspot Stress"] < 1e4].reset_index(drop=True)
        self.designs = self.data[DESIGN_COLUMNS].to_numpy(float)
        self.features = self.data[FEATURE_COLUMNS].to_numpy(float)
        self.outputs = self.data[TARGET_COLUMNS].to_numpy(float)
        self.outputs[:, 1:3] /= 1e9
        self.feasible = self.outputs[:, 3] <= STRESS_FINAL
        self.train_idx, self.test_idx = train_test_split(
            np.arange(len(self.designs)), test_size=.20, random_state=seed, stratify=self.feasible,
        )
        self.tree_models: list[ExtraTreesRegressor] = []
        self.smooth_models: list[HistGradientBoostingRegressor] = []
        self.model_weights = np.full(4, .5)
        self.residual_scale = np.ones(4)
        self.classifier: ExtraTreesClassifier | None = None
        self.stress_k = 1.96
        self.stress_screen_margin = 0.0

    def fit(self) -> dict[str, float]:
        reg_kw = dict(n_estimators=16, min_samples_leaf=3, max_features=.85, bootstrap=True, max_samples=.85, n_jobs=1)
        for index in range(4):
            model = ExtraTreesRegressor(**reg_kw, random_state=self.seed + index)
            model.fit(self.features[self.train_idx], self.outputs[self.train_idx, index])
            self.tree_models.append(model)
            smooth = HistGradientBoostingRegressor(
                learning_rate=.08, max_iter=50, max_leaf_nodes=31, l2_regularization=.2,
                min_samples_leaf=20, random_state=self.seed + 100 + index,
            )
            smooth.fit(self.features[self.train_idx], self.outputs[self.train_idx, index])
            self.smooth_models.append(smooth)
        self.classifier = ExtraTreesClassifier(
            n_estimators=64, min_samples_leaf=4, max_features=.85, bootstrap=True, max_samples=.85,
            class_weight="balanced", n_jobs=1, random_state=self.seed,
        )
        self.classifier.fit(self.features[self.train_idx], self.feasible[self.train_idx])
        tree_mu, tree_sd, smooth_mu = self._component_prediction(self.features[self.test_idx])
        tree_mae = np.abs(tree_mu - self.outputs[self.test_idx]).mean(axis=0)
        smooth_mae = np.abs(smooth_mu - self.outputs[self.test_idx]).mean(axis=0)
        self.model_weights = 1.0 / np.maximum(tree_mae, 1e-6)
        self.model_weights /= self.model_weights + 1.0 / np.maximum(smooth_mae, 1e-6)
        holdout_mu, holdout_sd = self._blend_prediction(tree_mu, tree_sd, smooth_mu)
        self.residual_scale = np.quantile(np.abs(self.outputs[self.test_idx] - holdout_mu), .75, axis=0)
        holdout_mu, holdout_sd = self.model_prediction(self.features[self.test_idx])
        ratio = np.abs(self.outputs[self.test_idx, 3] - holdout_mu[:, 3]) / np.maximum(holdout_sd[:, 3], 1e-6)
        self.stress_k = float(max(1.96, np.quantile(ratio, .90)))
        self.stress_screen_margin = float(np.quantile(np.abs(self.outputs[self.test_idx, 3] - holdout_mu[:, 3]), .75))
        probabilities = self.classifier.predict_proba(self.features[self.test_idx])[:, 1]
        return {
            "rows": float(len(self.data)),
            "stress_ucb_coverage": float(np.mean(self.outputs[self.test_idx, 3] <= holdout_mu[:, 3] + self.stress_k * holdout_sd[:, 3])),
            "stress_mae_mpa": float(np.abs(holdout_mu[:, 3] - self.outputs[self.test_idx, 3]).mean()),
            "tree_stress_mae_mpa": float(tree_mae[3]),
            "smooth_stress_mae_mpa": float(smooth_mae[3]),
            "tree_blend_weight": float(self.model_weights[3]),
            "feasibility_roc_auc": float(roc_auc_score(self.feasible[self.test_idx], probabilities)),
            "feasibility_average_precision": float(average_precision_score(self.feasible[self.test_idx], probabilities)),
        }

    def _component_prediction(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        means, stds = [], []
        smooth = []
        for tree_model, smooth_model in zip(self.tree_models, self.smooth_models):
            members = np.vstack([tree.predict(features) for tree in tree_model.estimators_])
            means.append(members.mean(axis=0))
            stds.append(members.std(axis=0, ddof=1))
            smooth.append(smooth_model.predict(features))
        return np.column_stack(means), np.column_stack(stds), np.column_stack(smooth)

    def _blend_prediction(self, tree_mu: np.ndarray, tree_sd: np.ndarray, smooth_mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        weights = self.model_weights[None, :]
        mean = weights * tree_mu + (1.0 - weights) * smooth_mu
        disagreement = np.abs(tree_mu - smooth_mu)
        # Residual calibration prevents a low tree spread from being mistaken for certainty.
        std = np.sqrt((weights * tree_sd) ** 2 + (weights * (1.0 - weights) * disagreement) ** 2 + (0.10 * self.residual_scale[None, :]) ** 2)
        return mean, std

    def model_prediction(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._blend_prediction(*self._component_prediction(features))

    @staticmethod
    def feature_matrix(designs: np.ndarray, mission: dict[str, float | str]) -> tuple[np.ndarray, np.ndarray]:
        repaired = repair_design(np.atleast_2d(designs))
        flight = np.array([mission["altitude"], mission["kcas"], mission["aoa"]], dtype=float)
        return repaired, np.column_stack([repaired, np.broadcast_to(flight, (len(repaired), 3))])

    def predict_candidate(self, designs: np.ndarray, mission: dict[str, float | str], uncertainty: bool = True):
        repaired, features = self.feature_matrix(designs, mission)
        if uncertainty:
            mean, std = self.model_prediction(features)
        else:
            tree_mu, tree_sd, smooth_mu = self._component_prediction(features)
            mean, _ = self._blend_prediction(tree_mu, tree_sd, smooth_mu)
            std = None
        frame = pd.DataFrame(repaired, columns=DESIGN_COLUMNS)
        ld = predict_ld_batch(frame, alt_kft=mission["altitude"], kcas=mission["kcas"], aoa=mission["aoa"])
        assert self.classifier is not None
        probability = self.classifier.predict_proba(features)[:, 1]
        loss = self.official_loss(ld, mean, mission)
        return repaired, ld, mean, std, probability, loss

    @staticmethod
    def official_loss(ld: np.ndarray, values: np.ndarray, mission: dict[str, float | str]) -> np.ndarray:
        """Match the README's one-sided scoring terms; stress is handled as a constraint."""
        return (
            .4 * values[:, 0] / mission.get("mass_target", 50.0)
            + .2 * np.maximum(0., (mission["ld_target"] - ld) / mission["ld_target"])
            + .2 * np.maximum(0., (mission["payload_target"] - values[:, 1]) / mission["payload_target"])
            + .2 * np.maximum(0., (mission["fuel_target"] - values[:, 2]) / mission["fuel_target"])
        )

    def constraint_state(self, ld: np.ndarray, mean: np.ndarray, std: np.ndarray | None, probability: np.ndarray, mission: dict[str, float | str]) -> tuple[np.ndarray, np.ndarray]:
        stress_std = np.zeros(len(mean)) if std is None else std[:, 3]
        stress_limit = float(mission.get("stress_limit", STRESS_FINAL))
        deficits = np.column_stack([
            np.maximum(0., (mission["ld_target"] - ld) / mission["ld_target"]),
            np.maximum(0., (mission["payload_target"] - mean[:, 1]) / mission["payload_target"]),
            np.maximum(0., (mission["fuel_target"] - mean[:, 2]) / mission["fuel_target"]),
            np.maximum(0., (mean[:, 3] + self.stress_k * stress_std + self.stress_screen_margin - stress_limit) / 20.0),
            np.maximum(0., MIN_FEAS_PROB - probability),
        ])
        return deficits.sum(axis=1), np.all(deficits <= 1e-12, axis=1)

    def observed_loss(self, rows: np.ndarray, mission: dict[str, float | str]) -> np.ndarray:
        frame = pd.DataFrame(self.designs[rows], columns=DESIGN_COLUMNS)
        ld = predict_ld_batch(frame, alt_kft=mission["altitude"], kcas=mission["kcas"], aoa=mission["aoa"])
        values = self.outputs[rows]
        return self.official_loss(ld, values, mission)

    def score(self, design: np.ndarray, mission: dict[str, float | str]) -> float:
        _, ld, mean, std, probability, loss = self.predict_candidate(np.asarray(design)[None, :], mission, uncertainty=True)
        assert std is not None
        violation, _ = self.constraint_state(ld, mean, std, probability, mission)
        return float(CONSTRAINT_WEIGHT * violation[0] ** 2 + loss[0])

    def optimize(self, mission: dict[str, float | str], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # A full Sobol-initialized global stage makes seed changes much less likely to
        # select an unrelated local basin than the old one-generation search.
        result = differential_evolution(
            lambda z: self.score(LO + z * (HI - LO), mission), list(zip(np.zeros(len(LO)), np.ones(len(HI)))),
            init="sobol", seed=seed, popsize=DE_POPSIZE, maxiter=DE_MAXITER, tol=1e-4,
            polish=True, workers=1, updating="immediate",
        )
        de = repair_design(LO + result.x * (HI - LO))[None, :]
        _, de_ld, de_mean, de_std, de_probability, de_loss = self.predict_candidate(de, mission)
        assert de_std is not None
        _, de_feasible = self.constraint_state(de_ld, de_mean, de_std, de_probability, mission)
        incumbent = float(de_loss[de_feasible].min()) if de_feasible.any() else float(self.score(de[0], mission))
        bo = []
        # The global solution plus the best matching safe database basins give the
        # local GP phase multiple deterministic places to refine.
        safe_rows = np.flatnonzero(self.outputs[:, 3] <= float(mission["stress_limit"]))
        # Evaluate a bounded database sample rather than all 13k rows for every
        # mission; full-database evaluation is reserved for the actual fallback.
        sampled_rows = safe_rows[:min(1024, len(safe_rows))]
        sampled_loss = self.observed_loss(sampled_rows, mission)
        seed_rows = sampled_rows[np.argsort(sampled_loss)[:32]]
        centers = np.vstack([de, self.designs[seed_rows[:2]]])
        for index, center in enumerate(centers):
            rows = np.flatnonzero(self.outputs[:, 3] <= STRESS_FINAL)
            nearest = rows[np.argsort(((normalize(self.designs[rows]) - normalize(center)) ** 2).sum(axis=1))[:120]]
            local_loss = self.observed_loss(nearest, mission)
            gp = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(length_scale=.25) + WhiteKernel(noise_level=.02),
                normalize_y=True, optimizer=None, random_state=seed + index,
            ).fit(normalize(self.designs[nearest]), local_loss)
            center_normalized = normalize(center)
            bounds = list(zip(np.maximum(0, center_normalized - .12), np.minimum(1, center_normalized + .12)))

            lower, upper = np.asarray(bounds).T
            samples = qmc.Sobol(d=len(bounds), scramble=True, seed=seed + 100 + index).random_base2(int(np.log2(BO_SAMPLES)))
            local_candidates = lower + samples * (upper - lower)
            gp_mu, gp_sd = gp.predict(local_candidates, return_std=True)
            improvement = incumbent - gp_mu
            standardized = improvement / np.maximum(gp_sd, 1e-6)
            expected_improvement = improvement * ndtr(standardized) + gp_sd * np.exp(-.5 * standardized ** 2) / np.sqrt(2 * np.pi)
            designs_batch = repair_design(LO + local_candidates * (HI - LO))
            _, local_ld, local_mean, local_std, local_probability, local_loss = self.predict_candidate(designs_batch, mission, uncertainty=True)
            assert local_std is not None
            local_violation, _ = self.constraint_state(local_ld, local_mean, local_std, local_probability, mission)
            bo.append(designs_batch[np.argmin(-expected_improvement + CONSTRAINT_WEIGHT * local_violation ** 2 + local_loss)])
        # Retain only a compact set of database comparisons; they are warm starts,
        # not the optimizer's default answer.
        database_seeds = self.designs[seed_rows]
        raw_pool = np.vstack([database_seeds, de, np.asarray(bo)])
        raw_origins = np.array(
            ["database_seed"] * len(database_seeds)
            + ["differential_evolution"] * len(de)
            + ["gaussian_process"] * len(bo),
            dtype=object,
        )
        rounded_pool = np.round(raw_pool, 10)
        _, first_indices = np.unique(rounded_pool, axis=0, return_index=True)
        keep_indices = np.sort(first_indices)
        pool = rounded_pool[keep_indices]
        origins = raw_origins[keep_indices]
        designs, ld, mean, std, probability, loss = self.predict_candidate(pool, mission)
        assert std is not None
        violation, feasible = self.constraint_state(ld, mean, std, probability, mission)
        stress_margin = mean[:, 3] + self.stress_k * std[:, 3] - float(mission["stress_limit"])
        order = np.lexsort((loss, stress_margin, violation, ~feasible))[:TOP_K]
        return designs[order], ld[order], mean[order], std[order], probability[order], loss[order], origins[order]

    def database_fallback(self, mission: dict[str, float | str]) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float, float, str, int]:
        """Return the closest stress-safe database design for a mission, if one exists."""
        pool = np.flatnonzero(self.outputs[:, 3] <= float(mission["stress_limit"]))
        origin = "database_fallback"
        if not len(pool):
            pool = np.arange(len(self.designs))
            origin = "stress_unsafe_fallback"
        # First screen with database-reported mass/volume values, then evaluate L/D
        # only for the most mission-aligned rows. This keeps the fallback fast while
        # returning an actual stress-safe dataset row rather than a surrogate claim.
        values = self.outputs[pool]
        preliminary = (
            np.maximum(0., (mission["payload_target"] - values[:, 1]) / mission["payload_target"])
            + np.maximum(0., (mission["fuel_target"] - values[:, 2]) / mission["fuel_target"])
        )
        finalists = pool[np.argsort(preliminary)[:min(64, len(pool))]]
        frame = pd.DataFrame(self.designs[finalists], columns=DESIGN_COLUMNS)
        ld = predict_ld_batch(frame, alt_kft=mission["altitude"], kcas=mission["kcas"], aoa=mission["aoa"])
        mean = self.outputs[finalists]
        std = np.zeros_like(mean)
        probability = np.ones(len(finalists))
        loss = self.official_loss(ld, mean, mission)
        violation, _ = self.constraint_state(ld, mean, std, probability, mission)
        chosen_local = np.lexsort((loss, violation))[0]
        chosen = int(finalists[chosen_local])
        return self.designs[chosen], float(ld[chosen_local]), mean[chosen_local], std[chosen_local], float(probability[chosen_local]), float(loss[chosen_local]), origin, chosen

    def nearest_database_design(self, design: np.ndarray) -> dict[str, float | int | bool]:
        distances = np.linalg.norm(normalize(self.designs) - normalize(design), axis=1)
        index = int(np.argmin(distances))
        distance = float(distances[index])
        return {
            "nearest_dataset_row": index,
            "nearest_normalized_distance": distance,
            "exact_database_match": distance <= 1e-8,
        }


def base_case_id(case_id: str) -> str:
    """Remove robustness-scenario suffixes while leaving regular case IDs intact."""
    return case_id.split("__", 1)[0]


def stable_case_seed(case_id: str, root_seed: int) -> int:
    """Keep all variants of one base mission on an identical deterministic search path."""
    digest = hashlib.sha256(base_case_id(case_id).encode("utf-8")).digest()
    return int(root_seed + int.from_bytes(digest[:4], "big") % 1_000_000)


def candidate_for_case(optimizer: HybridOptimizer, case: dict[str, object], root_seed: int) -> tuple[dict[str, object], dict[str, object]]:
    mission = mission_for_model(case["mission"], str(case["name"]))
    search_seed = stable_case_seed(str(case["id"]), root_seed)
    designs, ld, mean, std, probability, loss, origins = optimizer.optimize(mission, search_seed)
    violation, feasible = optimizer.constraint_state(ld, mean, std, probability, mission)
    fallback_row: int | None = None
    if feasible.any():
        selected = int(np.flatnonzero(feasible)[0])
        selected_origin = str(origins[selected])
    else:
        design, ld_value, metric, uncertainty, feasible_probability, objective, selected_origin, fallback_row = optimizer.database_fallback(mission)
        violation, feasible = optimizer.constraint_state(
            np.array([ld_value]), metric[None, :], uncertainty[None, :], np.array([feasible_probability]), mission,
        )
        selected = None
    if selected is not None:
        design, ld_value, metric, uncertainty, feasible_probability, objective = (
            designs[selected], ld[selected], mean[selected], std[selected], probability[selected], loss[selected]
        )
    design_record = {key: float(value) for key, value in zip(DESIGN_COLUMNS, design)}
    candidate = {
        "case_id": case["id"],
        "mission": case["mission"],
        "design": design_record,
        "metrics": {
            "empty_mass_kg": float(metric[0]), "ld": float(ld_value), "payload_volume_m3": float(metric[1]),
            "fuel_volume_m3": float(metric[2]), "max_hotspot_stress_mpa": float(metric[3]),
        },
    }
    diagnostic = {
        "case_id": case["id"],
        "base_case_id": base_case_id(str(case["id"])),
        "search_seed": search_seed,
        "selected_origin": selected_origin,
        "optimizer_feasible": bool(feasible[0]),
        "constraint_violation": float(violation[0]),
        "objective": float(objective),
        "stress_uncertainty_mpa": float(uncertainty[3]),
        "stress_ucb_mpa": float(metric[3] + optimizer.stress_k * uncertainty[3]),
        "feasibility_probability": float(feasible_probability),
        "fallback_dataset_row": fallback_row,
        **optimizer.nearest_database_design(design),
    }
    return candidate, diagnostic


def main() -> int:
    global DE_MAXITER, DE_POPSIZE, BO_SAMPLES
    parser = argparse.ArgumentParser(description="Run the hybrid sklearn BWB optimizer for gauntlet cases.")
    parser.add_argument("--cases", type=Path, required=True, help="Gauntlet cases JSON")
    parser.add_argument("--output", type=Path, required=True, help="Candidate JSON destination")
    parser.add_argument("--seed", type=int, default=RNG_SEED, help="Root deterministic search seed")
    parser.add_argument("--diagnostics", type=Path, help="Optional sidecar JSON for search provenance")
    parser.add_argument("--de-maxiter", type=int, default=DE_MAXITER, help="DE generations per mission; raise for offline final searches")
    parser.add_argument("--de-popsize", type=int, default=DE_POPSIZE, help="DE population multiplier")
    parser.add_argument("--bo-samples", type=int, default=BO_SAMPLES, help="Power-of-two Sobol samples per GP basin")
    args = parser.parse_args()
    if args.de_maxiter < 0 or args.de_popsize < 1 or args.bo_samples < 2 or args.bo_samples & (args.bo_samples - 1):
        parser.error("--de-maxiter must be non-negative, --de-popsize positive, and --bo-samples a power of two >= 2")
    DE_MAXITER, DE_POPSIZE, BO_SAMPLES = args.de_maxiter, args.de_popsize, args.bo_samples
    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    optimizer = HybridOptimizer()
    validation = optimizer.fit()
    print("Structural model validation:", json.dumps(validation, sort_keys=True))
    results = [candidate_for_case(optimizer, case, args.seed) for case in payload["cases"]]
    candidates = [candidate for candidate, _ in results]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "candidates": candidates}, indent=2) + "\n", encoding="utf-8")
    if args.diagnostics:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics.write_text(
            json.dumps({"schema_version": 1, "root_seed": args.seed, "candidates": [diagnostic for _, diagnostic in results]}, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"Wrote {len(candidates)} candidates to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
