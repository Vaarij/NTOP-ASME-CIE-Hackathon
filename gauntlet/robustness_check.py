#!/usr/bin/env python3
"""Measure an optimizer's stress response to volume-target perturbations.

This command is intentionally sensitivity-only. It validates every returned
candidate using the gauntlet contract, but a stress failure is reported rather
than used as this command's exit condition.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

from score_candidates import (
    CASES_PATH,
    ContractError,
    check_negative_fixtures,
    json_safe,
    load_json,
    run_target,
    score_output,
)


def make_perturbed_cases(cases_payload: dict[str, Any], fraction: float) -> dict[str, Any]:
    """Expand each base mission into baseline and independent volume perturbations."""
    scenarios: list[dict[str, Any]] = []
    variants = (
        ("baseline", None, 1.0),
        ("payload_minus", "payload_volume_min_m3", 1.0 - fraction),
        ("payload_plus", "payload_volume_min_m3", 1.0 + fraction),
        ("fuel_minus", "fuel_volume_min_m3", 1.0 - fraction),
        ("fuel_plus", "fuel_volume_min_m3", 1.0 + fraction),
    )
    for base in cases_payload["cases"]:
        for suffix, field, multiplier in variants:
            mission = copy.deepcopy(base["mission"])
            if field is not None:
                mission[field] *= multiplier
            scenarios.append({
                "id": f"{base['id']}__{suffix}",
                "source": f"robustness check: {base['id']} {suffix}",
                "name": f"{base['name']} ({suffix.replace('_', ' ')})",
                "mission": mission,
            })
    return {"schema_version": cases_payload["schema_version"], "units": cases_payload.get("units", {}), "cases": scenarios}


def _scenario(report: dict[str, Any], case_id: str) -> dict[str, Any]:
    row = next(row for row in report["cases"] if row["case_id"] == case_id)
    return {"metrics": row["metrics"], "stress_status": row["score"]["status"], "stress_reason": row["score"]["reason"]}


def build_sensitivity_report(base_cases: dict[str, Any], scored: dict[str, Any], fraction: float) -> dict[str, Any]:
    """Summarize centered stress slopes and one-sided changes per base mission."""
    summaries = []
    for base in base_cases["cases"]:
        prefix = base["id"]
        baseline = _scenario(scored, f"{prefix}__baseline")
        payload_minus = _scenario(scored, f"{prefix}__payload_minus")
        payload_plus = _scenario(scored, f"{prefix}__payload_plus")
        fuel_minus = _scenario(scored, f"{prefix}__fuel_minus")
        fuel_plus = _scenario(scored, f"{prefix}__fuel_plus")

        def axis_summary(field: str, minus: dict[str, Any], plus: dict[str, Any]) -> dict[str, Any]:
            nominal = float(base["mission"][field])
            baseline_stress = float(baseline["metrics"]["max_hotspot_stress_mpa"])
            minus_stress = float(minus["metrics"]["max_hotspot_stress_mpa"])
            plus_stress = float(plus["metrics"]["max_hotspot_stress_mpa"])
            denominator = 2.0 * fraction * nominal
            return {
                "target_field": field,
                "baseline_target_m3": nominal,
                "minus_target_m3": nominal * (1.0 - fraction),
                "plus_target_m3": nominal * (1.0 + fraction),
                "minus": minus,
                "plus": plus,
                "stress_delta_minus_mpa": minus_stress - baseline_stress,
                "stress_delta_plus_mpa": plus_stress - baseline_stress,
                "central_stress_sensitivity_mpa_per_m3": (plus_stress - minus_stress) / denominator,
            }

        summaries.append({
            "case_id": prefix,
            "name": base["name"],
            "baseline": baseline,
            "payload": axis_summary("payload_volume_min_m3", payload_minus, payload_plus),
            "fuel": axis_summary("fuel_volume_min_m3", fuel_minus, fuel_plus),
        })
    return {
        "schema_version": 1,
        "measurement": "optimizer-reported local response to independent volume-target perturbations",
        "perturbation_fraction": fraction,
        "cases": summaries,
    }


def print_report(report: dict[str, Any]) -> None:
    percent = report["perturbation_fraction"] * 100.0
    for row in report["cases"]:
        baseline = row["baseline"]["metrics"]["max_hotspot_stress_mpa"]
        print(f"{row['case_id']}: baseline stress={baseline:.3f} MPa ({row['baseline']['stress_status']})")
        for label in ("payload", "fuel"):
            axis = row[label]
            print(
                f"  {label:7} ±{percent:g}%: "
                f"Δstress(-)={axis['stress_delta_minus_mpa']:+.3f} MPa, "
                f"Δstress(+)={axis['stress_delta_plus_mpa']:+.3f} MPa, "
                f"central slope={axis['central_stress_sensitivity_mpa_per_m3']:+.3f} MPa/m³"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure stress sensitivity to independent payload/fuel target changes.")
    parser.add_argument("--target", type=Path, required=True, help="Optimizer Python script that accepts --cases and --output")
    parser.add_argument("--percent", type=float, default=10.0, help="Independent plus/minus target change in percent (default: 10)")
    parser.add_argument("--timeout", type=float, default=300.0, help="Target timeout in seconds (default: 300)")
    parser.add_argument("--report", type=Path, help="Optional JSON destination for the sensitivity report")
    args = parser.parse_args()
    if not 0.0 < args.percent < 100.0:
        parser.error("--percent must be between 0 and 100")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    try:
        check_negative_fixtures()
        base_cases = load_json(CASES_PATH)
        fraction = args.percent / 100.0
        perturbed_cases = make_perturbed_cases(base_cases, fraction)
        with tempfile.TemporaryDirectory(prefix="bwb_robustness_") as directory:
            directory_path = Path(directory)
            cases_path = directory_path / "perturbed_cases.json"
            output_path = directory_path / "candidates.json"
            cases_path.write_text(json.dumps(perturbed_cases, indent=2) + "\n", encoding="utf-8")
            run_target(args.target, cases_path, output_path, args.timeout)
            scored = score_output(load_json(output_path), perturbed_cases)
        report = build_sensitivity_report(base_cases, scored, fraction)
        print_report(report)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return 0
    except ContractError as exc:
        print(f"ROBUSTNESS ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
