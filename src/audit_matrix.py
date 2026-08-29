from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SHOTS = (1, 3, 5, 10)
SEEDS = (42, 43, 44)
MAIN_RUNS = [f"main_s{shot}_seed{seed}" for shot in SHOTS for seed in SEEDS]
ABLATIONS = [
    "ablate_copy_s5_seed42",
    "ablate_rule_s5_seed42",
    "ablate_snapshot_backbone_s5_seed42",
    "ablate_freq_s5_seed42",
    "ablate_history_s5_seed42",
    "ablate_support_s5_seed42",
    "ablate_router_s5_seed42",
    "ablate_rel_copy_s5_seed42",
    "ablate_temporal_calibration_s5_seed42",
    "ablate_candidate_rerank_s5_seed42",
    "ablate_alterego_tournament_s5_seed42",
    "strict_static_history_s5_seed42",
    "ablate_probability_mixture_s5_seed42",
]
EXPECTED = MAIN_RUNS + ABLATIONS


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit completeness, stability and MRR gate for 1.7.0alterego_v5")
    parser.add_argument("run_root")
    parser.add_argument("--low-memory-mb", type=float, default=256.0)
    parser.add_argument("--high-memory-mb", type=float, default=8192.0)
    parser.add_argument("--mrr-target", type=float, default=0.50)
    args = parser.parse_args()
    root = Path(args.run_root)

    runs = {}
    failures = []
    resource_warnings = []
    for name in EXPECTED:
        directory = root / name
        required = [directory / "best.pt", directory / "metrics.json", directory / "run_meta.json", directory / "exit_code.txt"]
        missing = [path.name for path in required if not path.exists()]
        if missing:
            failures.append(f"{name}: missing {', '.join(missing)}")
            continue
        exit_code = (directory / "exit_code.txt").read_text(encoding="utf-8").strip()
        metrics = load_json(directory / "metrics.json")
        meta = load_json(directory / "run_meta.json")
        peak = float(metrics.get("peak_gpu_allocated_mb", 0.0))
        problems = []
        if exit_code != "0":
            problems.append(f"exit={exit_code}")
        if not args.low_memory_mb <= peak <= args.high_memory_mb:
            resource_warnings.append(f"{name}: peak_allocated={peak:.1f}MB outside tuning target")
        if meta.get("checkpoint_precision") != "fp32":
            problems.append("checkpoint is not declared fp32")
        if meta.get("version") != "1.7.0alterego_v5":
            problems.append(f"unexpected version={meta.get('version')}")
        if int(meta.get("warmup", {}).get("batches_per_epoch", -1)) != 0:
            problems.append("warmup is not a complete fact-balanced epoch")
        if problems:
            failures.append(f"{name}: " + "; ".join(problems))
        runs[name] = {
            "test_tie_avg_mrr": float(metrics.get("test_tie_avg_mrr", 0.0)),
            "test_mrr": float(metrics.get("test_mrr", 0.0)),
            "peak_gpu_allocated_mb": peak,
            "parameter_count": int(float(metrics.get("parameter_count", 0))),
        }

    shot_stats = {}
    all_main = []
    for shot in SHOTS:
        values = [runs[name]["test_tie_avg_mrr"] for name in MAIN_RUNS if name.startswith(f"main_s{shot}_") and name in runs]
        shot_stats[str(shot)] = {
            "completed": len(values),
            "mean": statistics.mean(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "target_reached_all_seeds": len(values) == len(SEEDS) and all(value >= args.mrr_target for value in values),
        }
        all_main.extend(values)

    payload = {
        "expected_tasks": len(EXPECTED),
        "completed_tasks": len(runs),
        "failures": failures,
        "resource_warnings": resource_warnings,
        "memory_range_mb": [args.low_memory_mb, args.high_memory_mb],
        "mrr_target": args.mrr_target,
        "main_overall_mean": statistics.mean(all_main) if all_main else None,
        "main_target_reached_all": len(all_main) == len(MAIN_RUNS) and all(value >= args.mrr_target for value in all_main),
        "shots": shot_stats,
        "runs": runs,
        "acceptance_passed": not failures and len(runs) == len(EXPECTED),
    }
    (root / "audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# NineFuseTKG 1.7.0alterego_v5 Formal Matrix Audit",
        "",
        f"- Completed: {len(runs)}/{len(EXPECTED)}",
        f"- Resource/structure acceptance: {'PASS' if payload['acceptance_passed'] else 'FAIL'}",
        f"- Memory tuning target warnings: {len(resource_warnings)}",
        f"- All-main MRR >= {args.mrr_target:.2f}: {'PASS' if payload['main_target_reached_all'] else 'FAIL'}",
        f"- Main overall mean: {payload['main_overall_mean'] if payload['main_overall_mean'] is not None else 'N/A'}",
        "",
        "## Shot summary",
        "",
        "| Shot | Seeds | Mean | Min | Max | All seeds pass |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for shot in SHOTS:
        item = shot_stats[str(shot)]
        def render(value):
            return "N/A" if value is None else f"{value:.6f}"
        lines.append(
            f"| {shot} | {item['completed']}/3 | {render(item['mean'])} | {render(item['min'])} | "
            f"{render(item['max'])} | {'yes' if item['target_reached_all_seeds'] else 'no'} |"
        )
    lines.extend(["", "## Failures", ""])
    lines.extend([f"- {failure}" for failure in failures] or ["- None"])
    lines.extend(["", "## Resource warnings", ""])
    lines.extend([f"- {warning}" for warning in resource_warnings] or ["- None"])
    (root / "AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("completed_tasks", "acceptance_passed", "main_overall_mean", "main_target_reached_all")}, indent=2))
    if not payload["acceptance_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
