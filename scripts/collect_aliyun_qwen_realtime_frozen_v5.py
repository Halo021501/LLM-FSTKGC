#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List


SHOTS = (1, 3, 5, 10)
SEEDS = (42, 43, 44)
RANKING_MODES = ("confidence", "score", "rationale")
FROZEN_MODES = ("candidate", "score", "rationale")
METRIC_KEYS = ("mrr", "hits1", "hits3", "hits10")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def mean_std(values: List[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def collect(args: argparse.Namespace) -> Dict[str, Any]:
    run_root = Path(args.run_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    missing: List[str] = []

    diagnostic_rows: List[Dict[str, Any]] = []
    for shot in SHOTS:
        for mode in RANKING_MODES:
            path = run_root / "llm_only" / f"test_s{shot}_{mode}.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            result = load_json(path)
            metrics = result["metrics"]
            diagnostic_rows.append(
                {
                    "shot": shot,
                    "ranking_mode": mode,
                    "queries": metrics["queries"],
                    "cache_hit_rate": metrics["cache_hit_rate"],
                    "candidate_recall_at_10": metrics["candidate_recall_at_10"],
                    "mapping_rate": metrics["mapping_rate"],
                    "hallucination_rate": metrics["hallucination_rate"],
                    "tie_avg_mrr": metrics["tie_avg_mrr"],
                    "tie_avg_hits1": metrics["tie_avg_hits1"],
                    "tie_avg_hits3": metrics["tie_avg_hits3"],
                    "tie_avg_hits10": metrics["tie_avg_hits10"],
                    "avg_latency_ms": metrics["avg_latency_ms"],
                    "avg_prompt_tokens": metrics["avg_prompt_tokens"],
                    "avg_completion_tokens": metrics["avg_completion_tokens"],
                    "cache_sha256": result["protocol"]["cache_sha256"],
                }
            )

    frozen_rows: List[Dict[str, Any]] = []
    baseline_by_key: Dict[tuple[int, int], Dict[str, float]] = {}
    for shot in SHOTS:
        for seed in SEEDS:
            baseline_path = checkpoint_root / f"main_s{shot}_seed{seed}" / "metrics.json"
            if not baseline_path.is_file():
                missing.append(str(baseline_path))
                continue
            baseline_metrics = load_json(baseline_path)
            baseline = {key: float(baseline_metrics[f"test_tie_avg_{key}"]) for key in METRIC_KEYS}
            baseline_by_key[(shot, seed)] = baseline
            frozen_rows.append(
                {
                    "shot": shot,
                    "seed": seed,
                    "mode": "off",
                    **baseline,
                    "candidate_recall_at_10": 0.0,
                    "cache_hit_rate": 0.0,
                    "elapsed_seconds": baseline_metrics.get("elapsed_seconds", ""),
                    "peak_gpu_reserved_mb": baseline_metrics.get("peak_gpu_reserved_mb", ""),
                    "source": str(baseline_path),
                }
            )
            for mode in FROZEN_MODES:
                out_dir = run_root / "frozen" / f"frozen_s{shot}_seed{seed}_{mode}"
                metrics_path = out_dir / "metrics.json"
                meta_path = out_dir / "run_meta.json"
                exit_path = out_dir / "exit_code.txt"
                if not (metrics_path.is_file() and meta_path.is_file() and exit_path.is_file()):
                    missing.append(str(out_dir))
                    continue
                if exit_path.read_text(encoding="utf-8").strip() != "0":
                    raise ValueError(f"non-zero frozen task: {out_dir}")
                metrics = load_json(metrics_path)
                meta = load_json(meta_path)
                if meta["model_config"]["llm_mode"] != mode:
                    raise ValueError(f"mode mismatch: {out_dir}")
                if not bool(meta["llm"]["frozen_parent_evaluation"]):
                    raise ValueError(f"not a frozen parent evaluation: {out_dir}")
                if float(meta["llm"]["test_cache_coverage"]["cache_hit_rate"]) != 1.0:
                    raise ValueError(f"incomplete test cache coverage: {out_dir}")
                frozen_rows.append(
                    {
                        "shot": shot,
                        "seed": seed,
                        "mode": mode,
                        **{key: float(metrics[f"test_tie_avg_{key}"]) for key in METRIC_KEYS},
                        "candidate_recall_at_10": float(metrics["test_llm_candidate_recall_at_10"]),
                        "cache_hit_rate": float(metrics["test_llm_cache_hit_rate"]),
                        "elapsed_seconds": metrics.get("elapsed_seconds", ""),
                        "peak_gpu_reserved_mb": metrics.get("peak_gpu_reserved_mb", ""),
                        "source": str(metrics_path),
                    }
                )

    if missing and not args.allow_incomplete:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"missing {len(missing)} required artifacts:\n{preview}")

    grouped: List[Dict[str, Any]] = []
    for shot in SHOTS:
        for mode in ("off",) + FROZEN_MODES:
            rows = [row for row in frozen_rows if row["shot"] == shot and row["mode"] == mode]
            if not rows:
                continue
            summary: Dict[str, Any] = {"shot": shot, "mode": mode, "n": len(rows)}
            for key in METRIC_KEYS:
                mean, std = mean_std([float(row[key]) for row in rows])
                summary[f"{key}_mean"] = mean
                summary[f"{key}_std"] = std
            grouped.append(summary)

    paired_rows: List[Dict[str, Any]] = []
    for row in frozen_rows:
        if row["mode"] == "off":
            continue
        baseline = baseline_by_key[(int(row["shot"]), int(row["seed"]))]
        paired_rows.append(
            {
                "shot": row["shot"],
                "seed": row["seed"],
                "mode": row["mode"],
                **{f"delta_{key}": float(row[key]) - baseline[key] for key in METRIC_KEYS},
            }
        )

    decisions: List[Dict[str, Any]] = []
    summary_by_key = {(row["shot"], row["mode"]): row for row in grouped}
    for shot in (5, 10):
        base = summary_by_key.get((shot, "off"))
        for mode in FROZEN_MODES:
            active = summary_by_key.get((shot, mode))
            complete = bool(base and active and int(base["n"]) == 3 and int(active["n"]) == 3)
            passes = bool(
                complete
                and active["mrr_mean"] > base["mrr_mean"]
                and active["hits10_mean"] >= base["hits10_mean"] - 0.003
            )
            decisions.append(
                {
                    "shot": shot,
                    "mode": mode,
                    "complete_three_seed_pair": complete,
                    "mean_mrr_improved": bool(complete and active["mrr_mean"] > base["mrr_mean"]),
                    "hits10_drop_within_0_003": bool(
                        complete and active["hits10_mean"] >= base["hits10_mean"] - 0.003
                    ),
                    "paper_gate_passed": passes,
                }
            )

    atomic_csv(run_root / "diagnostics_summary.csv", diagnostic_rows, list(diagnostic_rows[0]) if diagnostic_rows else [])
    atomic_csv(run_root / "frozen_raw.csv", frozen_rows, list(frozen_rows[0]) if frozen_rows else [])
    atomic_csv(run_root / "frozen_mean_std.csv", grouped, list(grouped[0]) if grouped else [])
    paired_fields = list(paired_rows[0]) if paired_rows else ["shot", "seed", "mode"]
    atomic_csv(run_root / "paired_delta.csv", paired_rows, paired_fields)
    result = {
        "provider": "aliyun_qwen_realtime",
        "model": "qwen-flash",
        "history_protocol": "standard_rolling_history",
        "diagnostic_runs_complete": len(diagnostic_rows) == len(SHOTS) * len(RANKING_MODES),
        "frozen_active_runs_complete": len([row for row in frozen_rows if row["mode"] != "off"])
        == len(SHOTS) * len(SEEDS) * len(FROZEN_MODES),
        "missing_artifacts": missing,
        "paper_gate": decisions,
        "notes": [
            "s5/s10 are the taskbook primary conditions; s1/s3 are supplementary robustness conditions.",
            "LLM-only sparse-candidate ranks are diagnostics and are not a dense-model replacement.",
            "Off rows reuse the validation-selected v5 parent runs under the matching shot and seed.",
            "The current caches use rolling history only; static metrics in active frozen runs have LLM disabled.",
        ],
    }
    atomic_json(run_root / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Aliyun realtime LLM-only and frozen-v5 results")
    parser.add_argument(
        "--run-root",
        default="runs/aliyun_qwen_realtime_qwen_flash_20260809",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints/alterego_v5",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


if __name__ == "__main__":
    collect(build_parser().parse_args())
