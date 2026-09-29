#!/usr/bin/env python3
"""Paired semantic-off/rationale evaluation of one frozen checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("ICEWS14", "YAGO"), required=True)
    parser.add_argument("--shot", type=int, choices=(5, 10), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gate-diagnostics", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="debug-only number of original test rows")
    parser.add_argument(
        "--reuse-off-run", type=Path, default=None,
        help="reuse the completed training run's semantic-off test and evaluate rationale only",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.code_root))
    import torch
    from src.data import HistoryIndex, add_inverse, build_filter_dict, load_temporal_kg
    from src.llm_cache import LLMEvidenceCache, dataset_files_fingerprint
    from src.model import NineFuseTKG
    from src.train import evaluate, group_by_relation, set_seed

    started = time.monotonic()
    set_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["args"]
    if int(config["shot"]) != args.shot or int(config["seed"]) != args.seed:
        raise ValueError("checkpoint shot/seed does not match requested paired evaluation")
    kg = load_temporal_kg(str(args.data_dir))
    model = NineFuseTKG(
        kg.num_entities,
        kg.num_relations * 2,
        kg.num_times,
        dim=config["dim"],
        history_len=config["history_len"],
        channels=config["channels"],
        dropout=config["dropout"],
        use_copy=not config.get("disable_copy", False),
        use_rel_copy=not config.get("disable_rel_copy", False),
        use_rule=not config.get("disable_rule", False),
        use_geo=config.get("enable_geo", False) and not config.get("disable_geo", False),
        use_freq=not config.get("disable_freq", False),
        use_history=not config.get("disable_history", False),
        use_support=not config.get("disable_support", False),
        use_support_gate=not config.get("disable_support_gate", False),
        use_router=not config.get("disable_router", False),
        use_temporal_calibration=not config.get("disable_temporal_calibration", False),
        use_snapshot_backbone=not config.get("disable_snapshot_backbone", False),
        use_candidate_rerank=not config.get("disable_candidate_rerank", False),
        use_alterego_tournament=not config.get("disable_alterego_tournament", False),
        alterego_candidate_k=config.get("alterego_candidate_k", 96),
        alterego_tournament_rank=config.get("alterego_tournament_rank", 32),
        alterego_max_delta=config.get("alterego_max_delta", 0.5),
        llm_mode="rationale",
        llm_max_candidates=config.get("llm_max_candidates", 10),
        llm_max_delta=config.get("llm_max_delta", 0.35),
        llm_score_scale=config.get("llm_score_scale", 1.0),
        llm_disable_confidence=config.get("llm_disable_confidence", False),
        expert_dropout=config.get("expert_dropout", 0.05),
        max_residual_gate=config.get("max_residual_gate", 0.8),
        max_expert_mass=config.get("max_expert_mass", 0.65),
        fusion_mode=config.get("fusion_mode", "probability_mixture"),
        llm_residual_control=config.get("llm_residual_control", "hard"),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    cache = LLMEvidenceCache(
        str(args.cache),
        max_candidates=config.get("llm_max_candidates", 10),
        expected_shot=args.shot,
        expected_history_protocol="standard_rolling_history",
        expected_split="test",
        expected_dataset_fingerprint=dataset_files_fingerprint(str(args.data_dir)),
        require_generation_metadata=True,
    )
    history = HistoryIndex(
        kg.train_aug + kg.valid_aug,
        kg.num_entities,
        kg.num_relations * 2,
        config["history_len"],
        adaptive_decay=not config.get("disable_temporal_calibration", False),
    )
    filters = build_filter_dict(add_inverse(kg.train + kg.valid + kg.test, kg.num_relations))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trace: list[dict[str, float]] = []

    def gate_hook(module, inputs, output):
        gate, reliability = output
        support = inputs[4]
        before = torch.linalg.vector_norm(support, dim=-1)
        after = torch.linalg.vector_norm(support * gate, dim=-1)
        for g, rel, b, a in zip(gate.detach().cpu().flatten(), reliability.detach().cpu().flatten(), before.detach().cpu(), after.detach().cpu()):
            trace.append({
                "support_gate": float(g),
                "support_reliability": float(rel),
                "support_norm_before": float(b),
                "support_norm_after": float(a),
                "support_attenuation": float(1.0 - g),
            })

    hook = model.support_gate.register_forward_hook(gate_hook) if args.gate_diagnostics else None
    results = {}
    paths = {}
    expected_queries = 2 * (min(args.limit, len(kg.test)) if args.limit > 0 else len(kg.test))
    modes = ("off", "rationale")
    if args.reuse_off_run is not None:
        if args.gate_diagnostics:
            raise ValueError("gate diagnostics require a fresh semantic-off pass")
        old = json.loads((args.reuse_off_run / "metrics.json").read_text(encoding="utf-8"))
        results["off"] = {
            key[len("test_"):]: value for key, value in old.items()
            if key.startswith("test_") and not key.startswith("test_static_")
        }
        paths["off"] = args.reuse_off_run / "query_diagnostics" / "test_queries.jsonl"
        if int(results["off"].get("evaluated_query_count", -1)) != expected_queries or not paths["off"].is_file():
            raise RuntimeError("reused semantic-off run lacks complete metrics or query export")
        modes = ("rationale",)
    for mode in modes:
        trace_start = len(trace)
        model.set_llm_runtime_mode(mode)
        export = args.output_dir / f"{mode}_test_queries.jsonl"
        tick = time.monotonic()
        metrics = evaluate(
            model,
            kg.test,
            history,
            group_by_relation(kg.train_aug),
            kg.train_aug,
            filters,
            kg.num_relations,
            args.shot,
            device,
            limit=args.limit,
            eval_batch_size=512,
            history_protocol="standard_rolling_history",
            llm_cache=cache if mode == "rationale" else None,
            query_export_path=str(export),
        )
        if int(metrics["evaluated_query_count"]) != expected_queries:
            raise RuntimeError(f"incomplete evaluation: {metrics['evaluated_query_count']} != {expected_queries}")
        if mode == "rationale" and metrics["llm_cache_hit_rate"] != 1.0:
            raise RuntimeError("rationale evaluation did not obtain full semantic-cache coverage")
        results[mode] = metrics
        paths[mode] = export
        print(json.dumps({"mode": mode, "mrr": metrics["tie_avg_mrr"], "queries": expected_queries,
                          "gate_trace_rows": len(trace) - trace_start,
                          "elapsed_seconds": time.monotonic() - tick}), flush=True)
    if hook is not None:
        hook.remove()

    gate_summary = None
    if args.gate_diagnostics:
        # Gate values are independent of the semantic runtime switch; retain
        # the first complete evaluation and join it to both rank records.
        off_trace = trace[:expected_queries]
        off_records = [json.loads(line) for line in paths["off"].read_text(encoding="utf-8").splitlines()]
        rationale_records = [json.loads(line) for line in paths["rationale"].read_text(encoding="utf-8").splitlines()]
        if not (len(off_trace) == len(off_records) == len(rationale_records) == expected_queries):
            raise RuntimeError("gate trace and query exports are not aligned")
        diagnostic_path = args.output_dir / "gate_query_diagnostics.jsonl"
        values = []
        damaged = []
        with diagnostic_path.open("w", encoding="utf-8") as handle:
            for gate_row, off_row, semantic_row in zip(off_trace, off_records, rationale_records):
                locator = (off_row["subject"], off_row["relation"], off_row["timestamp"], off_row["direction"])
                other = (semantic_row["subject"], semantic_row["relation"], semantic_row["timestamp"], semantic_row["direction"])
                if locator != other:
                    raise RuntimeError("off/rationale query exports differ in order")
                joined = {
                    "subject": off_row["subject"],
                    "relation": off_row["relation"],
                    "timestamp": off_row["timestamp"],
                    "direction": off_row["direction"],
                    "off_tie_avg_rank": off_row["tie_avg_rank"],
                    "rationale_tie_avg_rank": semantic_row["tie_avg_rank"],
                    "rank_change_rationale_minus_off": semantic_row["tie_avg_rank"] - off_row["tie_avg_rank"],
                    **gate_row,
                }
                values.append(gate_row["support_gate"])
                if joined["rank_change_rationale_minus_off"] > 0:
                    damaged.append(gate_row["support_attenuation"])
                handle.write(json.dumps(joined, ensure_ascii=False, sort_keys=True) + "\n")
        ordered = sorted(values)
        gate_summary = {
            "query_count": len(values),
            "gate_min": ordered[0],
            "gate_q25": ordered[len(ordered) // 4],
            "gate_median": ordered[len(ordered) // 2],
            "gate_q75": ordered[(3 * len(ordered)) // 4],
            "gate_max": ordered[-1],
            "gate_mean": sum(values) / len(values),
            "mean_attenuation_rank_damaged_queries": sum(damaged) / len(damaged) if damaged else None,
            "rank_damaged_query_count": len(damaged),
            "per_query_path": str(diagnostic_path),
        }

    report = {
        "dataset": args.dataset,
        "shot": args.shot,
        "seed": args.seed,
        "support_gate_enabled": not config.get("disable_support_gate", False),
        "semantic_off": results["off"],
        "rationale": results["rationale"],
        "delta_mrr_rationale_minus_off": results["rationale"]["tie_avg_mrr"] - results["off"]["tie_avg_mrr"],
        "gate_diagnostics": gate_summary,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "run_meta.json").write_text(json.dumps({
        "command_argv": [sys.executable, *sys.argv],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "cache": str(args.cache),
        "cache_sha256": sha256(args.cache),
        "model_code_sha256": sha256(args.code_root / "src/model.py"),
        "checkpoint_selection": "semantic-off validation tie-average MRR",
        "design": "same frozen checkpoint; paired complete-test semantic runtime intervention",
        "reused_semantic_off_run": str(args.reuse_off_run) if args.reuse_off_run else None,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "delta_mrr": report["delta_mrr_rationale_minus_off"]}))


if __name__ == "__main__":
    main()
