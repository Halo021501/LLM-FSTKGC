#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import Quad, add_inverse, build_filter_dict, load_temporal_kg
from src.llm_cache import LLMEvidenceCache, dataset_files_fingerprint, query_locator
from src.train import summarize_ranks


def candidate_score(candidate: Dict[str, object], mode: str) -> float:
    confidence = float(candidate.get("confidence", 0.0))
    mapping = float(candidate.get("mapping_score", 0.0))
    template = float(candidate.get("template_agreement", 0.0))
    temporal = float(candidate.get("temporal_score", 0.0))
    if mode == "confidence":
        return confidence * mapping
    if mode == "score":
        return mapping * (0.55 * confidence + 0.25 * template + 0.20 * mapping)
    return mapping * (0.45 * confidence + 0.20 * template + 0.15 * mapping + 0.20 * temporal)


def oriented_rows(rows: Sequence[Quad], num_relations: int) -> List[Quad]:
    return list(rows) + [(o, relation + num_relations, s, timestamp) for s, relation, o, timestamp in rows]


def run(args: argparse.Namespace) -> Dict[str, float]:
    kg = load_temporal_kg(args.data_dir)
    cache = LLMEvidenceCache(
        args.cache,
        max_candidates=10,
        expected_shot=args.shot,
        expected_history_protocol=args.history_protocol,
        expected_split=args.split,
        expected_dataset_fingerprint=dataset_files_fingerprint(args.data_dir),
    )
    rows = oriented_rows({"valid": kg.valid, "test": kg.test}[args.split], kg.num_relations)
    if args.limit > 0:
        rows = rows[: args.limit]
    filters = build_filter_dict(add_inverse(kg.train + kg.valid + kg.test, kg.num_relations))

    optimistic_ranks: List[float] = []
    tie_average_ranks: List[float] = []
    pessimistic_ranks: List[float] = []
    candidate_hits = 0
    cache_hits = 0
    mapped_candidates = 0
    raw_candidates = 0
    latency_ms = 0.0
    prompt_tokens = 0
    completion_tokens = 0

    for known, relation, target, timestamp in rows:
        record = cache.records.get(query_locator(known, relation, timestamp))
        filtered = filters.get((known, relation, timestamp), set())
        candidates: List[Tuple[int, float, int]] = []
        if record is not None:
            cache_hits += 1
            latency_ms += float(record.get("latency_ms", 0.0))
            usage = record.get("token_usage", {})
            prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            raw_candidates += len(record.get("candidates", []))
            for source_rank, candidate in enumerate(record.get("candidates", []), start=1):
                mapped_id = candidate.get("mapped_entity_id")
                if mapped_id is None:
                    continue
                mapped_candidates += 1
                mapped_id = int(mapped_id)
                if mapped_id != target and mapped_id in filtered:
                    continue
                candidates.append((mapped_id, candidate_score(candidate, args.ranking_mode), source_rank))
        # Stable source rank resolves equal calibrated scores without target use.
        candidates.sort(key=lambda item: (-item[1], item[2], item[0]))
        unique = []
        seen = set()
        for candidate_id, score, source_rank in candidates:
            if candidate_id not in seen:
                seen.add(candidate_id)
                unique.append((candidate_id, score, source_rank))
        target_positions = [index + 1 for index, item in enumerate(unique) if item[0] == target]
        if target_positions:
            rank = float(target_positions[0])
            candidate_hits += int(rank <= 10)
            optimistic_ranks.append(rank)
            tie_average_ranks.append(rank)
            pessimistic_ranks.append(rank)
        else:
            known_ranked = len(unique)
            filtered_unranked = len([answer for answer in filtered if answer != target and answer not in seen])
            unranked = max(1, kg.num_entities - known_ranked - filtered_unranked)
            first = float(known_ranked + 1)
            last = float(known_ranked + unranked)
            optimistic_ranks.append(first)
            tie_average_ranks.append(0.5 * (first + last))
            pessimistic_ranks.append(last)

    count = len(rows)
    metrics: Dict[str, float] = {
        "queries": float(count),
        "cache_hit_rate": cache_hits / max(1, count),
        "candidate_recall_at_10": candidate_hits / max(1, count),
        "mapping_rate": mapped_candidates / max(1, raw_candidates),
        "unmapped_rate": (raw_candidates - mapped_candidates) / max(1, raw_candidates),
        "avg_latency_ms": latency_ms / max(1, cache_hits),
        "avg_prompt_tokens": prompt_tokens / max(1, cache_hits),
        "avg_completion_tokens": completion_tokens / max(1, cache_hits),
    }
    metrics.update({f"optimistic_{key}": value for key, value in summarize_ranks(optimistic_ranks).items()})
    metrics.update({f"tie_avg_{key}": value for key, value in summarize_ranks(tie_average_ranks).items()})
    metrics.update({f"pessimistic_{key}": value for key, value in summarize_ranks(pessimistic_ranks).items()})
    result = {
        "protocol": {
            "model": "LLM-only mapped sparse candidate ranking",
            "split": args.split,
            "shot": args.shot,
            "history_protocol": args.history_protocol,
            "ranking_mode": args.ranking_mode,
            "unlisted_entities": "average-tie primary; optimistic and pessimistic bounds also reported",
            "cache_sha256": cache.sha256,
        },
        "metrics": metrics,
    }
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an LLM-only target-blind cache baseline")
    parser.add_argument("--data-dir", default="data/ICEWS14")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--split", choices=["valid", "test"], required=True)
    parser.add_argument("--shot", type=int, choices=[1, 3, 5, 10], required=True)
    parser.add_argument(
        "--history-protocol",
        choices=["standard_rolling_history", "strict_static_history"],
        default="standard_rolling_history",
    )
    parser.add_argument("--ranking-mode", choices=["confidence", "score", "rationale"], default="confidence")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
