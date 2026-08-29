#!/usr/bin/env python3
"""Strictly merge collected Qwen realtime cache shards offline."""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import stlp_aliyun_qwen_batch as batch_cli
from src.aliyun_qwen_batch import (
    canonical_json,
    jsonl_bytes,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    write_bytes_atomic,
    write_json_atomic,
)
from src.data import load_temporal_kg
from src.llm_cache import LLMEvidenceCache, dataset_files_fingerprint, target_blind_query_key


UTC = dt.timezone.utc


def _sum_dict(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, int]:
    return {key: sum(int(row.get(key, 0)) for row in rows) for key in keys}


def merge(args: argparse.Namespace) -> dict[str, object]:
    data_dir = Path(args.data_dir).resolve()
    output = Path(args.output).resolve()
    meta_output = Path(str(output) + ".meta.json")
    source_dirs = [Path(path).resolve() for path in args.source_job_dir]
    part_caches = [Path(path).resolve() for path in args.part_cache]
    if len(source_dirs) != len(part_caches) or not source_dirs:
        raise ValueError("provide the same non-zero number of source jobs and part caches")
    if output.exists() or meta_output.exists():
        raise FileExistsError(f"refusing to overwrite merged cache: {output}")

    dataset_fingerprint = dataset_files_fingerprint(str(data_dir))
    plans: list[dict[str, Any]] = []
    indexes_by_id: dict[str, dict[str, Any]] = {}
    part_rows: list[dict[str, Any]] = []
    part_metas: list[dict[str, Any]] = []
    seen_shards: set[int] = set()
    common_parent: tuple[Any, ...] | None = None

    for source_dir, part_cache in zip(source_dirs, part_caches):
        plan, _, index_rows = batch_cli._verify_job(source_dir)
        partition = plan.get("partition")
        if not isinstance(partition, dict):
            raise ValueError(f"source plan lacks partition metadata: {source_dir}")
        shard_id = int(partition.get("shard_id", -1))
        num_shards = int(partition.get("num_shards", -1))
        if shard_id < 0 or num_shards < 1 or shard_id >= num_shards:
            raise ValueError(f"invalid partition metadata: {source_dir}")
        if partition.get("assignment") != "canonical_unique_locator_ordinal_mod":
            raise ValueError(f"unexpected partition assignment: {source_dir}")
        if shard_id in seen_shards:
            raise ValueError(f"duplicate shard id: {shard_id}")
        seen_shards.add(shard_id)
        parent = (
            num_shards,
            int(plan.get("parent_request_count", -1)),
            plan.get("parent_request_sha256"),
            plan.get("parent_index_sha256"),
        )
        if common_parent is None:
            common_parent = parent
        elif common_parent != parent:
            raise ValueError("shard plans do not share one complete parent plan")
        if (
            plan.get("split") != args.split
            or int(plan.get("shot", -1)) != args.shot
            or plan.get("dataset_fingerprint") != dataset_fingerprint
        ):
            raise ValueError(f"source plan protocol mismatch: {source_dir}")

        cache = LLMEvidenceCache(
            str(part_cache),
            max_candidates=int(plan["max_candidates"]),
            expected_shot=args.shot,
            expected_history_protocol=str(plan["history_protocol"]),
            expected_split=args.split,
            expected_dataset_fingerprint=dataset_fingerprint,
            require_generation_metadata=True,
        )
        meta = cache.generation_metadata
        if not isinstance(meta, dict) or bool(meta.get("formal_full_split")):
            raise ValueError(f"part cache must be marked incomplete before merge: {part_cache}")
        provenance = meta.get("provider_provenance", {})
        if (
            provenance.get("source_request_sha256") != plan.get("request_sha256")
            or provenance.get("source_index_sha256") != plan.get("index_sha256")
        ):
            raise ValueError(f"part cache does not match source plan: {part_cache}")

        for index in index_rows:
            custom_id = str(index["custom_id"])
            if custom_id in indexes_by_id:
                raise ValueError(f"duplicate custom id across source shards: {custom_id}")
            indexes_by_id[custom_id] = index
        rows = read_jsonl(part_cache)
        if len(rows) != len(index_rows):
            raise ValueError(f"part cache/index count mismatch: {part_cache}")
        part_rows.extend(rows)
        plans.append(plan)
        part_metas.append(meta)

    assert common_parent is not None
    num_shards, parent_count, parent_request_sha, parent_index_sha = common_parent
    if seen_shards != set(range(int(num_shards))):
        raise ValueError(f"incomplete shard set: {sorted(seen_shards)} of {num_shards}")
    if len(indexes_by_id) != int(parent_count):
        raise ValueError(
            f"source shard union has {len(indexes_by_id)} requests; parent has {parent_count}"
        )

    records_by_id: dict[str, dict[str, Any]] = {}
    locator_set: set[tuple[int, int, int]] = set()
    for record in part_rows:
        query = record.get("query", {})
        custom_id = str(record.get("query_key", ""))
        if custom_id != target_blind_query_key(query) or custom_id not in indexes_by_id:
            raise ValueError(f"invalid or unknown cache query key: {custom_id}")
        if custom_id in records_by_id:
            raise ValueError(f"duplicate cache query key: {custom_id}")
        locator = (
            int(query["known_entity_id"]),
            int(query["oriented_relation_id"]),
            int(query["timestamp"]),
        )
        if locator in locator_set:
            raise ValueError(f"duplicate public locator across shards: {locator}")
        locator_set.add(locator)
        records_by_id[custom_id] = record
    if set(records_by_id) != set(indexes_by_id):
        missing = sorted(set(indexes_by_id).difference(records_by_id))
        raise ValueError(f"merged cache is missing {len(missing)} records: {missing[:3]}")

    kg = load_temporal_kg(str(data_dir))
    rows = {"valid": kg.valid, "test": kg.test}[args.split]
    expected_locators = {(s, r, t) for s, r, _o, t in rows}
    expected_locators.update((o, r + kg.num_relations, t) for s, r, o, t in rows)
    if locator_set != expected_locators:
        raise ValueError(
            f"merged locator set differs from dataset: {len(locator_set)} vs {len(expected_locators)}"
        )

    ordered_ids = sorted(records_by_id, key=lambda key: int(indexes_by_id[key]["ordinal"]))
    cache_bytes = jsonl_bytes([records_by_id[key] for key in ordered_ids])
    first = part_metas[0]
    invariant_keys = (
        "schema_version",
        "purpose",
        "split",
        "shot",
        "seed",
        "history_protocol",
        "provider",
        "model",
        "dataset_fingerprint",
        "query_key_excludes_target",
        "api_called_inside_training_or_evaluation",
        "prompt_ablation",
        "decoding",
    )
    for meta in part_metas[1:]:
        if any(meta.get(key) != first.get(key) for key in invariant_keys):
            raise ValueError("scientific metadata differs across collected cache shards")

    audits = [dict(meta["generation_audit"]) for meta in part_metas]
    usage = _sum_dict(
        [dict(audit.get("token_usage", {})) for audit in audits],
        ("prompt_tokens", "completion_tokens", "total_tokens"),
    )
    successful = sum(int(audit.get("successful_response_count", 0)) for audit in audits)
    weighted_latency = sum(
        float(audit.get("avg_latency_ms", 0.0))
        * int(audit.get("successful_response_count", 0))
        for audit in audits
    ) / max(1, successful)
    resolved_models = sorted(
        {
            str(model)
            for meta in part_metas
            for model in meta.get("provider_provenance", {}).get("resolved_models", [])
        }
    )
    provenance0 = dict(first.get("provider_provenance", {}))
    part_manifest = []
    for source_dir, part_cache, plan, meta in zip(source_dirs, part_caches, plans, part_metas):
        partition = dict(plan["partition"])
        part_manifest.append(
            {
                "shard_id": int(partition["shard_id"]),
                "source_job_dir": str(source_dir),
                "source_plan_sha256": sha256_file(source_dir / batch_cli.PLAN_FILENAME),
                "source_request_sha256": plan["request_sha256"],
                "source_index_sha256": plan["index_sha256"],
                "part_cache": str(part_cache),
                "part_cache_sha256": sha256_file(part_cache),
                "part_metadata_sha256": sha256_file(Path(str(part_cache) + ".meta.json")),
                "records": int(meta["generation_audit"]["request_count"]),
            }
        )
    part_manifest.sort(key=lambda row: int(row["shard_id"]))

    metadata = {key: first.get(key) for key in invariant_keys}
    metadata["formal_full_split"] = True
    metadata["provider_provenance"] = {
        "provider_managed_model": True,
        "exact_weight_revision_available": False,
        "requested_model": first["model"],
        "resolved_models": resolved_models,
        "region": provenance0.get("region"),
        "api_base_url": provenance0.get("api_base_url"),
        "parent_request_count": int(parent_count),
        "parent_request_sha256": parent_request_sha,
        "parent_index_sha256": parent_index_sha,
        "provider_abstention_policy": provenance0.get("provider_abstention_policy"),
        "provenance_file": provenance0.get("provenance_file"),
        "provenance_file_sha256": provenance0.get("provenance_file_sha256"),
        "official_documentation": provenance0.get("official_documentation", {}),
        "sharded_source_plans": part_manifest,
    }
    metadata["generation_audit"] = {
        "finalized_at_utc": dt.datetime.now(UTC).isoformat(),
        "command_argv": [sys.executable, *sys.argv],
        "hostname": platform.node(),
        "mode": "strict_offline_merge_of_realtime_shards",
        "num_shards": int(num_shards),
        "request_count": len(records_by_id),
        "successful_response_count": successful,
        "provider_abstention_count": sum(
            int(audit.get("provider_abstention_count", 0)) for audit in audits
        ),
        "provider_abstention_codes": {
            code: sum(
                int(audit.get("provider_abstention_codes", {}).get(code, 0))
                for audit in audits
            )
            for code in sorted(
                {
                    code
                    for audit in audits
                    for code in audit.get("provider_abstention_codes", {})
                }
            )
        },
        "provider_abstention_policy": provenance0.get("provider_abstention_policy"),
        "token_usage": usage,
        "estimated_list_price_cny": sum(
            float(audit.get("estimated_list_price_cny", 0.0)) for audit in audits
        ),
        "price_verified_at": first["generation_audit"].get("price_verified_at"),
        "attempt_count": sum(int(audit.get("attempt_count", 0)) for audit in audits),
        "retry_count": sum(int(audit.get("retry_count", 0)) for audit in audits),
        "avg_latency_ms": weighted_latency,
        "per_request_latency_available": all(
            bool(audit.get("per_request_latency_available")) for audit in audits
        ),
        "per_successful_response_latency_available": successful > 0,
        "network_called_during_merge": False,
        "cache_sha256": sha256_bytes(cache_bytes),
        "part_manifest": part_manifest,
    }

    write_bytes_atomic(output, cache_bytes)
    write_json_atomic(meta_output, metadata)
    merged = LLMEvidenceCache(
        str(output),
        max_candidates=int(plans[0]["max_candidates"]),
        expected_shot=args.shot,
        expected_history_protocol=str(plans[0]["history_protocol"]),
        expected_split=args.split,
        expected_dataset_fingerprint=dataset_fingerprint,
        require_generation_metadata=True,
    )
    if len(merged.records) != len(expected_locators):
        raise RuntimeError("post-merge formal cache count mismatch")
    result = {
        "status": "realtime_shards_merged_offline",
        "network_called": False,
        "output": str(output),
        "records": len(merged.records),
        "sha256": merged.sha256,
        "metadata_sha256": merged.generation_metadata_sha256,
    }
    print(canonical_json(result))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict offline merge of Qwen realtime cache shards"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", choices=["valid", "test"], required=True)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--source-job-dir", action="append", required=True)
    parser.add_argument("--part-cache", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser


if __name__ == "__main__":
    merge(build_parser().parse_args())
