#!/usr/bin/env python3
"""Prepare, submit, collect, and audit target-blind Qwen Batch jobs.

``prepare``, ``estimate``, ``collect``, and ``prepare-retry`` are strictly
offline.  Network actions additionally require ``--execute-api`` and local
environment confirmation gates.  The resulting cache implements the same
provider-independent contract consumed by v1.7.0alterego_v5_llm without
changing the parent model or the running local-Qwen pipeline.
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import os
import platform
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stlp_generate_candidates import initial_facts, split_rows
from src.aliyun_qwen_batch import (
    DEFAULT_BASE_URL,
    DEFAULT_BATCH_MODEL,
    PROVIDER_NAME,
    ROLLING_BATCH_MODEL,
    AliyunQwenBatchClient,
    canonical_json,
    jsonl_bytes,
    parse_candidate_content,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    validate_batch_requests,
    write_bytes_atomic,
    write_json_atomic,
)
from src.data import HistoryIndex, Quad, load_temporal_kg
from src.llm_cache import (
    LLMEvidenceCache,
    dataset_files_fingerprint,
    query_locator,
    target_blind_query_key,
)
from src.stlp import (
    EntityMapper,
    TargetBlindQuery,
    build_query_metadata,
    build_stlp_prompt,
    load_id_map,
    parse_and_map_response,
    sha256_text,
)
from src.train import choose_causal_support, group_by_relation


UTC = dt.timezone.utc
PROMPT_TEMPLATE_VERSION = "stlp-aliyun-qwen-batch-v1"
REQUEST_FILENAME = "batch_requests.jsonl"
INDEX_FILENAME = "batch_index.jsonl"
PLAN_FILENAME = "batch_plan.json"
STATE_FILENAME = "batch_state.json"
OUTPUT_FILENAME = "batch_output.jsonl"
ERROR_FILENAME = "batch_errors.jsonl"
SUPPORTED_MODELS = {DEFAULT_BATCH_MODEL, ROLLING_BATCH_MODEL}
SUPPORTED_SHOTS = {1, 3, 5, 10}
# CNY per million tokens after the documented 50% Batch discount.  These are
# an estimate only and are stamped with the verification date in provenance.
CURRENT_BATCH_RATES = {
    DEFAULT_BATCH_MODEL: {"input": 0.10, "output": 0.40},
    ROLLING_BATCH_MODEL: {"input": 0.075, "output": 0.75},
}


def utc_now() -> str:
    return dt.datetime.now(UTC).isoformat()


def recent_public_history(
    index: HistoryIndex, known_entity_id: int, timestamp: int, limit: int
) -> List[Quad]:
    rows = index.events_by_subject[known_entity_id]
    position = bisect.bisect_left(rows, (timestamp, -1, -1))
    selected = rows[max(0, position - limit) : position]
    return [
        (known_entity_id, relation, candidate, event_time)
        for event_time, relation, candidate in selected
    ]


def deterministic_support_permutation(
    support: Sequence[Quad], *, known_entity_id: int, oriented_relation_id: int,
    timestamp: int, seed: int
) -> List[Quad]:
    """Apply a deterministic non-identity cyclic order control.

    A non-zero rotation is used instead of an unconstrained shuffle so every
    query with at least two support facts is guaranteed to change order while
    preserving the exact multiset of public facts.
    """

    rows = list(support)
    if len(rows) < 2:
        return rows
    material = f"{known_entity_id}:{oriented_relation_id}:{timestamp}:{seed}:support-order"
    shift = 1 + int(sha256_text(material)[:16], 16) % (len(rows) - 1)
    return rows[shift:] + rows[:shift]


def deterministic_entity_placeholders(
    entity_names: Sequence[str], *, seed: int
) -> List[str]:
    """Return stable opaque placeholders for the complete entity vocabulary."""

    names = list(entity_names)
    aliases = [
        "node_" + sha256_text(f"{int(seed)}:{index}:entity-placeholder")[:16]
        for index in range(len(names))
    ]
    if len(set(aliases)) != len(aliases):
        raise RuntimeError("entity placeholder collision")
    if set(names).intersection(aliases):
        raise RuntimeError("entity placeholder collides with an original entity name")
    return aliases


def response_name_map_for_plan(
    original_name_to_id: Mapping[str, int],
    original_entity_names: Sequence[str],
    plan: Mapping[str, object],
) -> Dict[str, int]:
    ablation = plan.get("prompt_ablation", {})
    if not isinstance(ablation, Mapping) or not bool(ablation.get("replace_entity_names")):
        return dict(original_name_to_id)
    placeholders = deterministic_entity_placeholders(
        original_entity_names, seed=int(plan["seed"])
    )
    expected = plan.get("entity_name_replacement", {})
    digest = sha256_text(json.dumps(placeholders, ensure_ascii=False, separators=(",", ":")))
    if not isinstance(expected, Mapping) or expected.get("placeholder_sha256") != digest:
        raise ValueError("entity-name replacement hash mismatch")
    return {name: entity_id for entity_id, name in enumerate(placeholders)}


def _assert_prepare_protocol(args: argparse.Namespace) -> None:
    if args.shot not in SUPPORTED_SHOTS:
        raise ValueError(
            f"the extended task protocol permits only shots {sorted(SUPPORTED_SHOTS)}"
        )
    if args.history_protocol not in {"standard_rolling_history", "strict_static_history"}:
        raise ValueError("unknown history protocol")
    if args.model not in SUPPORTED_MODELS:
        raise ValueError(
            f"unsupported Batch model {args.model!r}; choose one of {sorted(SUPPORTED_MODELS)}"
        )
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.max_candidates < 1 or args.max_candidates > 10:
        raise ValueError("--max-candidates must be between 1 and 10")


def _job_paths(job_dir: Path) -> Dict[str, Path]:
    return {
        "request": job_dir / REQUEST_FILENAME,
        "index": job_dir / INDEX_FILENAME,
        "plan": job_dir / PLAN_FILENAME,
        "state": job_dir / STATE_FILENAME,
        "output": job_dir / OUTPUT_FILENAME,
        "error": job_dir / ERROR_FILENAME,
    }


def _prepare_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    _assert_prepare_protocol(args)
    kg = load_temporal_kg(str(args.data_dir))
    _, entity_names = load_id_map(
        str(args.data_dir / "entity2id.txt"), expected_size=kg.num_entities
    )
    _, relation_names = load_id_map(
        str(args.data_dir / "relation2id.txt"), expected_size=kg.num_relations
    )
    fingerprint = dataset_files_fingerprint(str(args.data_dir))
    base_facts = initial_facts(kg, args.split)
    history = HistoryIndex(
        base_facts,
        kg.num_entities,
        kg.num_relations * 2,
        history_len=max(args.history_len, 16),
    )
    support_by_relation = group_by_relation(base_facts)
    snapshots: Dict[int, List[Quad]] = defaultdict(list)
    for row in split_rows(kg, args.split):
        snapshots[row[3]].append(row)

    request_rows: List[Dict[str, object]] = []
    index_rows: List[Dict[str, object]] = []
    seen_locators: set[Tuple[int, int, int]] = set()
    seen_ids: set[str] = set()
    stop = False
    permute_support_order = bool(getattr(args, "permute_support_order", False))
    replace_entity_names = bool(getattr(args, "replace_entity_names", False))
    prompt_entity_names = entity_names
    entity_name_replacement: Dict[str, object] | None = None
    if replace_entity_names:
        prompt_entity_names = deterministic_entity_placeholders(
            entity_names, seed=int(args.seed)
        )
        entity_name_replacement = {
            "method": "full-vocabulary deterministic opaque placeholders",
            "entity_count": len(entity_names),
            "placeholder_sha256": sha256_text(
                json.dumps(prompt_entity_names, ensure_ascii=False, separators=(",", ":"))
            ),
            "inverse_mapping_applied_during_collection": True,
            "original_names_sent_to_provider": False,
        }
    prompt_variant = PROMPT_TEMPLATE_VERSION
    if args.omit_support:
        prompt_variant += "-no-support"
    if args.omit_history:
        prompt_variant += "-no-history"
    if permute_support_order:
        prompt_variant += "-support-order-permuted"
    if replace_entity_names:
        prompt_variant += "-entity-names-replaced"
    support_order_eligible = 0
    support_order_changed = 0
    for timestamp in sorted(snapshots):
        snapshot = snapshots[timestamp]
        oriented_rows = list(snapshot) + [
            (o, relation + kg.num_relations, s, event_time)
            for s, relation, o, event_time in snapshot
        ]
        for known_entity_id, oriented_relation_id, _hidden_target, event_time in oriented_rows:
            locator = query_locator(known_entity_id, oriented_relation_id, event_time)
            if locator in seen_locators:
                continue
            seen_locators.add(locator)
            if args.limit and len(request_rows) >= args.limit:
                stop = True
                break
            direction = "tail" if oriented_relation_id < kg.num_relations else "head"
            public_query = TargetBlindQuery(
                split=args.split,
                direction=direction,
                known_entity_id=known_entity_id,
                oriented_relation_id=oriented_relation_id,
                timestamp=event_time,
            )
            relation_rows = support_by_relation.get(oriented_relation_id, [])
            if relation_rows and relation_rows[0][3] < event_time:
                support = choose_causal_support(
                    support_by_relation,
                    (known_entity_id, oriented_relation_id, 0, event_time),
                    args.shot,
                )
            else:
                support = []
            recent = recent_public_history(history, known_entity_id, event_time, args.history_len)
            prompt_support = [] if args.omit_support else support
            if permute_support_order and prompt_support:
                if len(prompt_support) >= 2:
                    support_order_eligible += 1
                permuted_support = deterministic_support_permutation(
                    prompt_support,
                    known_entity_id=known_entity_id,
                    oriented_relation_id=oriented_relation_id,
                    timestamp=event_time,
                    seed=int(args.seed),
                )
                if permuted_support != list(prompt_support):
                    support_order_changed += 1
                prompt_support = permuted_support
            prompt_history = [] if args.omit_history else recent
            prompt = build_stlp_prompt(
                public_query,
                prompt_support,
                prompt_history,
                prompt_entity_names,
                relation_names,
                kg.num_relations,
                max_candidates=args.max_candidates,
                compact_response_keys=False,
            )
            query = build_query_metadata(
                public_query,
                shot=args.shot,
                seed=args.seed,
                history_protocol=args.history_protocol,
                support=prompt_support,
                history=prompt_history,
                dataset_fingerprint=fingerprint,
                prompt_template_version=prompt_variant,
            )
            custom_id = target_blind_query_key(query)
            if custom_id in seen_ids:
                raise ValueError(f"duplicate deterministic custom_id: {custom_id}")
            seen_ids.add(custom_id)
            request_rows.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model,
                        "enable_thinking": False,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "Return valid JSON only. Use only target-blind causal "
                                    "context supplied by the user."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.0,
                    },
                }
            )
            index_rows.append(
                {
                    "schema_version": 1,
                    "ordinal": len(index_rows),
                    "custom_id": custom_id,
                    "query_key": custom_id,
                    "query": query,
                    "prompt_hash": sha256_text(prompt),
                    # A strict prompt-only no-support ablation removes examples
                    # from the prompt, but must preserve the parent model's
                    # downstream evidence features.  Keep the causal support
                    # names used by template_agreement separate from
                    # prompt_support.
                    "support_candidate_names": [prompt_entity_names[row[2]] for row in support],
                }
            )
        if stop:
            break
        if args.history_protocol == "standard_rolling_history" or args.split == "train":
            history.add_facts(oriented_rows)
            for fact in oriented_rows:
                support_by_relation.setdefault(fact[1], []).append(fact)

    if not request_rows:
        raise ValueError("prepare produced no target-blind queries")
    request_bytes = jsonl_bytes(request_rows)
    index_bytes = jsonl_bytes(index_rows)
    plan: Dict[str, object] = {
        "schema_version": 1,
        "purpose": "target-blind STLP Alibaba Qwen Batch staging",
        "provider": PROVIDER_NAME,
        "region": "cn-beijing",
        "api_base_url": DEFAULT_BASE_URL,
        "model": args.model,
        "split": args.split,
        "shot": args.shot,
        "seed": args.seed,
        "history_protocol": args.history_protocol,
        "history_len": args.history_len,
        "max_candidates": args.max_candidates,
        "dataset_fingerprint": fingerprint,
        "prompt_template_version": prompt_variant,
        "prompt_ablation": {
            "omit_support": bool(args.omit_support),
            "omit_history": bool(args.omit_history),
            "permute_support_order": permute_support_order,
            "replace_entity_names": replace_entity_names,
        },
        "support_order_permutation": {
            "method": "deterministic non-zero cyclic rotation",
            "eligible_queries": support_order_eligible,
            "changed_queries": support_order_changed,
            "fact_multiset_preserved": True,
        }
        if permute_support_order
        else None,
        "entity_name_replacement": entity_name_replacement,
        "query_key_excludes_target": True,
        "complete_split": args.limit == 0,
        "request_count": len(request_rows),
        "request_file": REQUEST_FILENAME,
        "request_sha256": sha256_bytes(request_bytes),
        "request_bytes": len(request_bytes),
        "index_file": INDEX_FILENAME,
        "index_sha256": sha256_bytes(index_bytes),
        "decoding": {
            "enable_thinking": False,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "max_tokens_omitted": True,
        },
        "staging_contains_public_prompts": True,
        "staging_contains_hidden_targets": False,
    }
    return request_rows, index_rows, plan


def prepare_job(args: argparse.Namespace) -> Dict[str, object]:
    args.data_dir = Path(args.data_dir).resolve()
    args.job_dir = Path(args.job_dir).resolve()
    request_rows, index_rows, plan = _prepare_rows(args)
    paths = _job_paths(args.job_dir)
    for key in ("request", "index", "plan", "state", "output", "error"):
        if paths[key].exists():
            raise FileExistsError(f"refusing to reuse a non-empty Batch job: {paths[key]}")
    args.job_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.job_dir, 0o700)
    write_bytes_atomic(paths["request"], jsonl_bytes(request_rows))
    write_bytes_atomic(paths["index"], jsonl_bytes(index_rows))
    write_json_atomic(paths["plan"], plan)
    validation = validate_batch_requests(paths["request"])
    if validation["sha256"] != plan["request_sha256"]:
        raise RuntimeError("post-write Batch request hash mismatch")
    result = {
        "status": "prepared_offline",
        "job_dir": str(args.job_dir),
        "request_count": plan["request_count"],
        "model": plan["model"],
        "complete_split": plan["complete_split"],
        "network_called": False,
    }
    print(canonical_json(result))
    return result


def _verify_job(job_dir: Path) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    paths = _job_paths(job_dir)
    plan = read_json(paths["plan"])
    if plan.get("provider") != PROVIDER_NAME or plan.get("api_base_url") != DEFAULT_BASE_URL:
        raise ValueError("Batch plan provider or official endpoint mismatch")
    request_validation = validate_batch_requests(paths["request"])
    if request_validation["sha256"] != plan.get("request_sha256"):
        raise ValueError("Batch request file hash differs from the immutable plan")
    request_rows = read_jsonl(paths["request"])
    index_rows = read_jsonl(paths["index"])
    if sha256_file(paths["index"]) != plan.get("index_sha256"):
        raise ValueError("Batch index file hash differs from the immutable plan")
    if len(request_rows) != len(index_rows) or len(request_rows) != int(plan.get("request_count", -1)):
        raise ValueError("Batch request/index count mismatch")
    request_by_id: Dict[str, Dict[str, object]] = {}
    index_by_id: Dict[str, Dict[str, object]] = {}
    for request, index in zip(request_rows, index_rows):
        custom_id = str(request.get("custom_id", ""))
        if custom_id != index.get("custom_id") or index.get("query_key") != custom_id:
            raise ValueError("Batch request/index custom_id mismatch")
        query = index.get("query")
        if not isinstance(query, dict) or target_blind_query_key(query) != custom_id:
            raise ValueError(f"invalid target-blind query key in Batch index: {custom_id}")
        body = request.get("body")
        messages = body.get("messages") if isinstance(body, dict) else None
        if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
            raise ValueError(f"invalid messages for Batch request {custom_id}")
        prompt = messages[-1].get("content")
        if not isinstance(prompt, str) or sha256_text(prompt) != index.get("prompt_hash"):
            raise ValueError(f"prompt hash mismatch for Batch request {custom_id}")
        if custom_id in request_by_id or custom_id in index_by_id:
            raise ValueError(f"duplicate Batch custom_id: {custom_id}")
        request_by_id[custom_id] = request
        index_by_id[custom_id] = index
    return plan, request_rows, index_rows


def estimate_job(args: argparse.Namespace) -> Dict[str, object]:
    job_dir = Path(args.job_dir).resolve()
    plan, _, _ = _verify_job(job_dir)
    rates = CURRENT_BATCH_RATES.get(str(plan["model"]))
    input_rate = args.input_rate if args.input_rate is not None else (rates or {}).get("input")
    output_rate = args.output_rate if args.output_rate is not None else (rates or {}).get("output")
    if input_rate is None or output_rate is None:
        raise ValueError("unknown model pricing; provide --input-rate and --output-rate")
    requests = int(plan["request_count"])
    input_tokens = requests * float(args.avg_prompt_tokens)
    output_tokens = requests * float(args.avg_output_tokens)
    cost = (input_tokens * float(input_rate) + output_tokens * float(output_rate)) / 1_000_000
    result = {
        "status": "estimate_only",
        "model": plan["model"],
        "requests": requests,
        "assumed_average_prompt_tokens": float(args.avg_prompt_tokens),
        "assumed_average_output_tokens": float(args.avg_output_tokens),
        "batch_input_cny_per_million": float(input_rate),
        "batch_output_cny_per_million": float(output_rate),
        "estimated_cost_cny": round(cost, 4),
        "pricing_must_be_rechecked_before_submit": True,
        "network_called": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def _require_network_authority(args: argparse.Namespace, *, paid_upload: bool) -> None:
    if not args.execute_api:
        raise ValueError("network action requires the explicit --execute-api flag")
    upload_confirmation = os.environ.get(
        "CONFIRM_ALIYUN_QWEN_DATA_UPLOAD",
        os.environ.get("CONFIRM_ALIYUN_QWEN_UPLOAD", "NO"),
    )
    if upload_confirmation != "YES":
        raise ValueError("set CONFIRM_ALIYUN_QWEN_DATA_UPLOAD=YES after reviewing the data policy")
    if paid_upload and os.environ.get("CONFIRM_ALIYUN_QWEN_PAID_BATCH", "NO") != "YES":
        raise ValueError("set CONFIRM_ALIYUN_QWEN_PAID_BATCH=YES after reviewing current pricing")


def _client(args: argparse.Namespace) -> AliyunQwenBatchClient:
    return AliyunQwenBatchClient.from_environment(timeout_seconds=args.timeout)


def _safe_state_view(value: Mapping[str, object]) -> Dict[str, object]:
    allowed = {
        "schema_version",
        "provider",
        "model",
        "input_file_id",
        "batch_id",
        "status",
        "output_file_id",
        "error_file_id",
        "submitted_at_utc",
        "last_checked_at_utc",
        "downloaded_at_utc",
        "cancel_requested_at_utc",
        "request_sha256",
        "output_sha256",
        "error_sha256",
    }
    return {key: value[key] for key in sorted(allowed) if key in value}


def submit_job(args: argparse.Namespace) -> Dict[str, object]:
    _require_network_authority(args, paid_upload=True)
    job_dir = Path(args.job_dir).resolve()
    plan, _, _ = _verify_job(job_dir)
    paths = _job_paths(job_dir)
    if paths["state"].exists():
        raise FileExistsError("Batch state already exists; refusing a duplicate paid submission")
    match = re.fullmatch(r"(\d+)h", args.completion_window)
    if not match or not 24 <= int(match.group(1)) <= 336:
        raise ValueError("--completion-window must be an integer from 24h through 336h")
    client = _client(args)
    uploaded = client.upload_batch_file(paths["request"])
    state: Dict[str, object] = {
        "schema_version": 1,
        "provider": PROVIDER_NAME,
        "model": plan["model"],
        "request_sha256": plan["request_sha256"],
        "input_file_id": uploaded["id"],
        "status": f"file_{uploaded.get('status', 'uploaded')}",
        "submitted_at_utc": utc_now(),
    }
    write_json_atomic(paths["state"], state)
    processed = client.wait_until_file_processed(
        str(uploaded["id"]),
        timeout_seconds=args.file_process_timeout,
        poll_seconds=args.poll_seconds,
    )
    state["status"] = f"file_{processed.get('status', 'processed')}"
    write_json_atomic(paths["state"], state, replace=True)
    batch = client.create_batch(
        str(uploaded["id"]),
        completion_window=args.completion_window,
        metadata={
            "project": "NineFuseTKG-v1.7.0alterego-v5-llm",
            "split": str(plan["split"]),
            "shot": str(plan["shot"]),
            "request_sha256": str(plan["request_sha256"]),
        },
    )
    state.update(
        {
            "batch_id": batch["id"],
            "status": batch.get("status", "submitted"),
            "last_checked_at_utc": utc_now(),
            "output_file_id": batch.get("output_file_id"),
            "error_file_id": batch.get("error_file_id"),
        }
    )
    write_json_atomic(paths["state"], state, replace=True)
    result = _safe_state_view(state)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def status_job(args: argparse.Namespace) -> Dict[str, object]:
    _require_network_authority(args, paid_upload=False)
    job_dir = Path(args.job_dir).resolve()
    _verify_job(job_dir)
    paths = _job_paths(job_dir)
    state = read_json(paths["state"])
    if not state.get("batch_id"):
        raise ValueError("Batch state has no batch_id")
    batch = _client(args).get_batch(str(state["batch_id"]))
    state.update(
        {
            "status": batch.get("status", "unknown"),
            "last_checked_at_utc": utc_now(),
            "output_file_id": batch.get("output_file_id"),
            "error_file_id": batch.get("error_file_id"),
            "request_counts": batch.get("request_counts"),
            "batch_created_at": batch.get("created_at"),
            "batch_completed_at": batch.get("completed_at"),
        }
    )
    write_json_atomic(paths["state"], state, replace=True)
    result = _safe_state_view(state)
    if "request_counts" in state:
        result["request_counts"] = state["request_counts"]
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def cancel_job(args: argparse.Namespace) -> Dict[str, object]:
    _require_network_authority(args, paid_upload=False)
    job_dir = Path(args.job_dir).resolve()
    _verify_job(job_dir)
    paths = _job_paths(job_dir)
    state = read_json(paths["state"])
    batch_id = state.get("batch_id")
    if not batch_id:
        raise ValueError("Batch state has no batch_id")
    current = _client(args).get_batch(str(batch_id))
    status = str(current.get("status", "unknown"))
    if status in {"completed", "failed", "expired", "cancelled"}:
        batch = current
    else:
        batch = _client(args).cancel_batch(str(batch_id))
    state.update(
        {
            "status": batch.get("status", status),
            "last_checked_at_utc": utc_now(),
            "cancel_requested_at_utc": utc_now(),
            "output_file_id": batch.get("output_file_id"),
            "error_file_id": batch.get("error_file_id"),
            "request_counts": batch.get("request_counts"),
            "batch_cancelled_at": batch.get("cancelled_at"),
        }
    )
    write_json_atomic(paths["state"], state, replace=True)
    result = _safe_state_view(state)
    if "request_counts" in state:
        result["request_counts"] = state["request_counts"]
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def _write_download(path: Path, data: bytes) -> None:
    if path.exists():
        if sha256_file(path) != sha256_bytes(data):
            raise FileExistsError(f"download target exists with different bytes: {path}")
        return
    write_bytes_atomic(path, data)


def download_job(args: argparse.Namespace) -> Dict[str, object]:
    _require_network_authority(args, paid_upload=False)
    job_dir = Path(args.job_dir).resolve()
    _verify_job(job_dir)
    paths = _job_paths(job_dir)
    state = read_json(paths["state"])
    batch_id = state.get("batch_id")
    if not batch_id:
        raise ValueError("Batch state has no batch_id")
    client = _client(args)
    batch = client.get_batch(str(batch_id))
    status = str(batch.get("status", ""))
    if status not in {"completed", "failed", "expired", "cancelled"}:
        raise RuntimeError(f"Batch is not in a downloadable terminal state: {status}")
    output_file_id = batch.get("output_file_id")
    error_file_id = batch.get("error_file_id")
    if output_file_id:
        _write_download(paths["output"], client.download_file(str(output_file_id)))
    if error_file_id:
        _write_download(paths["error"], client.download_file(str(error_file_id)))
    state.update(
        {
            "status": status,
            "last_checked_at_utc": utc_now(),
            "downloaded_at_utc": utc_now(),
            "output_file_id": output_file_id,
            "error_file_id": error_file_id,
        }
    )
    if paths["output"].exists():
        state["output_sha256"] = sha256_file(paths["output"])
    if paths["error"].exists():
        state["error_sha256"] = sha256_file(paths["error"])
    write_json_atomic(paths["state"], state, replace=True)
    result = _safe_state_view(state)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def _result_paths(args: argparse.Namespace, job_dir: Path) -> Tuple[List[Path], List[Path]]:
    result_paths = [Path(value).resolve() for value in (args.result_file or [])]
    error_paths = [Path(value).resolve() for value in (args.error_file or [])]
    paths = _job_paths(job_dir)
    if not result_paths and paths["output"].exists():
        result_paths = [paths["output"]]
    if not error_paths and paths["error"].exists():
        error_paths = [paths["error"]]
    if not result_paths:
        raise FileNotFoundError("no Batch result file supplied or downloaded")
    return result_paths, error_paths


def _classify_results(
    result_paths: Sequence[Path],
    error_paths: Sequence[Path],
    expected_ids: set[str],
    *,
    max_candidates: int,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, str]]:
    successes: Dict[str, Dict[str, object]] = {}
    failures: Dict[str, str] = {}
    seen_success_rows: set[str] = set()
    for path in result_paths:
        seen_in_file: set[str] = set()
        for line_number, row in enumerate(read_jsonl(path), start=1):
            custom_id = row.get("custom_id")
            if not isinstance(custom_id, str) or custom_id not in expected_ids:
                raise ValueError(f"unknown or invalid custom_id at {path}:{line_number}")
            if custom_id in seen_in_file:
                raise ValueError(f"duplicate custom_id at {path}:{line_number}: {custom_id}")
            seen_in_file.add(custom_id)
            error = row.get("error")
            response = row.get("response")
            if error not in (None, {}):
                failures[custom_id] = f"provider error: {error}"
                continue
            try:
                if not isinstance(response, dict) or int(response.get("status_code", 0)) != 200:
                    raise ValueError(f"non-200 Batch response: {response}")
                body = response.get("body")
                if not isinstance(body, dict):
                    raise ValueError("Batch response body is missing")
                choices = body.get("choices")
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    raise ValueError("Batch response has no first choice")
                choice = choices[0]
                finish_reason = choice.get("finish_reason")
                if finish_reason == "length":
                    raise ValueError("Batch response was truncated at the output limit")
                if finish_reason not in (None, "stop"):
                    raise ValueError(f"unsupported finish_reason: {finish_reason!r}")
                message = choice.get("message")
                if not isinstance(message, dict):
                    raise ValueError("Batch response choice has no message")
                content = message.get("content")
                parse_candidate_content(content, max_candidates=max_candidates)
            except (TypeError, ValueError) as exc:
                failures[custom_id] = str(exc)
                continue
            if custom_id in seen_success_rows:
                raise ValueError(f"duplicate successful result across files: {custom_id}")
            seen_success_rows.add(custom_id)
            successes[custom_id] = {
                "body": body,
                "content": content,
                "request_id": response.get("request_id") or row.get("id"),
                "source_path": str(path),
            }
            failures.pop(custom_id, None)
    for path in error_paths:
        seen_in_file: set[str] = set()
        for line_number, row in enumerate(read_jsonl(path), start=1):
            custom_id = row.get("custom_id")
            if not isinstance(custom_id, str) or custom_id not in expected_ids:
                raise ValueError(f"unknown custom_id in Batch error file at {path}:{line_number}")
            if custom_id in seen_in_file:
                raise ValueError(f"duplicate custom_id in Batch error file: {custom_id}")
            seen_in_file.add(custom_id)
            if custom_id not in successes:
                failures[custom_id] = f"Batch error file entry: {row.get('error')}"
    return successes, failures


def _state_if_present(job_dir: Path) -> Dict[str, object]:
    path = _job_paths(job_dir)["state"]
    return read_json(path) if path.exists() else {}


def collect_job(args: argparse.Namespace) -> Dict[str, object]:
    job_dir = Path(args.job_dir).resolve()
    output = Path(args.output).resolve()
    meta_output = Path(str(output) + ".meta.json")
    if output.exists() or meta_output.exists():
        raise FileExistsError(f"refusing to overwrite formal cache artifacts: {output}")
    plan, request_rows, index_rows = _verify_job(job_dir)
    if not bool(plan.get("complete_split")) and not args.allow_incomplete_cache:
        raise ValueError("pilot/incomplete plan requires --allow-incomplete-cache and must not be reported as formal")
    state = _state_if_present(job_dir)
    if bool(plan.get("complete_split")) and not state.get("batch_id"):
        raise ValueError("formal full-split collection requires the submitted Batch state and batch_id")
    result_paths, error_paths = _result_paths(args, job_dir)
    expected_ids = {str(row["custom_id"]) for row in index_rows}
    successes, failures = _classify_results(
        result_paths,
        error_paths,
        expected_ids,
        max_candidates=int(plan["max_candidates"]),
    )
    unresolved = sorted(expected_ids.difference(successes))
    if unresolved:
        examples = [f"{custom_id}:{failures.get(custom_id, 'missing')}" for custom_id in unresolved[:5]]
        raise ValueError(
            f"Batch collection is incomplete: {len(unresolved)} of {len(expected_ids)} unresolved; "
            f"examples={examples}. Run prepare-retry; no cache was written."
        )

    kg = load_temporal_kg(str(args.data_dir))
    name_to_id, entity_names = load_id_map(
        str(Path(args.data_dir) / "entity2id.txt"), expected_size=kg.num_entities
    )
    response_name_to_id = response_name_map_for_plan(name_to_id, entity_names, plan)
    mapper = EntityMapper(
        response_name_to_id,
        fuzzy_threshold=args.fuzzy_threshold,
        fuzzy_margin=args.fuzzy_margin,
    )
    if dataset_files_fingerprint(str(args.data_dir)) != plan.get("dataset_fingerprint"):
        raise ValueError("current dataset files differ from the prepared Batch plan")
    records: List[Dict[str, object]] = []
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    resolved_models: set[str] = set()
    for index in sorted(index_rows, key=lambda row: int(row["ordinal"])):
        custom_id = str(index["custom_id"])
        result = successes[custom_id]
        candidates, diagnostics = parse_and_map_response(
            str(result["content"]),
            mapper,
            list(index.get("support_candidate_names", [])),
            max_candidates=int(plan["max_candidates"]),
        )
        if bool(plan.get("prompt_ablation", {}).get("replace_entity_names")):
            for candidate in candidates:
                mapped_id = candidate.get("mapped_entity_id")
                if mapped_id is not None:
                    candidate["mapped_entity_name"] = entity_names[int(mapped_id)]
                    candidate["mapping_method"] = "placeholder_" + str(
                        candidate.get("mapping_method", "unknown")
                    )
        body = result["body"]
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        token_usage = {
            key: int(usage.get(key, 0) or 0) if isinstance(usage, dict) else 0
            for key in usage_totals
        }
        for key, value in token_usage.items():
            usage_totals[key] += value
        resolved_model = str(body.get("model") or plan["model"])
        resolved_models.add(resolved_model)
        records.append(
            {
                "schema_version": 1,
                "query_key": custom_id,
                "query": index["query"],
                "prompt_hash": index["prompt_hash"],
                "provider": PROVIDER_NAME,
                "model": resolved_model,
                "response_id": body.get("id") or result.get("request_id"),
                "candidates": candidates,
                "diagnostics": diagnostics,
                # Batch does not expose comparable per-request latency.  Keep
                # the legacy numeric field for cache compatibility and mark it
                # unavailable in the authoritative sidecar below.
                "latency_ms": 0.0,
                "token_usage": token_usage,
            }
        )
    cache_bytes = jsonl_bytes(records)
    provenance_path = PROJECT_ROOT / "ALIYUN_QWEN_BATCH_PROVENANCE.json"
    provenance = read_json(provenance_path)
    result_hashes = [sha256_file(path) for path in result_paths]
    error_hashes = [sha256_file(path) for path in error_paths]
    metadata: Dict[str, object] = {
        "schema_version": 2,
        "purpose": "target-blind STLP candidate cache",
        "split": plan["split"],
        "shot": plan["shot"],
        "seed": plan["seed"],
        "history_protocol": plan["history_protocol"],
        "provider": PROVIDER_NAME,
        "model": plan["model"],
        "provider_provenance": {
            "provider_managed_model": True,
            "exact_weight_revision_available": False,
            "requested_model": plan["model"],
            "resolved_models": sorted(resolved_models),
            "region": plan["region"],
            "api_base_url": plan["api_base_url"],
            "batch_id": state.get("batch_id"),
            "input_file_id": state.get("input_file_id"),
            "output_file_id": state.get("output_file_id"),
            "error_file_id": state.get("error_file_id"),
            "request_sha256": plan["request_sha256"],
            "index_sha256": plan["index_sha256"],
            "result_sha256": result_hashes,
            "error_sha256": error_hashes,
            "provenance_file": provenance_path.name,
            "provenance_file_sha256": sha256_file(provenance_path),
            "official_documentation": provenance.get("official_documentation", {}),
        },
        "dataset_fingerprint": plan["dataset_fingerprint"],
        "query_key_excludes_target": True,
        "api_called_inside_training_or_evaluation": False,
        "formal_full_split": bool(plan["complete_split"]),
        "prompt_ablation": plan["prompt_ablation"],
        "decoding": plan["decoding"],
        "generation_audit": {
            "finalized_at_utc": utc_now(),
            "command_argv": [sys.executable, *sys.argv],
            "hostname": platform.node(),
            "batch_submitted_at_utc": state.get("submitted_at_utc"),
            "batch_downloaded_at_utc": state.get("downloaded_at_utc"),
            "batch_status": state.get("status"),
            "request_count": len(records),
            "token_usage": usage_totals,
            "per_request_latency_available": False,
            "latency_reporting_contract": "report Batch job wall time/throughput, not record latency_ms",
            "network_called_during_collection": False,
            "cache_sha256": sha256_bytes(cache_bytes),
        },
    }
    # Every possible parse, completeness, schema, and dataset failure has
    # already happened.  Only now are formal artifacts committed.
    write_bytes_atomic(output, cache_bytes)
    write_json_atomic(meta_output, metadata)
    cache = LLMEvidenceCache(
        str(output),
        max_candidates=int(plan["max_candidates"]),
        expected_shot=int(plan["shot"]),
        expected_history_protocol=str(plan["history_protocol"]),
        expected_split=str(plan["split"]),
        expected_dataset_fingerprint=str(plan["dataset_fingerprint"]),
        require_generation_metadata=True,
    )
    result = {
        "status": "cache_collected_offline",
        "output": str(output),
        "records": len(cache.records),
        "sha256": cache.sha256,
        "formal_full_split": bool(plan["complete_split"]),
        "network_called": False,
    }
    print(canonical_json(result))
    return result


def prepare_retry_job(args: argparse.Namespace) -> Dict[str, object]:
    job_dir = Path(args.job_dir).resolve()
    retry_dir = Path(args.retry_dir).resolve()
    plan, request_rows, index_rows = _verify_job(job_dir)
    result_paths, error_paths = _result_paths(args, job_dir)
    expected_ids = {str(row["custom_id"]) for row in index_rows}
    successes, failures = _classify_results(
        result_paths,
        error_paths,
        expected_ids,
        max_candidates=int(plan["max_candidates"]),
    )
    retry_ids = expected_ids.difference(successes)
    if not retry_ids:
        raise ValueError("all Batch requests already have valid successful results; no retry is needed")
    request_subset = [row for row in request_rows if str(row["custom_id"]) in retry_ids]
    index_subset = [row for row in index_rows if str(row["custom_id"]) in retry_ids]
    request_bytes = jsonl_bytes(request_subset)
    index_bytes = jsonl_bytes(index_subset)
    retry_plan = dict(plan)
    retry_plan.update(
        {
            "request_count": len(request_subset),
            "request_sha256": sha256_bytes(request_bytes),
            "request_bytes": len(request_bytes),
            "index_sha256": sha256_bytes(index_bytes),
            "complete_split": False,
            "retry_of_request_sha256": plan["request_sha256"],
            "retry_reason_counts": {
                "missing_or_invalid": len(retry_ids),
                "provider_failures_observed": len(failures),
            },
        }
    )
    paths = _job_paths(retry_dir)
    if retry_dir.exists() and any(retry_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty retry directory: {retry_dir}")
    retry_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(retry_dir, 0o700)
    write_bytes_atomic(paths["request"], request_bytes)
    write_bytes_atomic(paths["index"], index_bytes)
    write_json_atomic(paths["plan"], retry_plan)
    validate_batch_requests(paths["request"])
    result = {
        "status": "retry_prepared_offline",
        "retry_dir": str(retry_dir),
        "retry_requests": len(retry_ids),
        "network_called": False,
    }
    print(canonical_json(result))
    return result


def cleanup_remote(args: argparse.Namespace) -> Dict[str, object]:
    _require_network_authority(args, paid_upload=False)
    if not args.confirm_delete_remote:
        raise ValueError("remote deletion additionally requires --confirm-delete-remote")
    job_dir = Path(args.job_dir).resolve()
    _verify_job(job_dir)
    paths = _job_paths(job_dir)
    state = read_json(paths["state"])
    client = _client(args)
    deleted: List[str] = []
    for key in ("input_file_id", "output_file_id", "error_file_id"):
        file_id = state.get(key)
        if file_id:
            client.delete_file(str(file_id))
            deleted.append(key)
    state["remote_files_deleted_at_utc"] = utc_now()
    state["remote_file_fields_deleted"] = deleted
    write_json_atomic(paths["state"], state, replace=True)
    result = {"status": "remote_files_deleted", "deleted_fields": deleted}
    print(canonical_json(result))
    return result


def _add_job_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--job-dir", required=True)


def _add_network(parser: argparse.ArgumentParser) -> None:
    _add_job_dir(parser)
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)


def _add_result_files(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--result-file", action="append", default=[])
    parser.add_argument("--error-file", action="append", default=[])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Target-blind Alibaba Qwen Batch staging and cache collection"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="offline: build immutable target-blind Batch JSONL")
    _add_job_dir(prepare)
    prepare.add_argument("--data-dir", default="data/ICEWS14")
    prepare.add_argument("--split", choices=["valid", "test"], required=True)
    prepare.add_argument("--shot", type=int, choices=sorted(SUPPORTED_SHOTS), required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument(
        "--history-protocol",
        choices=["standard_rolling_history", "strict_static_history"],
        default="standard_rolling_history",
    )
    prepare.add_argument("--history-len", type=int, default=16)
    prepare.add_argument("--max-candidates", type=int, default=10)
    prepare.add_argument("--model", default=DEFAULT_BATCH_MODEL)
    prepare.add_argument("--limit", type=int, default=0)
    prepare.add_argument("--omit-support", action="store_true")
    prepare.add_argument("--omit-history", action="store_true")
    prepare.add_argument(
        "--permute-support-order",
        action="store_true",
        help="prompt audit: preserve support facts but apply a deterministic non-identity order",
    )
    prepare.add_argument(
        "--replace-entity-names",
        action="store_true",
        help="prompt audit: replace every entity name by a deterministic opaque placeholder",
    )
    prepare.set_defaults(function=prepare_job)

    estimate = commands.add_parser("estimate", help="offline: estimate current Batch token cost")
    _add_job_dir(estimate)
    estimate.add_argument("--avg-prompt-tokens", type=float, default=705.0)
    estimate.add_argument("--avg-output-tokens", type=float, default=180.0)
    estimate.add_argument("--input-rate", type=float)
    estimate.add_argument("--output-rate", type=float)
    estimate.set_defaults(function=estimate_job)

    submit = commands.add_parser("submit", help="network: upload and create one paid Batch job")
    _add_network(submit)
    submit.add_argument("--completion-window", default="24h")
    submit.add_argument("--file-process-timeout", type=float, default=300.0)
    submit.add_argument("--poll-seconds", type=float, default=2.0)
    submit.set_defaults(function=submit_job)

    status = commands.add_parser("status", help="network: retrieve Batch job status")
    _add_network(status)
    status.set_defaults(function=status_job)

    cancel = commands.add_parser("cancel", help="network: cancel a queued or running Batch job")
    _add_network(cancel)
    cancel.set_defaults(function=cancel_job)

    download = commands.add_parser("download", help="network: download terminal output/error files")
    _add_network(download)
    download.set_defaults(function=download_job)

    collect = commands.add_parser("collect", help="offline: strictly map unordered results into a cache")
    _add_job_dir(collect)
    collect.add_argument("--data-dir", default="data/ICEWS14")
    collect.add_argument("--output", required=True)
    collect.add_argument("--allow-incomplete-cache", action="store_true")
    collect.add_argument("--fuzzy-threshold", type=float, default=0.90)
    collect.add_argument("--fuzzy-margin", type=float, default=0.04)
    _add_result_files(collect)
    collect.set_defaults(function=collect_job)

    retry = commands.add_parser("prepare-retry", help="offline: stage only missing/invalid requests")
    _add_job_dir(retry)
    retry.add_argument("--retry-dir", required=True)
    _add_result_files(retry)
    retry.set_defaults(function=prepare_retry_job)

    cleanup = commands.add_parser("cleanup-remote", help="network: explicitly delete retained remote files")
    _add_network(cleanup)
    cleanup.add_argument("--confirm-delete-remote", action="store_true")
    cleanup.set_defaults(function=cleanup_remote)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
