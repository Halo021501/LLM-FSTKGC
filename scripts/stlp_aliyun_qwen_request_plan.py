#!/usr/bin/env python3
"""Build and verify target-blind request plans for Alibaba Cloud Qwen realtime.

This module is an offline preprocessing utility. It never reads credentials,
opens a socket, trains a model, or evaluates a checkpoint.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aliyun_qwen_io import (
    DEFAULT_BASE_URL,
    DEFAULT_REALTIME_MODEL,
    PROVIDER_NAME,
    canonical_json,
    jsonl_bytes,
    parse_candidate_content,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    validate_qwen_requests,
    write_bytes_atomic,
    write_json_atomic,
)
from src.data import HistoryIndex, Quad, load_temporal_kg
from src.llm_cache import dataset_files_fingerprint, query_locator, target_blind_query_key
from src.stlp import (
    TargetBlindQuery,
    build_query_metadata,
    build_stlp_prompt,
    load_id_map,
    sha256_text,
)
from src.train import choose_causal_support, group_by_relation


PROMPT_TEMPLATE_VERSION = "stlp-aliyun-qwen-realtime-v1"
REQUEST_FILENAME = "requests.jsonl"
INDEX_FILENAME = "request_index.jsonl"
PLAN_FILENAME = "request_plan.json"
SUPPORTED_MODELS = {DEFAULT_REALTIME_MODEL}
SUPPORTED_SHOTS = {1, 3, 5, 10}


def initial_facts(kg, split: str) -> List[Quad]:
    if split == "train":
        return []
    if split == "valid":
        return list(kg.train_aug)
    return list(kg.train_aug) + list(kg.valid_aug)


def split_rows(kg, split: str) -> Sequence[Quad]:
    return {"train": kg.train, "valid": kg.valid, "test": kg.test}[split]


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
            f"unsupported Qwen model {args.model!r}; choose one of {sorted(SUPPORTED_MODELS)}"
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
        "purpose": "target-blind STLP Alibaba Cloud Qwen realtime request plan",
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
    for key in ("request", "index", "plan"):
        if paths[key].exists():
            raise FileExistsError(f"refusing to reuse a non-empty Qwen plan: {paths[key]}")
    args.job_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.job_dir, 0o700)
    write_bytes_atomic(paths["request"], jsonl_bytes(request_rows))
    write_bytes_atomic(paths["index"], jsonl_bytes(index_rows))
    write_json_atomic(paths["plan"], plan)
    validation = validate_qwen_requests(paths["request"])
    if validation["sha256"] != plan["request_sha256"]:
        raise RuntimeError("post-write Qwen request hash mismatch")
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
        raise ValueError("Qwen plan provider or official endpoint mismatch")
    request_validation = validate_qwen_requests(paths["request"])
    if request_validation["sha256"] != plan.get("request_sha256"):
        raise ValueError("Qwen request file hash differs from the immutable plan")
    request_rows = read_jsonl(paths["request"])
    index_rows = read_jsonl(paths["index"])
    if sha256_file(paths["index"]) != plan.get("index_sha256"):
        raise ValueError("Qwen index file hash differs from the immutable plan")
    if len(request_rows) != len(index_rows) or len(request_rows) != int(plan.get("request_count", -1)):
        raise ValueError("Qwen request/index count mismatch")
    request_by_id: Dict[str, Dict[str, object]] = {}
    index_by_id: Dict[str, Dict[str, object]] = {}
    for request, index in zip(request_rows, index_rows):
        custom_id = str(request.get("custom_id", ""))
        if custom_id != index.get("custom_id") or index.get("query_key") != custom_id:
            raise ValueError("Qwen request/index custom_id mismatch")
        query = index.get("query")
        if not isinstance(query, dict) or target_blind_query_key(query) != custom_id:
            raise ValueError(f"invalid target-blind query key in Qwen index: {custom_id}")
        body = request.get("body")
        messages = body.get("messages") if isinstance(body, dict) else None
        if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
            raise ValueError(f"invalid messages for Qwen request {custom_id}")
        prompt = messages[-1].get("content")
        if not isinstance(prompt, str) or sha256_text(prompt) != index.get("prompt_hash"):
            raise ValueError(f"prompt hash mismatch for Qwen request {custom_id}")
        if custom_id in request_by_id or custom_id in index_by_id:
            raise ValueError(f"duplicate Qwen custom_id: {custom_id}")
        request_by_id[custom_id] = request
        index_by_id[custom_id] = index
    return plan, request_rows, index_rows


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
                    raise ValueError(f"non-200 Qwen response: {response}")
                body = response.get("body")
                if not isinstance(body, dict):
                    raise ValueError("Qwen response body is missing")
                choices = body.get("choices")
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    raise ValueError("Qwen response has no first choice")
                choice = choices[0]
                finish_reason = choice.get("finish_reason")
                if finish_reason == "length":
                    raise ValueError("Qwen response was truncated at the output limit")
                if finish_reason not in (None, "stop"):
                    raise ValueError(f"unsupported finish_reason: {finish_reason!r}")
                message = choice.get("message")
                if not isinstance(message, dict):
                    raise ValueError("Qwen response choice has no message")
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
                raise ValueError(f"unknown custom_id in Qwen error file at {path}:{line_number}")
            if custom_id in seen_in_file:
                raise ValueError(f"duplicate custom_id in Qwen error file: {custom_id}")
            seen_in_file.add(custom_id)
            if custom_id not in successes:
                failures[custom_id] = f"Qwen error file entry: {row.get('error')}"
    return successes, failures




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an immutable target-blind Alibaba Cloud Qwen realtime request plan"
    )
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--data-dir", default="data/ICEWS14")
    parser.add_argument("--split", choices=["valid", "test"], required=True)
    parser.add_argument("--shot", type=int, choices=sorted(SUPPORTED_SHOTS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--history-protocol",
        choices=["standard_rolling_history", "strict_static_history"],
        default="standard_rolling_history",
    )
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_REALTIME_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--omit-support", action="store_true")
    parser.add_argument("--omit-history", action="store_true")
    parser.add_argument("--permute-support-order", action="store_true")
    parser.add_argument("--replace-entity-names", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    prepare_job(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
