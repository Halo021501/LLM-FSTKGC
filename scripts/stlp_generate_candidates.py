#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import os
import platform
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import HistoryIndex, Quad, load_temporal_kg
from src.llm_cache import (
    cache_file_sha256,
    canonical_json,
    dataset_files_fingerprint,
    query_locator,
    target_blind_query_key,
)
from src.stlp import (
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_LOCAL_QWEN_MODEL,
    DeepSeekClient,
    EntityMapper,
    LOCAL_QWEN_PROMPT_TEMPLATE_VERSION,
    LocalQwenClient,
    TargetBlindQuery,
    build_query_metadata,
    build_stlp_prompt,
    load_id_map,
    parse_and_map_response,
    sha256_text,
)
from src.train import choose_causal_support, group_by_relation


MIN_LOCAL_QWEN_TIMEOUT_SECONDS = 360.0
LOCAL_QWEN_ABSTENTION_POLICY = "provider_abstention_empty_llm_evidence"
FORMAL_LOCAL_QWEN_CACHE = (
    PROJECT_ROOT / "cache/standard_rolling_history/qwen2.5-7b-awq"
).resolve()
FORMAL_LOCAL_QWEN_REBOOT_INHIBIT = (
    PROJECT_ROOT
    / "logs/formal_qwen7b_stable_gpu3_20260809_102602/PRE_REBOOT_CHECKPOINT.lock"
)


def recent_public_history(index: HistoryIndex, known_entity_id: int, timestamp: int, limit: int) -> List[Quad]:
    rows = index.events_by_subject[known_entity_id]
    position = bisect.bisect_left(rows, (timestamp, -1, -1))
    selected = rows[max(0, position - limit) : position]
    return [(known_entity_id, relation, candidate, event_time) for event_time, relation, candidate in selected]


def mock_response(support: Sequence[Quad], entity_names: Sequence[str], max_candidates: int) -> str:
    """Deterministic target-blind provider for tests; never use it for reported results."""

    candidates = []
    seen = set()
    for _, _, candidate, _ in reversed(support):
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(
            {
                "entity_name": entity_names[candidate],
                "confidence": max(0.2, 0.8 - 0.05 * len(candidates)),
                "temporal_rationale": "Appears in strictly earlier support for the same oriented relation.",
                "temporal_consistency": 0.7,
            }
        )
        if len(candidates) >= max_candidates:
            break
    return json.dumps({"candidates": candidates}, ensure_ascii=False)


def existing_locators(path: str) -> set[Tuple[int, int, int]]:
    locators: set[Tuple[int, int, int]] = set()
    if not os.path.exists(path):
        return locators
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            query = record.get("query", {})
            if record.get("query_key") != target_blind_query_key(query):
                raise ValueError(f"invalid existing query key at {path}:{line_number}")
            locators.add(
                query_locator(query["known_entity_id"], query["oriented_relation_id"], query["timestamp"])
            )
    return locators


def append_record(path: str, record: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(canonical_json(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def provider_provenance(provider: str, client) -> Dict[str, object]:
    """Return immutable provider identity recorded beside every cache.

    A served model alias is not a reproducible model identity.  For the formal
    local provider, tie the cache to the exact repository revision, model
    manifest, tested runtime lock, and serving kernel recorded at release time.
    """

    model_alias = client.model if client else "deterministic-target-blind-mock"
    if provider == "mock":
        return {
            "model_alias": model_alias,
            "implementation": "deterministic-target-blind-mock",
            "network_provider_called": False,
        }
    if provider == "deepseek":
        return {
            "model_alias": model_alias,
            "exact_weight_revision_available": False,
            "provider_managed_model": True,
        }

    provenance_path = PROJECT_ROOT / "LLM_EXTENSION_PROVENANCE.json"
    with provenance_path.open("r", encoding="utf-8") as handle:
        release = json.load(handle)
    required = {
        "local_model_repository",
        "local_model_revision",
        "local_model_quantization",
        "local_model_manifest_sha256",
        "local_runtime_lock",
        "local_runtime_lock_sha256",
        "local_serving_profile",
    }
    missing = sorted(required.difference(release))
    if missing:
        raise ValueError(f"local provider provenance is missing fields: {missing}")

    lock_path = PROJECT_ROOT / str(release["local_runtime_lock"])
    if cache_file_sha256(str(lock_path)) != release["local_runtime_lock_sha256"]:
        raise ValueError("local Qwen runtime lock no longer matches release provenance")
    model_dir = Path(
        os.environ.get(
            "LOCAL_QWEN_MODEL_DIR",
            str(release.get("local_model_directory", "models/Qwen2.5-7B-Instruct-AWQ")),
        )
    )
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    model_manifest_path = model_dir / "MODEL_MANIFEST.sha256"
    if cache_file_sha256(str(model_manifest_path)) != release["local_model_manifest_sha256"]:
        raise ValueError("local Qwen model manifest no longer matches release provenance")

    serving_profile = dict(release["local_serving_profile"])
    serving_profile["quantization_kernel"] = os.environ.get(
        "LOCAL_QWEN_QUANTIZATION", str(serving_profile["quantization_kernel"])
    )
    return {
        "model_alias": model_alias,
        "model_repository": release["local_model_repository"],
        "model_revision": release["local_model_revision"],
        "model_quantization": release["local_model_quantization"],
        "model_manifest_sha256": release["local_model_manifest_sha256"],
        "runtime_lock_sha256": release["local_runtime_lock_sha256"],
        "serving_profile": serving_profile,
        "release_provenance_sha256": cache_file_sha256(str(provenance_path)),
    }


def ensure_cache_metadata(
    output_path: str,
    invariant_meta: Dict[str, object],
    generation_audit: Dict[str, object],
    *,
    resume: bool,
) -> None:
    """Create metadata or reject a resume under a different protocol.

    Generation time and the original command are audit facts from the first
    session.  Every other field is an invariant and must match exactly before
    any existing locator may be skipped.
    """

    meta_path = output_path + ".meta.json"
    output_exists = os.path.exists(output_path)
    meta_exists = os.path.exists(meta_path)
    if not resume and (output_exists or meta_exists):
        existing = output_path if output_exists else meta_path
        raise FileExistsError(f"output already exists; pass --resume or choose a new path: {existing}")
    if output_exists and not meta_exists:
        raise FileNotFoundError(f"cannot resume cache without metadata: {meta_path}")

    if meta_exists:
        with open(meta_path, "r", encoding="utf-8") as handle:
            existing_meta = json.load(handle)
        if not isinstance(existing_meta, dict):
            raise ValueError(f"cache metadata must be a JSON object: {meta_path}")
        existing_invariants = {
            key: value for key, value in existing_meta.items() if key != "generation_audit"
        }
        differing = sorted(
            key
            for key in set(existing_invariants).union(invariant_meta)
            if existing_invariants.get(key) != invariant_meta.get(key)
        )
        if differing:
            raise ValueError(
                f"resume metadata mismatch at {meta_path}; differing fields: {differing}"
            )
        if not isinstance(existing_meta.get("generation_audit"), dict):
            raise ValueError(f"cache metadata is missing generation_audit: {meta_path}")
        return

    os.makedirs(os.path.dirname(os.path.abspath(meta_path)), exist_ok=True)
    metadata = dict(invariant_meta)
    metadata["generation_audit"] = generation_audit
    with open(meta_path, "x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def initial_facts(kg, split: str) -> List[Quad]:
    if split == "train":
        return []
    if split == "valid":
        return list(kg.train_aug)
    return list(kg.train_aug) + list(kg.valid_aug)


def split_rows(kg, split: str) -> Sequence[Quad]:
    return {"train": kg.train, "valid": kg.valid, "test": kg.test}[split]


def make_api_response(client, prompt: str, args) -> Dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(args.max_retries + 1):
        try:
            max_tokens = (
                args.retry_max_tokens
                if attempt > 0 and args.retry_max_tokens > 0
                else args.max_tokens
            )
            return client.complete_json(prompt, max_tokens=max_tokens)
        except Exception as exc:
            last_error = exc
            if attempt >= args.max_retries:
                break
            time.sleep(min(10.0, 1.5 * (2**attempt)))
    assert last_error is not None
    raise last_error


def local_qwen_abstention(error: Exception, attempts: int) -> Dict[str, object] | None:
    """Classify an exhausted per-query local-provider failure without raw text.

    Only failures emitted while making or validating a LocalQwenClient request
    are eligible.  Unrelated programming errors must still fail the shard so
    that they are not silently converted into empty evidence.  The serialized
    audit intentionally records a stable code and exception type rather than a
    potentially large or environment-specific provider error message.
    """

    message = str(error)
    lowered = message.casefold()
    if isinstance(error, json.JSONDecodeError):
        code = "invalid_json_exhausted"
    elif isinstance(error, (TimeoutError, ConnectionError, OSError)):
        code = "transport_exhausted"
    elif isinstance(error, RuntimeError) and message.startswith("Local Qwen "):
        if "finish_reason='length'" in lowered or 'finish_reason="length"' in lowered:
            code = "length_exhausted"
        elif "invalid json" in lowered:
            code = "invalid_json_exhausted"
        elif message.startswith("Local Qwen HTTP ") or message.startswith(
            "Local Qwen request failed:"
        ):
            code = "transport_exhausted"
        else:
            code = "invalid_response_exhausted"
    else:
        return None
    return {
        "schema_version": 1,
        "policy": LOCAL_QWEN_ABSTENTION_POLICY,
        "code": code,
        "attempts": int(attempts),
        "error_type": type(error).__name__,
    }


def make_local_qwen_response(client, prompt: str, args) -> Dict[str, object]:
    """Return a provider response or an auditable empty-evidence abstention."""

    try:
        return make_api_response(client, prompt, args)
    except Exception as exc:
        abstention = local_qwen_abstention(exc, args.max_retries + 1)
        if abstention is None:
            raise
        print(
            canonical_json(
                {
                    "event": "local_qwen_provider_abstention",
                    "policy": abstention["policy"],
                    "code": abstention["code"],
                    "attempts": abstention["attempts"],
                }
            ),
            flush=True,
        )
        return {
            "content": '{"candidates":[]}',
            "latency_ms": 0.0,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "response_id": None,
            "model": client.model,
            "provider_abstention": abstention,
        }


def run(args: argparse.Namespace) -> None:
    if args.provider == "deepseek" and not args.execute_api:
        raise ValueError("DeepSeek calls require explicit --execute-api; omit it for a no-network dry setup")
    if args.shot not in {1, 3, 5, 10}:
        raise ValueError("the extended task protocol permits only shot 1, 3, 5, or 10")
    if args.history_protocol not in {"standard_rolling_history", "strict_static_history"}:
        raise ValueError("unknown history protocol")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, --num-shards)")
    output_path = Path(args.output).resolve()
    if (
        FORMAL_LOCAL_QWEN_CACHE in output_path.parents
        and FORMAL_LOCAL_QWEN_REBOOT_INHIBIT.exists()
    ):
        raise RuntimeError(
            "pre-reboot checkpoint inhibits all writes to the formal local-Qwen cache: "
            f"{FORMAL_LOCAL_QWEN_REBOOT_INHIBIT}"
        )
    random.seed(args.seed)

    kg = load_temporal_kg(args.data_dir)
    _, entity_names = load_id_map(
        os.path.join(args.data_dir, "entity2id.txt"), expected_size=kg.num_entities
    )
    _, relation_names = load_id_map(
        os.path.join(args.data_dir, "relation2id.txt"), expected_size=kg.num_relations
    )
    mapper = EntityMapper(
        {name: index for index, name in enumerate(entity_names)},
        fuzzy_threshold=args.fuzzy_threshold,
        fuzzy_margin=args.fuzzy_margin,
    )
    fingerprint = dataset_files_fingerprint(args.data_dir)
    base_facts = initial_facts(kg, args.split)
    history = HistoryIndex(
        base_facts,
        kg.num_entities,
        kg.num_relations * 2,
        history_len=max(args.history_len, 16),
    )
    support_by_relation = group_by_relation(base_facts)
    requested_timeout = float(args.timeout)
    effective_timeout = (
        max(requested_timeout, MIN_LOCAL_QWEN_TIMEOUT_SECONDS)
        if args.provider == "local_qwen"
        else requested_timeout
    )
    if args.provider == "deepseek":
        client = DeepSeekClient.from_environment(timeout_seconds=effective_timeout)
    elif args.provider == "local_qwen":
        client = LocalQwenClient.from_environment(timeout_seconds=effective_timeout)
    else:
        client = None
    meta = {
        "schema_version": 2,
        "purpose": "target-blind STLP candidate cache",
        "split": args.split,
        "shot": args.shot,
        "seed": args.seed,
        "history_protocol": args.history_protocol,
        "provider": args.provider,
        "model": client.model if client else "deterministic-target-blind-mock",
        "provider_provenance": provider_provenance(args.provider, client),
        "dataset_fingerprint": fingerprint,
        "query_key_excludes_target": True,
        "api_called_inside_training_or_evaluation": False,
        "prompt_ablation": {
            "omit_support": args.omit_support,
            "omit_history": args.omit_history,
        },
        "decoding": {
            "max_candidates": args.max_candidates,
            "max_tokens": args.max_tokens,
            "retry_max_tokens": args.retry_max_tokens,
            "max_retries": args.max_retries,
        },
        "partition": {
            "num_shards": args.num_shards,
            "shard_id": args.shard_id,
            "assignment": "canonical_unique_locator_ordinal_mod",
        },
    }
    generation_audit = {
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command_argv": [sys.executable, *sys.argv],
        "hostname": platform.node(),
        "physical_gpu_id": os.environ.get("LOCAL_QWEN_GPU_ID")
        if args.provider == "local_qwen"
        else os.environ.get("CUDA_VISIBLE_DEVICES"),
        "requested_request_timeout_seconds": requested_timeout,
        "effective_request_timeout_seconds": effective_timeout,
    }
    if effective_timeout != requested_timeout:
        print(
            canonical_json(
                {
                    "event": "local_qwen_timeout_floor_applied",
                    "requested_timeout_seconds": requested_timeout,
                    "effective_timeout_seconds": effective_timeout,
                }
            ),
            flush=True,
        )
    ensure_cache_metadata(args.output, meta, generation_audit, resume=args.resume)
    completed = existing_locators(args.output) if args.resume else set()

    snapshots: Dict[int, List[Quad]] = defaultdict(list)
    for row in split_rows(kg, args.split):
        snapshots[row[3]].append(row)

    written = 0
    visited = 0
    unique_ordinal = 0
    seen_locators: set[Tuple[int, int, int]] = set()
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
            assigned_to_shard = unique_ordinal % args.num_shards == args.shard_id
            unique_ordinal += 1
            if not assigned_to_shard:
                continue
            if locator in completed:
                continue
            completed.add(locator)
            visited += 1
            if args.limit and visited > args.limit:
                break
            direction = "tail" if oriented_relation_id < kg.num_relations else "head"
            public_query = TargetBlindQuery(
                split=args.split,
                direction=direction,
                known_entity_id=known_entity_id,
                oriented_relation_id=oriented_relation_id,
                timestamp=event_time,
            )
            # The helper ignores _hidden_target and reads only s, r, and t.
            relation_rows = support_by_relation.get(oriented_relation_id, [])
            if relation_rows and relation_rows[0][3] < event_time:
                support = choose_causal_support(
                    support_by_relation,
                    (known_entity_id, oriented_relation_id, 0, event_time),
                    args.shot,
                )
            else:
                # The parent model uses a neutral structural placeholder when no
                # support exists.  An LLM prompt must not present that synthetic
                # row as a real historical fact, so the prompt support is empty.
                support = []
            recent = recent_public_history(history, known_entity_id, event_time, args.history_len)
            prompt_support = [] if args.omit_support else support
            prompt_history = [] if args.omit_history else recent
            prompt_variant = (
                LOCAL_QWEN_PROMPT_TEMPLATE_VERSION
                if args.provider == "local_qwen"
                else "stlp-deepseek-v1"
            )
            if args.omit_support:
                prompt_variant += "-no-support"
            if args.omit_history:
                prompt_variant += "-no-history"
            prompt = build_stlp_prompt(
                public_query,
                prompt_support,
                prompt_history,
                entity_names,
                relation_names,
                kg.num_relations,
                max_candidates=args.max_candidates,
                compact_response_keys=args.provider == "local_qwen",
            )
            query_metadata = build_query_metadata(
                public_query,
                shot=args.shot,
                seed=args.seed,
                history_protocol=args.history_protocol,
                support=prompt_support,
                history=prompt_history,
                dataset_fingerprint=fingerprint,
                prompt_template_version=prompt_variant,
            )
            if client is None:
                response = {
                    "content": mock_response(support, entity_names, args.max_candidates),
                    "latency_ms": 0.0,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "response_id": None,
                    "model": "deterministic-target-blind-mock",
                }
            elif args.provider == "local_qwen":
                response = make_local_qwen_response(client, prompt, args)
            else:
                response = make_api_response(client, prompt, args)
            # Prompt ablation must not also ablate template_agreement.
            # Preserve the original causal support names for response mapping
            # and downstream feature construction even when the prompt omits
            # support examples.
            support_names = [entity_names[row[2]] for row in support]
            candidates, diagnostics = parse_and_map_response(
                str(response["content"]),
                mapper,
                support_names,
                max_candidates=args.max_candidates,
            )
            abstention = response.get("provider_abstention")
            if isinstance(abstention, dict):
                diagnostics = dict(diagnostics)
                diagnostics["provider_abstention"] = 1.0
                diagnostics["provider_abstention_attempts"] = float(abstention["attempts"])
                diagnostics[f"provider_abstention_{abstention['code']}"] = 1.0
            usage = response.get("usage", {})
            record: Dict[str, object] = {
                "schema_version": 1,
                "query_key": target_blind_query_key(query_metadata),
                "query": query_metadata,
                "prompt_hash": sha256_text(prompt),
                "provider": args.provider,
                "model": response.get(
                    "model",
                    DEFAULT_LOCAL_QWEN_MODEL if args.provider == "local_qwen" else DEFAULT_DEEPSEEK_MODEL,
                ),
                "response_id": response.get("response_id"),
                "candidates": candidates,
                "diagnostics": diagnostics,
                "latency_ms": float(response.get("latency_ms", 0.0)),
                "token_usage": {
                    "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                    "total_tokens": int(usage.get("total_tokens", 0) or 0),
                },
            }
            if isinstance(abstention, dict):
                record["provider_abstention"] = dict(abstention)
            # No target id, target name, raw prompt, or hidden answer is serialized.
            append_record(args.output, record)
            written += 1
            if written % args.progress_every == 0:
                print(f"written={written} visited={visited} output={args.output}", flush=True)
            if args.provider == "deepseek" and args.request_interval > 0:
                time.sleep(args.request_interval)
        if args.limit and visited >= args.limit:
            break
        if args.history_protocol == "standard_rolling_history" or args.split == "train":
            history.add_facts(oriented_rows)
            for fact in oriented_rows:
                support_by_relation.setdefault(fact[1], []).append(fact)

    print(
        canonical_json(
            {
                "status": "complete",
                "new_records": written,
                "provider": args.provider,
                "output": os.path.abspath(args.output),
                "target_blind": True,
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
            }
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate target-blind STLP JSONL candidates")
    parser.add_argument("--data-dir", default="data/ICEWS14")
    parser.add_argument("--split", choices=["train", "valid", "test"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shot", type=int, choices=[1, 3, 5, 10], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--history-protocol",
        choices=["standard_rolling_history", "strict_static_history"],
        default="standard_rolling_history",
    )
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--provider", choices=["mock", "deepseek", "local_qwen"], default="mock")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument(
        "--retry-max-tokens",
        type=int,
        default=0,
        help="optional larger output budget used only after a failed request",
    )
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--request-interval", type=float, default=0.2)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.90)
    parser.add_argument("--fuzzy-margin", type=float, default=0.04)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--omit-support", action="store_true", help="prompt ablation; cache separately")
    parser.add_argument("--omit-history", action="store_true", help="prompt ablation; cache separately")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
