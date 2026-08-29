#!/usr/bin/env python3
"""Concurrent, rate-limited Alibaba Qwen realtime cache generation.

The runner consumes already-reviewed target-blind request/index plans, writes
crash-safe raw responses, and can resume only missing custom_ids.  Collection
is offline and produces the provider-independent cache contract used by v5.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import random
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stlp_aliyun_qwen_batch import _classify_results, _verify_job
from src.aliyun_qwen_batch import (
    canonical_json,
    jsonl_bytes,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    write_bytes_atomic,
    write_json_atomic,
)
from src.aliyun_qwen_realtime import (
    DEFAULT_REALTIME_MODEL,
    OFFICIAL_FLOATING_MODEL_RPM,
    OFFICIAL_FLOATING_MODEL_TPM,
    PROVIDER_NAME,
    AliyunQwenRealtimeClient,
    RealtimeAPIError,
    SlidingWindowRateLimiter,
)
from src.data import load_temporal_kg
from src.llm_cache import LLMEvidenceCache, dataset_files_fingerprint
from src.stlp import EntityMapper, load_id_map, parse_and_map_response


UTC = dt.timezone.utc
RESPONSE_FILENAME = "realtime_responses.jsonl"
FAILURE_FILENAME = "realtime_failures.jsonl"
ABSTENTION_FILENAME = "realtime_abstentions.jsonl"
STATE_FILENAME = "realtime_state.json"
INSPECTION_ERROR_CODE = "data_inspection_failed"
ABSTENTION_POLICY = "provider_abstention_empty_llm_evidence"
UNEXPECTED_WORKER_ERROR_CODE = "unexpected_worker_exception"
REALTIME_INPUT_RATE_CNY_PER_MILLION = 0.15
REALTIME_OUTPUT_RATE_CNY_PER_MILLION = 1.50


def utc_now() -> str:
    return dt.datetime.now(UTC).isoformat()


def _paths(run_dir: Path) -> Dict[str, Path]:
    return {
        "response": run_dir / RESPONSE_FILENAME,
        "failure": run_dir / FAILURE_FILENAME,
        "abstention": run_dir / ABSTENTION_FILENAME,
        "state": run_dir / STATE_FILENAME,
    }


def _append_jsonl(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    payload = (canonical_json(dict(row)) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(descriptor, "ab", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if path.exists():
            os.chmod(path, 0o600)


def _require_network_authority(args: argparse.Namespace) -> None:
    if not args.execute_api:
        raise ValueError("realtime calls require explicit --execute-api")
    if os.environ.get("CONFIRM_ALIYUN_QWEN_DATA_UPLOAD", "NO") != "YES":
        raise ValueError("set CONFIRM_ALIYUN_QWEN_DATA_UPLOAD=YES after reviewing prompts")
    if os.environ.get("CONFIRM_ALIYUN_QWEN_PAID_REALTIME", "NO") != "YES":
        raise ValueError("set CONFIRM_ALIYUN_QWEN_PAID_REALTIME=YES after reviewing pricing")


def _existing_results(path: Path, expected_ids: set[str], max_candidates: int) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or custom_id not in expected_ids:
            raise ValueError(f"unknown custom_id in existing realtime output line {line_number}")
        if custom_id in seen:
            raise ValueError(f"duplicate custom_id in existing realtime output: {custom_id}")
        seen.add(custom_id)
    successes, failures = _classify_results([path], [], expected_ids, max_candidates=max_candidates)
    invalid_seen = sorted(seen.intersection(failures))
    if invalid_seen:
        raise ValueError(f"existing realtime output contains invalid rows: {invalid_seen[:3]}")
    return successes


def _existing_abstentions(
    path: Path, expected_ids: set[str]
) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    abstentions: Dict[str, Dict[str, object]] = {}
    required = {
        "schema_version",
        "custom_id",
        "provider",
        "code",
        "status_code",
        "attempts",
        "observed_at_utc",
        "policy",
        "attempt_audit",
    }
    for line_number, row in enumerate(read_jsonl(path), start=1):
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or custom_id not in expected_ids:
            raise ValueError(
                f"unknown custom_id in existing realtime abstention line {line_number}"
            )
        if custom_id in abstentions:
            raise ValueError(f"duplicate realtime abstention custom_id: {custom_id}")
        if set(row) != required:
            raise ValueError(
                f"invalid realtime abstention schema at line {line_number}"
            )
        if row.get("schema_version") != 1 or row.get("provider") != PROVIDER_NAME:
            raise ValueError(
                f"invalid realtime abstention provenance at line {line_number}"
            )
        if row.get("code") != INSPECTION_ERROR_CODE or row.get("policy") != ABSTENTION_POLICY:
            raise ValueError(
                f"unsupported realtime abstention reason at line {line_number}"
            )
        if row.get("status_code") != 400:
            raise ValueError(
                f"invalid realtime abstention status at line {line_number}"
            )
        if not isinstance(row.get("attempts"), int) or int(row["attempts"]) < 1:
            raise ValueError(
                f"invalid realtime abstention attempt count at line {line_number}"
            )
        if not isinstance(row.get("observed_at_utc"), str) or not row["observed_at_utc"]:
            raise ValueError(
                f"invalid realtime abstention timestamp at line {line_number}"
            )
        if not isinstance(row.get("attempt_audit"), list):
            raise ValueError(
                f"invalid realtime abstention attempt audit at line {line_number}"
            )
        abstentions[custom_id] = dict(row)
    return abstentions


def _backoff_seconds(
    custom_id: str,
    attempt: int,
    *,
    base_seconds: float,
    max_seconds: float,
    retry_after_seconds: float | None,
) -> float:
    if retry_after_seconds is not None:
        return min(max_seconds, max(0.0, retry_after_seconds))
    ceiling = min(max_seconds, base_seconds * (2 ** max(0, attempt - 1)))
    seed = int(custom_id[:16], 16) ^ attempt
    return ceiling * random.Random(seed).uniform(0.75, 1.25)


def _call_one(
    row: Mapping[str, object],
    *,
    client: AliyunQwenRealtimeClient,
    limiter: SlidingWindowRateLimiter,
    max_attempts: int,
    inspection_max_attempts: int,
    backoff_base: float,
    backoff_max: float,
    stop_event: threading.Event,
    sleep_fn=time.sleep,
) -> Tuple[Dict[str, object] | None, Dict[str, object] | None]:
    custom_id = str(row["custom_id"])
    errors = []
    for attempt in range(1, max_attempts + 1):
        if stop_event.is_set():
            return None, {
                "custom_id": custom_id,
                "code": "cancelled_after_fatal_peer_error",
                "attempts": attempt - 1,
                "fatal": False,
            }
        limiter.acquire()
        try:
            result = client.complete(row["body"])
            body = result["body"]
            return {
                "id": body.get("id"),
                "custom_id": custom_id,
                "response": {
                    "status_code": 200,
                    "request_id": body.get("id"),
                    "body": body,
                },
                "error": None,
                "realtime_audit": {
                    "attempts": attempt,
                    "latency_ms": float(result["latency_ms"]),
                    "completed_at_utc": utc_now(),
                    "retry_codes": [item["code"] for item in errors],
                },
            }, None
        except RealtimeAPIError as exc:
            errors.append(
                {
                    "attempt": attempt,
                    "status_code": exc.status_code,
                    "code": exc.code,
                    "retriable": exc.retriable,
                }
            )
            inspection_rejection = (
                exc.status_code == 400
                and str(exc.code).casefold() == INSPECTION_ERROR_CODE
            )
            if inspection_rejection:
                if attempt >= min(max_attempts, inspection_max_attempts):
                    return None, {
                        "custom_id": custom_id,
                        "code": INSPECTION_ERROR_CODE,
                        "status_code": 400,
                        "attempts": attempt,
                        "fatal": False,
                        "provider_abstention": True,
                        "policy": ABSTENTION_POLICY,
                        "observed_at_utc": utc_now(),
                        "attempt_audit": errors,
                    }
                delay = _backoff_seconds(
                    custom_id,
                    attempt,
                    base_seconds=backoff_base,
                    max_seconds=backoff_max,
                    retry_after_seconds=exc.retry_after_seconds,
                )
                sleep_fn(delay)
                continue
            fatal = exc.status_code in {400, 401, 403, 404}
            if fatal or not exc.retriable or attempt >= max_attempts:
                return None, {
                    "custom_id": custom_id,
                    "code": exc.code,
                    "status_code": exc.status_code,
                    "attempts": attempt,
                    "fatal": fatal,
                    "retriable_exhausted": bool(exc.retriable and not fatal),
                    "observed_at_utc": utc_now(),
                    "attempt_audit": errors,
                }
            delay = _backoff_seconds(
                custom_id,
                attempt,
                base_seconds=backoff_base,
                max_seconds=backoff_max,
                retry_after_seconds=exc.retry_after_seconds,
            )
            if exc.status_code == 429:
                limiter.defer(delay)
            sleep_fn(delay)
    raise AssertionError("unreachable retry loop")


def _state_payload(
    *,
    plan: Mapping[str, object],
    args: argparse.Namespace,
    successful: int,
    abstained: int,
    failed: int,
    status: str,
    started_at: str,
    last_progress_at: str | None = None,
    in_flight: int = 0,
    not_submitted: int = 0,
    fatal_code: str | None = None,
) -> Dict[str, object]:
    completed = int(successful) + int(abstained)
    heartbeat_at = utc_now()
    return {
        "schema_version": 1,
        "provider": PROVIDER_NAME,
        "model": args.model,
        "status": status,
        "source_job_dir": str(Path(args.source_job_dir).resolve()),
        "source_request_sha256": plan["request_sha256"],
        "source_index_sha256": plan["index_sha256"],
        "started_at_utc": started_at,
        "updated_at_utc": heartbeat_at,
        "heartbeat_at_utc": heartbeat_at,
        "last_progress_at_utc": last_progress_at or started_at,
        "session_id": started_at,
        "request_counts": {
            "total": int(plan["request_count"]),
            "completed": completed,
            "successful_responses": int(successful),
            "provider_abstentions": int(abstained),
            "failed_last_session": int(failed),
            "remaining": int(plan["request_count"]) - completed,
            "in_flight": int(in_flight),
            "not_submitted": int(not_submitted),
        },
        "transport": {
            "workers": args.workers,
            "max_rpm": args.max_rpm,
            "max_tpm": args.max_tpm,
            "token_reservation": args.token_reservation,
            "max_attempts": args.max_attempts,
            "inspection_max_attempts": args.inspection_max_attempts,
            "provider_abstention_policy": ABSTENTION_POLICY,
            "timeout_seconds": args.timeout,
            "backoff_base_seconds": args.backoff_base,
            "backoff_max_seconds": args.backoff_max,
            "heartbeat_seconds": args.heartbeat_seconds,
            "bounded_in_flight": True,
        },
        "fatal_code": fatal_code,
    }


def run_job(args: argparse.Namespace) -> Dict[str, object]:
    _require_network_authority(args)
    if args.model != DEFAULT_REALTIME_MODEL:
        raise ValueError("the reviewed realtime experiment uses qwen-flash")
    if args.workers < 1 or args.workers > 64:
        raise ValueError("--workers must be between 1 and 64")
    if args.max_rpm > OFFICIAL_FLOATING_MODEL_RPM:
        raise ValueError("configured RPM exceeds the documented qwen-flash Beijing limit")
    if args.max_tpm > OFFICIAL_FLOATING_MODEL_TPM:
        raise ValueError("configured TPM exceeds the documented qwen-flash Beijing limit")
    if args.max_attempts < 1 or args.max_attempts > 10:
        raise ValueError("--max-attempts must be between 1 and 10")
    if args.inspection_max_attempts < 1 or args.inspection_max_attempts > args.max_attempts:
        raise ValueError("--inspection-max-attempts must be between 1 and --max-attempts")
    if args.heartbeat_seconds <= 0.0:
        raise ValueError("--heartbeat-seconds must be positive")

    source_job_dir = Path(args.source_job_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    plan, request_rows, index_rows = _verify_job(source_job_dir)
    if str(plan["model"]) != args.model:
        raise ValueError("prepared source plan model differs from realtime model")
    expected_ids = {str(row["custom_id"]) for row in index_rows}
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(run_dir, 0o700)
    paths = _paths(run_dir)
    if (paths["state"].exists() or paths["response"].exists()) and not args.resume:
        raise FileExistsError("realtime artifacts exist; use --resume after reviewing them")
    completed = _existing_results(
        paths["response"], expected_ids, max_candidates=int(plan["max_candidates"])
    )
    abstentions = _existing_abstentions(paths["abstention"], expected_ids)
    overlap = sorted(set(completed).intersection(abstentions))
    if overlap:
        raise ValueError(
            f"custom_id cannot be both a realtime success and abstention: {overlap[:3]}"
        )
    if paths["state"].exists():
        prior = read_json(paths["state"])
        if prior.get("source_request_sha256") != plan["request_sha256"]:
            raise ValueError("resume source request hash differs from the prior realtime state")
        if prior.get("source_index_sha256") != plan["index_sha256"]:
            raise ValueError("resume source index hash differs from the prior realtime state")
        if prior.get("model") != args.model:
            raise ValueError("resume model differs from the prior realtime state")
    covered_ids = set(completed).union(abstentions)
    pending = [row for row in request_rows if str(row["custom_id"]) not in covered_ids]
    started_at = utc_now()
    write_json_atomic(
        paths["state"],
        _state_payload(
            plan=plan,
            args=args,
            successful=len(completed),
            abstained=len(abstentions),
            failed=0,
            status="running",
            started_at=started_at,
            last_progress_at=started_at,
            not_submitted=len(pending),
        ),
        replace=paths["state"].exists(),
    )
    if not pending:
        state = _state_payload(
            plan=plan,
            args=args,
            successful=len(completed),
            abstained=len(abstentions),
            failed=0,
            status="completed",
            started_at=started_at,
        )
        write_json_atomic(paths["state"], state, replace=True)
        print(canonical_json(state))
        return state

    client = AliyunQwenRealtimeClient.from_environment(
        model=args.model, timeout_seconds=args.timeout
    )
    limiter = SlidingWindowRateLimiter(
        max_rpm=args.max_rpm,
        max_tpm=args.max_tpm,
        token_reservation=args.token_reservation,
    )
    stop_event = threading.Event()
    failed = 0
    fatal_failure: Dict[str, object] | None = None
    last_progress_at = started_at
    pending_rows = iter(pending)
    not_submitted = len(pending)
    futures: Dict[Future, str] = {}
    executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="qwen-rt")

    def submit_window() -> None:
        nonlocal not_submitted
        while fatal_failure is None and len(futures) < args.workers:
            try:
                row = next(pending_rows)
            except StopIteration:
                return
            future = executor.submit(
                _call_one,
                row,
                client=client,
                limiter=limiter,
                max_attempts=args.max_attempts,
                inspection_max_attempts=args.inspection_max_attempts,
                backoff_base=args.backoff_base,
                backoff_max=args.backoff_max,
                stop_event=stop_event,
            )
            futures[future] = str(row["custom_id"])
            not_submitted -= 1

    def write_runtime_state(status: str) -> Dict[str, object]:
        state = _state_payload(
            plan=plan,
            args=args,
            successful=len(completed),
            abstained=len(abstentions),
            failed=failed,
            status=status,
            started_at=started_at,
            last_progress_at=last_progress_at,
            in_flight=len(futures),
            not_submitted=not_submitted,
            fatal_code=(str(fatal_failure.get("code")) if fatal_failure else None),
        )
        write_json_atomic(paths["state"], state, replace=True)
        return state

    try:
        submit_window()
        write_runtime_state("running")
        while futures:
            done, _ = wait(
                tuple(futures),
                timeout=args.heartbeat_seconds,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                state = write_runtime_state("failed" if fatal_failure else "running")
                print(
                    canonical_json(
                        {
                            "event": "heartbeat",
                            "completed": len(completed) + len(abstentions),
                            "in_flight": len(futures),
                            "not_submitted": not_submitted,
                            "last_progress_at_utc": state["last_progress_at_utc"],
                        }
                    ),
                    flush=True,
                )
                continue

            covered_before = len(completed) + len(abstentions)
            for future in done:
                custom_id = futures.pop(future)
                try:
                    result, failure = future.result()
                except Exception as exc:
                    result = None
                    failure = {
                        "custom_id": custom_id,
                        "code": UNEXPECTED_WORKER_ERROR_CODE,
                        "exception_type": type(exc).__name__,
                        "attempts": 0,
                        "fatal": True,
                        "retriable_exhausted": False,
                        "observed_at_utc": utc_now(),
                        "attempt_audit": [],
                    }

                if result is not None:
                    _append_jsonl(paths["response"], result)
                    completed[str(result["custom_id"])] = result
                elif failure is not None:
                    if bool(failure.get("provider_abstention")):
                        abstention = {
                            "schema_version": 1,
                            "custom_id": str(failure["custom_id"]),
                            "provider": PROVIDER_NAME,
                            "code": INSPECTION_ERROR_CODE,
                            "status_code": 400,
                            "attempts": int(failure["attempts"]),
                            "observed_at_utc": str(failure["observed_at_utc"]),
                            "policy": ABSTENTION_POLICY,
                            "attempt_audit": failure["attempt_audit"],
                        }
                        _append_jsonl(paths["abstention"], abstention)
                        abstentions[str(abstention["custom_id"])] = abstention
                    else:
                        _append_jsonl(paths["failure"], failure)
                        failed += 1
                        if bool(failure.get("fatal")) and fatal_failure is None:
                            fatal_failure = failure
                            stop_event.set()
                last_progress_at = utc_now()

            if fatal_failure is not None:
                # The window is bounded to workers, so a fatal event can leave
                # at most workers-1 live calls.  Cancel requests which have not
                # started and keep consuming every running result durably.
                for future in list(futures):
                    if future.cancel():
                        futures.pop(future)
            else:
                # Submit only after every finished outcome above is fsynced.
                submit_window()

            write_runtime_state("failed" if fatal_failure else "running")
            covered = len(completed) + len(abstentions)
            if covered > covered_before and (
                covered // max(1, args.progress_every)
                > covered_before // max(1, args.progress_every)
            ):
                print(
                    canonical_json(
                        {
                            "event": "progress",
                            "completed": covered,
                            "successful_responses": len(completed),
                            "provider_abstentions": len(abstentions),
                            "total": len(request_rows),
                            "failed_last_session": failed,
                        }
                    ),
                    flush=True,
                )
    finally:
        if sys.exc_info()[0] is not None:
            stop_event.set()
            for future in futures:
                future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)

    covered = len(completed) + len(abstentions)
    final_status = "completed" if covered == len(request_rows) else "partial_failure"
    if fatal_failure is not None:
        final_status = "failed"
    state = _state_payload(
        plan=plan,
        args=args,
        successful=len(completed),
        abstained=len(abstentions),
        failed=failed,
        status=final_status,
        started_at=started_at,
        last_progress_at=last_progress_at,
        in_flight=0,
        not_submitted=not_submitted,
        fatal_code=(str(fatal_failure.get("code")) if fatal_failure else None),
    )
    write_json_atomic(paths["state"], state, replace=True)
    print(canonical_json(state), flush=True)
    if final_status != "completed":
        raise RuntimeError(
            f"realtime job ended {final_status}: completed={covered}/{len(request_rows)}; "
            "review the sanitized failure audit and rerun with --resume"
        )
    return state


def collect_job(args: argparse.Namespace) -> Dict[str, object]:
    source_job_dir = Path(args.source_job_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output).resolve()
    meta_output = Path(str(output) + ".meta.json")
    if output.exists() or meta_output.exists():
        raise FileExistsError(f"refusing to overwrite realtime cache artifacts: {output}")
    plan, _, index_rows = _verify_job(source_job_dir)
    if not bool(plan["complete_split"]) and not args.allow_incomplete_cache:
        raise ValueError("smoke/pilot collection requires --allow-incomplete-cache")
    run_paths = _paths(run_dir)
    response_path = run_paths["response"]
    abstention_path = run_paths["abstention"]
    expected_ids = {str(row["custom_id"]) for row in index_rows}
    successes = _existing_results(
        response_path, expected_ids, max_candidates=int(plan["max_candidates"])
    )
    abstentions = _existing_abstentions(abstention_path, expected_ids)
    overlap = sorted(set(successes).intersection(abstentions))
    if overlap:
        raise ValueError(
            f"custom_id cannot be both a realtime success and abstention: {overlap[:3]}"
        )
    resolved_ids = set(successes).union(abstentions)
    unresolved = sorted(expected_ids.difference(resolved_ids))
    if unresolved:
        raise ValueError(
            f"realtime collection incomplete: {len(unresolved)} unresolved; {unresolved[:5]}"
        )
    raw_by_id = (
        {str(row["custom_id"]): row for row in read_jsonl(response_path)}
        if response_path.exists()
        else {}
    )

    data_dir = Path(args.data_dir)
    kg = load_temporal_kg(str(data_dir))
    name_to_id, _ = load_id_map(
        str(data_dir / "entity2id.txt"), expected_size=kg.num_entities
    )
    if dataset_files_fingerprint(str(data_dir)) != plan["dataset_fingerprint"]:
        raise ValueError("current dataset files differ from the reviewed request plan")
    mapper = EntityMapper(
        name_to_id,
        fuzzy_threshold=args.fuzzy_threshold,
        fuzzy_margin=args.fuzzy_margin,
    )
    records = []
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    resolved_models: set[str] = set()
    latency_values = []
    attempt_total = 0
    abstention_codes: Dict[str, int] = {}
    for index in sorted(index_rows, key=lambda value: int(value["ordinal"])):
        custom_id = str(index["custom_id"])
        if custom_id in successes:
            result = successes[custom_id]
            body = result["body"]
            candidates, diagnostics = parse_and_map_response(
                str(result["content"]),
                mapper,
                list(index.get("support_candidate_names", [])),
                max_candidates=int(plan["max_candidates"]),
            )
            usage = body.get("usage", {}) if isinstance(body, dict) else {}
            token_usage = {
                key: int(usage.get(key, 0) or 0) if isinstance(usage, dict) else 0
                for key in usage_totals
            }
            for key, value in token_usage.items():
                usage_totals[key] += value
            audit = raw_by_id[custom_id].get("realtime_audit", {})
            latency_ms = (
                float(audit.get("latency_ms", 0.0)) if isinstance(audit, dict) else 0.0
            )
            attempts = int(audit.get("attempts", 1)) if isinstance(audit, dict) else 1
            latency_values.append(latency_ms)
            response_id = body.get("id") or result.get("request_id")
            resolved_model = str(body.get("model") or plan["model"])
        else:
            abstention = abstentions[custom_id]
            candidates, diagnostics = parse_and_map_response(
                '{"candidates":[]}',
                mapper,
                list(index.get("support_candidate_names", [])),
                max_candidates=int(plan["max_candidates"]),
            )
            diagnostics = dict(diagnostics)
            diagnostics["provider_abstention"] = 1.0
            diagnostics["provider_data_inspection_failed"] = 1.0
            token_usage = {key: 0 for key in usage_totals}
            latency_ms = 0.0
            attempts = int(abstention["attempts"])
            response_id = None
            resolved_model = str(plan["model"])
            code = str(abstention["code"])
            abstention_codes[code] = abstention_codes.get(code, 0) + 1
        attempt_total += attempts
        resolved_models.add(resolved_model)
        records.append(
            {
                "schema_version": 1,
                "query_key": custom_id,
                "query": index["query"],
                "prompt_hash": index["prompt_hash"],
                "provider": PROVIDER_NAME,
                "model": resolved_model,
                "response_id": response_id,
                "candidates": candidates,
                "diagnostics": diagnostics,
                "latency_ms": latency_ms,
                "token_usage": token_usage,
            }
        )
    cache_bytes = jsonl_bytes(records)
    state_path = run_paths["state"]
    state = read_json(state_path) if state_path.exists() else {}
    provenance_path = PROJECT_ROOT / "ALIYUN_QWEN_REALTIME_PROVENANCE.json"
    provenance = read_json(provenance_path)
    exact_cost = (
        usage_totals["prompt_tokens"] * REALTIME_INPUT_RATE_CNY_PER_MILLION
        + usage_totals["completion_tokens"] * REALTIME_OUTPUT_RATE_CNY_PER_MILLION
    ) / 1_000_000.0
    metadata = {
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
            "region": "cn-beijing",
            "api_base_url": provenance["api_base_url"],
            "source_request_sha256": plan["request_sha256"],
            "source_index_sha256": plan["index_sha256"],
            "result_sha256": sha256_file(response_path) if response_path.exists() else None,
            "abstention_sha256": (
                sha256_file(abstention_path) if abstention_path.exists() else None
            ),
            "provider_abstention_policy": ABSTENTION_POLICY,
            "provenance_file": provenance_path.name,
            "provenance_file_sha256": sha256_file(provenance_path),
            "official_documentation": provenance["official_documentation"],
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
            "request_count": len(records),
            "successful_response_count": len(successes),
            "provider_abstention_count": len(abstentions),
            "provider_abstention_codes": abstention_codes,
            "provider_abstention_policy": ABSTENTION_POLICY,
            "token_usage": usage_totals,
            "estimated_list_price_cny": exact_cost,
            "price_verified_at": "2026-08-09",
            "attempt_count": attempt_total,
            "retry_count": attempt_total - len(records),
            "avg_latency_ms": sum(latency_values) / max(1, len(latency_values)),
            "per_request_latency_available": not abstentions,
            "per_successful_response_latency_available": bool(successes),
            "transport": state.get("transport", {}),
            "network_called_during_collection": False,
            "cache_sha256": sha256_bytes(cache_bytes),
        },
    }
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
        "estimated_list_price_cny": exact_cost,
        "provider_abstentions": len(abstentions),
        "formal_full_split": bool(plan["complete_split"]),
        "network_called": False,
    }
    print(canonical_json(result))
    return result


def status_job(args: argparse.Namespace) -> Dict[str, object]:
    state = read_json(_paths(Path(args.run_dir).resolve())["state"])
    print(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
    return state


def estimate_job(args: argparse.Namespace) -> Dict[str, object]:
    plan, _, _ = _verify_job(Path(args.source_job_dir).resolve())
    if args.avg_prompt_tokens < 0 or args.avg_completion_tokens < 0:
        raise ValueError("average token estimates must be non-negative")
    if not 0.0 <= args.retry_buffer <= 1.0:
        raise ValueError("--retry-buffer must be between 0 and 1")
    request_count = int(plan["request_count"])
    base_cost = request_count * (
        args.avg_prompt_tokens * REALTIME_INPUT_RATE_CNY_PER_MILLION
        + args.avg_completion_tokens * REALTIME_OUTPUT_RATE_CNY_PER_MILLION
    ) / 1_000_000.0
    result = {
        "status": "estimated_offline",
        "network_called": False,
        "model": plan["model"],
        "split": plan["split"],
        "shot": plan["shot"],
        "request_count": request_count,
        "avg_prompt_tokens": args.avg_prompt_tokens,
        "avg_completion_tokens": args.avg_completion_tokens,
        "input_rate_cny_per_million": REALTIME_INPUT_RATE_CNY_PER_MILLION,
        "output_rate_cny_per_million": REALTIME_OUTPUT_RATE_CNY_PER_MILLION,
        "base_estimated_cost_cny": base_cost,
        "retry_buffer": args.retry_buffer,
        "buffered_estimated_cost_cny": base_cost * (1.0 + args.retry_buffer),
        "token_basis": args.token_basis,
    }
    print(canonical_json(result))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Concurrent target-blind Alibaba Qwen realtime runner")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="network: generate or resume one realtime request plan")
    run.add_argument("--source-job-dir", required=True)
    run.add_argument("--run-dir", required=True)
    run.add_argument("--model", default=DEFAULT_REALTIME_MODEL)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--max-rpm", type=int, default=120)
    run.add_argument("--max-tpm", type=int, default=150_000)
    run.add_argument("--token-reservation", type=int, default=1_200)
    run.add_argument("--max-attempts", type=int, default=5)
    run.add_argument("--inspection-max-attempts", type=int, default=3)
    run.add_argument("--backoff-base", type=float, default=1.0)
    run.add_argument("--backoff-max", type=float, default=30.0)
    run.add_argument("--timeout", type=float, default=120.0)
    run.add_argument("--heartbeat-seconds", type=float, default=30.0)
    run.add_argument("--progress-every", type=int, default=5)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--execute-api", action="store_true")
    run.set_defaults(function=run_job)

    collect = commands.add_parser("collect", help="offline: build a validated realtime cache")
    collect.add_argument("--source-job-dir", required=True)
    collect.add_argument("--run-dir", required=True)
    collect.add_argument("--data-dir", default="data/ICEWS14")
    collect.add_argument("--output", required=True)
    collect.add_argument("--allow-incomplete-cache", action="store_true")
    collect.add_argument("--fuzzy-threshold", type=float, default=0.90)
    collect.add_argument("--fuzzy-margin", type=float, default=0.04)
    collect.set_defaults(function=collect_job)

    estimate = commands.add_parser("estimate", help="offline: estimate realtime list-price cost")
    estimate.add_argument("--source-job-dir", required=True)
    estimate.add_argument("--avg-prompt-tokens", type=float, default=734.0)
    estimate.add_argument("--avg-completion-tokens", type=float, default=648.42)
    estimate.add_argument("--retry-buffer", type=float, default=0.10)
    estimate.add_argument(
        "--token-basis",
        default="2026-08-09 qwen-flash 50-query realtime smoke",
    )
    estimate.set_defaults(function=estimate_job)

    status = commands.add_parser("status", help="offline: show the local realtime state")
    status.add_argument("--run-dir", required=True)
    status.set_defaults(function=status_job)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
