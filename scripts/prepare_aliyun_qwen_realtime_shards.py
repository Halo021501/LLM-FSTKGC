#!/usr/bin/env python3
"""Prepare deterministic, provider-legal STLP request shards offline.

ICEWS18 valid/test contain more public query locators than Alibaba's 50,000
requests-per-file limit.  This helper reuses the audited target-blind request
builder, then partitions the complete plan without making a network call.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import stlp_aliyun_qwen_batch as batch_cli
from src.aliyun_qwen_batch import (
    MAX_REQUESTS_PER_FILE,
    canonical_json,
    jsonl_bytes,
    sha256_bytes,
    validate_batch_requests,
    write_bytes_atomic,
    write_json_atomic,
)


def prepare(args: argparse.Namespace) -> dict[str, object]:
    args.data_dir = Path(args.data_dir).resolve()
    args.job_root = Path(args.job_root).resolve()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if args.job_root.exists() and any(args.job_root.iterdir()):
        raise FileExistsError(f"refusing non-empty shard root: {args.job_root}")

    # _prepare_rows is the same reviewed target-blind builder used by the
    # ICEWS14 Batch/realtime path.  It performs no network or credential read.
    request_rows, index_rows, base_plan = batch_cli._prepare_rows(args)
    if len(request_rows) != len(index_rows):
        raise RuntimeError("complete request/index count mismatch")
    minimum_shards = math.ceil(len(request_rows) / MAX_REQUESTS_PER_FILE)
    if args.num_shards < minimum_shards:
        raise ValueError(
            f"{len(request_rows)} requests require at least {minimum_shards} shards "
            f"under the {MAX_REQUESTS_PER_FILE}-request provider limit"
        )

    parent_request_bytes = jsonl_bytes(request_rows)
    parent_index_bytes = jsonl_bytes(index_rows)
    parent_request_sha256 = sha256_bytes(parent_request_bytes)
    parent_index_sha256 = sha256_bytes(parent_index_bytes)
    args.job_root.mkdir(parents=True, exist_ok=False, mode=0o700)

    parts: list[dict[str, object]] = []
    for shard_id in range(args.num_shards):
        selected = [
            (request, index)
            for ordinal, (request, index) in enumerate(zip(request_rows, index_rows))
            if ordinal % args.num_shards == shard_id
        ]
        if not selected:
            raise ValueError(f"empty shard {shard_id}")
        shard_requests = [item[0] for item in selected]
        shard_indexes = [item[1] for item in selected]
        request_bytes = jsonl_bytes(shard_requests)
        index_bytes = jsonl_bytes(shard_indexes)
        shard_dir = args.job_root / f"part_{shard_id:04d}-of-{args.num_shards:04d}"
        shard_dir.mkdir(mode=0o700)

        plan = dict(base_plan)
        plan.update(
            {
                "complete_split": False,
                "complete_partition": True,
                "request_count": len(shard_requests),
                "request_sha256": sha256_bytes(request_bytes),
                "request_bytes": len(request_bytes),
                "index_sha256": sha256_bytes(index_bytes),
                "partition": {
                    "num_shards": args.num_shards,
                    "shard_id": shard_id,
                    "assignment": "canonical_unique_locator_ordinal_mod",
                },
                "parent_complete_split": True,
                "parent_request_count": len(request_rows),
                "parent_request_sha256": parent_request_sha256,
                "parent_index_sha256": parent_index_sha256,
            }
        )
        request_path = shard_dir / batch_cli.REQUEST_FILENAME
        index_path = shard_dir / batch_cli.INDEX_FILENAME
        plan_path = shard_dir / batch_cli.PLAN_FILENAME
        write_bytes_atomic(request_path, request_bytes)
        write_bytes_atomic(index_path, index_bytes)
        write_json_atomic(plan_path, plan)
        validation = validate_batch_requests(request_path)
        if validation["sha256"] != plan["request_sha256"]:
            raise RuntimeError(f"post-write request hash mismatch: {shard_dir}")
        verified, _, _ = batch_cli._verify_job(shard_dir)
        if int(verified["request_count"]) != len(shard_requests):
            raise RuntimeError(f"post-write plan count mismatch: {shard_dir}")
        parts.append(
            {
                "shard_id": shard_id,
                "job_dir": str(shard_dir),
                "request_count": len(shard_requests),
                "request_sha256": plan["request_sha256"],
                "index_sha256": plan["index_sha256"],
            }
        )

    result = {
        "status": "prepared_sharded_offline",
        "network_called": False,
        "split": args.split,
        "shot": args.shot,
        "request_count": len(request_rows),
        "num_shards": args.num_shards,
        "parent_request_sha256": parent_request_sha256,
        "parent_index_sha256": parent_index_sha256,
        "parts": parts,
    }
    print(canonical_json(result))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline preparation of provider-legal sharded Qwen realtime plans"
    )
    parser.add_argument("--job-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", choices=["valid", "test"], required=True)
    parser.add_argument(
        "--shot", type=int, choices=sorted(batch_cli.SUPPORTED_SHOTS), required=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--history-protocol",
        choices=["standard_rolling_history", "strict_static_history"],
        default="standard_rolling_history",
    )
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--model", default=batch_cli.ROLLING_BATCH_MODEL)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--omit-support", action="store_true")
    parser.add_argument("--omit-history", action="store_true")
    parser.add_argument("--permute-support-order", action="store_true")
    parser.add_argument("--replace-entity-names", action="store_true")
    return parser


if __name__ == "__main__":
    prepare(build_parser().parse_args())
