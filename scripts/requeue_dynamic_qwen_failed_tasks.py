#!/usr/bin/env python3
"""Safely preview or requeue exhausted dynamic local-Qwen shard tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dynamic_local_qwen_pool import (  # noqa: E402
    append_event,
    connect_database,
    requeue_failed_tasks,
)
from src.llm_cache import canonical_json  # noqa: E402
from src.reboot_gate import queue_mutation_gate  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preview failed tasks by default; --apply grants exactly one additional "
            "claim attempt while preserving all task_attempts audit rows"
        )
    )
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--task-id", type=int, action="append", default=[])
    parser.add_argument(
        "--all-failed",
        action="store_true",
        help="required with --apply when no explicit --task-id is supplied",
    )
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.task_id and args.all_failed:
        raise ValueError("choose explicit --task-id values or --all-failed, not both")
    if args.apply and not args.task_id and not args.all_failed:
        raise ValueError("--apply requires --task-id or explicit --all-failed")
    state_dir = args.state_dir.resolve()
    database = state_dir / "queue.sqlite3"
    if not database.is_file():
        raise FileNotFoundError(database)
    with queue_mutation_gate(
        state_dir,
        exclusive=False,
        reject_reboot_inhibit=args.apply,
    ):
        connection = connect_database(database)
        try:
            plan = requeue_failed_tasks(
                connection,
                args.task_id if args.task_id else None,
                apply=args.apply,
            )
        finally:
            connection.close()
        if args.apply and plan:
            append_event(
                state_dir / "events.jsonl",
                {
                    "event": "failed_tasks_manually_requeued",
                    "task_ids": [item["task_id"] for item in plan],
                    "previous_attempts": {
                        str(item["task_id"]): item["previous_attempts"] for item in plan
                    },
                    "next_attempts": {
                        str(item["task_id"]): item["next_attempt"] for item in plan
                    },
                },
            )
    print(
        canonical_json(
            {
                "status": "applied" if args.apply else "dry_run",
                "database": str(database),
                "task_count": len(plan),
                "tasks": plan,
                "attempts_preserved": True,
                "historical_attempt_rows_modified": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
