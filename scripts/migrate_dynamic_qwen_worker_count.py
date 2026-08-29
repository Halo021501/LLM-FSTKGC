#!/usr/bin/env python3
"""Auditably change only the operational worker count of a stopped Qwen queue."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reboot_gate import queue_mutation_gate  # noqa: E402


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def controller_is_live(state_dir: Path) -> bool:
    pid_file = state_dir / "controller.pid"
    text = pid_file.read_text(encoding="utf-8").strip() if pid_file.is_file() else ""
    if not text.isdigit() or not Path(f"/proc/{text}").is_dir():
        return False
    command_path = Path(f"/proc/{text}/cmdline")
    if not command_path.is_file():
        return False
    command = command_path.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    return "dynamic_local_qwen_pool.py" in command and str(state_dir) in command


def load_configuration(connection: sqlite3.Connection) -> tuple[dict[str, object], str]:
    rows = dict(connection.execute("SELECT key,value FROM run_meta"))
    if "configuration_json" not in rows or "configuration_sha256" not in rows:
        raise RuntimeError("queue is missing formal configuration metadata")
    configuration = json.loads(rows["configuration_json"])
    if not isinstance(configuration, dict):
        raise RuntimeError("configuration_json is not an object")
    observed_hash = str(rows["configuration_sha256"])
    calculated_hash = sha256_text(canonical_json(configuration))
    if observed_hash != calculated_hash:
        raise RuntimeError("stored configuration hash does not match configuration_json")
    return configuration, observed_hash


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preview or apply a workers_per_gpu-only Qwen queue migration"
    )
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--workers-per-gpu", type=int, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-operational-migration", action="store_true")
    args = parser.parse_args()

    state_dir = args.state_dir.resolve()
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be positive")
    if args.apply and not args.confirm_operational_migration:
        raise ValueError("--apply requires --confirm-operational-migration")
    if args.apply and controller_is_live(state_dir):
        raise RuntimeError("refusing to migrate a live authoritative controller")

    database = state_dir / "queue.sqlite3"
    if not database.is_file():
        raise FileNotFoundError(database)
    with queue_mutation_gate(
        state_dir,
        exclusive=False,
        reject_reboot_inhibit=args.apply,
    ):
        connection = sqlite3.connect(str(database), timeout=60)
        try:
            configuration, old_hash = load_configuration(connection)
            old_workers = int(configuration["workers_per_gpu"])
            target = dict(configuration)
            target["workers_per_gpu"] = int(args.workers_per_gpu)
            target_json = canonical_json(target)
            target_hash = sha256_text(target_json)
            backup_path: Path | None = None
            backup_sha256: str | None = None
            if args.apply and old_workers != args.workers_per_gpu:
                stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                backup_dir = state_dir / "backups"
                backup_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
                backup_path = backup_dir / (
                    f"queue_before_workers_{old_workers}_to_{args.workers_per_gpu}_{stamp}.sqlite3"
                )
                if backup_path.exists():
                    raise FileExistsError(backup_path)
                backup = sqlite3.connect(str(backup_path))
                try:
                    connection.backup(backup)
                finally:
                    backup.close()
                os.chmod(backup_path, 0o600)
                backup_sha256 = sha256_file(backup_path)
                check = sqlite3.connect(f"file:{backup_path}?mode=ro", uri=True)
                try:
                    result = check.execute("PRAGMA integrity_check").fetchone()
                finally:
                    check.close()
                if result is None or result[0] != "ok":
                    raise RuntimeError("SQLite backup integrity check failed")

                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "UPDATE run_meta SET value=? WHERE key='configuration_json'", (target_json,)
                )
                connection.execute(
                    "UPDATE run_meta SET value=? WHERE key='configuration_sha256'", (target_hash,)
                )
                connection.commit()
                verified, verified_hash = load_configuration(connection)
                if verified != target or verified_hash != target_hash:
                    raise RuntimeError("post-migration configuration verification failed")
        finally:
            connection.close()

        event = {
            "at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "event": "worker_count_migration_applied"
            if args.apply
            else "worker_count_migration_preview",
            "old_workers_per_gpu": old_workers,
            "new_workers_per_gpu": int(args.workers_per_gpu),
            "old_configuration_sha256": old_hash,
            "new_configuration_sha256": target_hash,
            "backup_path": str(backup_path) if backup_path else None,
            "backup_sha256": backup_sha256,
        }
        if args.apply and old_workers != args.workers_per_gpu:
            with (state_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(canonical_json(event) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
    print(canonical_json(event))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
