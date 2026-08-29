#!/usr/bin/env python3
"""Attach workers for an already-running loopback Qwen server to a live pool.

This sidecar never initializes, recovers, merges, or finalizes the queue.  It
only uses the controller's existing SQLite transaction helpers to claim and
finish shards, so the authoritative controller remains the sole finalizer.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dynamic_local_qwen_pool import (  # noqa: E402
    STOP_EVENT,
    DynamicPool,
    Server,
    endpoint_ready,
    utc_now,
)


def live_controller(state_dir: Path) -> int:
    pid_path = state_dir / "controller.pid"
    text = pid_path.read_text(encoding="utf-8").strip() if pid_path.exists() else ""
    if not text.isdigit() or not Path(f"/proc/{text}").exists():
        raise RuntimeError("the authoritative dynamic Qwen controller is not running")
    command_path = Path(f"/proc/{text}/cmdline")
    command = command_path.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
    if "dynamic_local_qwen_pool.py" not in command:
        raise RuntimeError("controller.pid points to an unexpected live process")
    return int(text)


def queue_configuration(state_dir: Path) -> dict[str, object]:
    database = state_dir / "queue.sqlite3"
    if not database.is_file():
        raise FileNotFoundError(database)
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=30)
    try:
        row = connection.execute(
            "SELECT value FROM run_meta WHERE key='configuration_json'"
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise RuntimeError("queue has no formal configuration metadata")
    value = json.loads(str(row[0]))
    if not isinstance(value, dict):
        raise RuntimeError("queue configuration is not an object")
    required = {
        "cache_dir",
        "data_dir",
        "history_protocol",
        "max_retries",
        "max_tokens",
        "model",
        "num_shards",
        "retry_max_tokens",
        "seed",
        "shots",
        "split",
        "task_max_attempts",
        "task_retry_backoff_seconds",
        "workers_per_gpu",
    }
    if set(value) != required:
        raise RuntimeError("queue configuration fields do not match the supported schema")
    return value


def active_worker_conflict(state_dir: Path, gpu_id: int) -> list[tuple[object, ...]]:
    database = state_dir / "queue.sqlite3"
    connection = sqlite3.connect(str(database), timeout=30)
    try:
        return list(
            connection.execute(
                """
                SELECT worker_id,status,task_id FROM workers
                WHERE gpu_id=? AND status IN ('ready','running','retry_wait')
                ORDER BY worker_id
                """,
                (gpu_id,),
            )
        )
    finally:
        connection.close()


def mark_sidecar_workers_stopped(
    state_dir: Path,
    gpu_id: int,
    endpoint: str,
    worker_count: int,
) -> tuple[int, list[tuple[object, ...]]]:
    """Retire this sidecar's non-running worker rows before it exits.

    ``DynamicPool.worker_loop`` records ``ready`` after a generator subprocess
    returns.  When the sidecar then observes ``STOP_EVENT`` and exits, that
    final ``ready`` row used to look like a live worker forever and blocked a
    later, safe re-attach.  The sidecar has already joined all of its worker
    threads before this function is called, so only its exact GPU/slot/endpoint
    rows in non-running states are retired.  A ``running`` row is deliberately
    left untouched for manual recovery rather than guessing about task
    ownership.
    """

    database = state_dir / "queue.sqlite3"
    connection = sqlite3.connect(str(database), timeout=30)
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("BEGIN IMMEDIATE")
        worker_ids = [f"gpu{gpu_id}-slot{slot}" for slot in range(worker_count)]
        changed = 0
        for worker_id in worker_ids:
            cursor = connection.execute(
                """
                UPDATE workers
                SET status='stopped',task_id=NULL,updated_at_utc=?,error=NULL
                WHERE worker_id=? AND gpu_id=? AND endpoint=?
                  AND status IN ('ready','retry_wait','idle','failed','stopped')
                """,
                (utc_now(), worker_id, gpu_id, endpoint.rstrip("/")),
            )
            changed += int(cursor.rowcount)
        placeholders = ",".join("?" for _ in worker_ids)
        remaining = list(
            connection.execute(
                f"""
                SELECT worker_id,status,task_id FROM workers
                WHERE worker_id IN ({placeholders}) AND gpu_id=? AND endpoint=?
                  AND status IN ('ready','running','retry_wait')
                ORDER BY worker_id
                """,
                (*worker_ids, gpu_id, endpoint.rstrip("/")),
            )
        )
        connection.commit()
        return changed, [tuple(row) for row in remaining]
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def build_pool_args(args: argparse.Namespace, config: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        state_dir=args.state_dir,
        cache_dir=Path(str(config["cache_dir"])),
        data_dir=Path(str(config["data_dir"])),
        model_dir=args.model_dir,
        model=str(config["model"]),
        python_bin=args.python_bin,
        split=str(config["split"]),
        shots=[int(value) for value in config["shots"]],
        num_shards=int(config["num_shards"]),
        workers_per_gpu=int(config["workers_per_gpu"]),
        max_tokens=int(config["max_tokens"]),
        retry_max_tokens=int(config["retry_max_tokens"]),
        max_retries=int(config["max_retries"]),
        task_max_attempts=int(config["task_max_attempts"]),
        task_retry_backoff_seconds=float(config["task_retry_backoff_seconds"]),
        request_timeout=float(args.request_timeout),
        history_protocol=str(config["history_protocol"]),
        seed=int(config["seed"]),
        progress_every=int(args.progress_every),
        initial_gpus=[args.gpu_id],
        candidate_gpus=[],
        adopt_endpoint=[],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Attach one shared Qwen GPU to a live queue")
    parser.add_argument("--confirm-shared-worker", action="store_true")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument(
        "--model-dir", type=Path, default=PROJECT_ROOT / "models/Qwen2.5-7B-Instruct-AWQ"
    )
    parser.add_argument(
        "--python-bin", type=Path, default=Path(sys.executable)
    )
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--stop-server-on-exit", action="store_true")
    args = parser.parse_args()
    args.state_dir = args.state_dir.resolve()
    args.model_dir = args.model_dir.resolve()
    args.python_bin = args.python_bin.resolve()

    reboot_inhibit = args.state_dir / "PRE_REBOOT_CHECKPOINT.lock"
    if reboot_inhibit.exists():
        raise RuntimeError(
            f"pre-reboot checkpoint inhibits shared-Qwen attachment: {reboot_inhibit}"
        )

    if not args.confirm_shared_worker:
        raise ValueError("shared sidecar requires --confirm-shared-worker")
    if not args.endpoint.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise ValueError("only a loopback Qwen endpoint may be attached")
    if not args.model_dir.joinpath("config.json").is_file():
        raise FileNotFoundError(args.model_dir / "config.json")
    if not args.python_bin.is_file():
        raise FileNotFoundError(args.python_bin)

    controller_pid = live_controller(args.state_dir)
    config = queue_configuration(args.state_dir)
    if str(config["split"]) != "test" or sorted(config["shots"]) != [5, 10]:
        raise RuntimeError("sidecar is restricted to the formal test shot-5/10 queue")
    conflicts = active_worker_conflict(args.state_dir, args.gpu_id)
    if conflicts:
        raise RuntimeError(f"GPU {args.gpu_id} already has active queue workers: {conflicts}")
    if not endpoint_ready(args.endpoint, timeout=5.0):
        raise RuntimeError(f"Qwen endpoint is not ready: {args.endpoint}")

    pool = DynamicPool(build_pool_args(args, config))
    server = Server(args.gpu_id, args.endpoint.rstrip("/"), None, False)
    pool.log_event(
        "shared_sidecar_started",
        pid=os.getpid(),
        controller_pid=controller_pid,
        gpu_id=args.gpu_id,
        endpoint=server.endpoint,
        workers=int(config["workers_per_gpu"]),
    )
    pool.add_server(server)

    def stop(_signum: int, _frame: object) -> None:
        STOP_EVENT.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while any(thread.is_alive() for thread in pool.worker_threads):
            time.sleep(2)
    finally:
        try:
            stopped_rows, remaining_rows = mark_sidecar_workers_stopped(
                args.state_dir,
                args.gpu_id,
                server.endpoint,
                int(config["workers_per_gpu"]),
            )
            pool.log_event(
                "shared_sidecar_workers_stopped",
                pid=os.getpid(),
                gpu_id=args.gpu_id,
                stopped_rows=stopped_rows,
                remaining_active_rows=remaining_rows,
            )
        except Exception as exc:
            pool.log_event(
                "shared_sidecar_worker_cleanup_failed",
                pid=os.getpid(),
                gpu_id=args.gpu_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        pool.log_event(
            "shared_sidecar_stopped",
            pid=os.getpid(),
            gpu_id=args.gpu_id,
            stopped_by_signal=STOP_EVENT.is_set(),
        )
        if args.stop_server_on_exit:
            environment = os.environ.copy()
            environment.update(
                {
                    "LOCAL_QWEN_ENV_FILE": "/dev/null",
                    "LOCAL_QWEN_STATE_DIR": str(
                        args.state_dir / "servers" / f"gpu{args.gpu_id}"
                    ),
                }
            )
            result = subprocess.run(
                ["bash", "scripts/stop_local_qwen_server.sh"],
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=45,
                check=False,
            )
            pool.log_event(
                "shared_sidecar_server_stop",
                pid=os.getpid(),
                gpu_id=args.gpu_id,
                exit_code=result.returncode,
                output=result.stdout[-2000:],
            )
    return 130 if STOP_EVENT.is_set() else 0


if __name__ == "__main__":
    raise SystemExit(main())
