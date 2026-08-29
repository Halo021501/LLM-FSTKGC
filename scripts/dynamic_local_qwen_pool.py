#!/usr/bin/env python3
"""Persistent multi-GPU scheduler for formal local-Qwen cache generation.

The scheduler uses many deterministic query shards and four independent HTTP
workers per vLLM server.  It can attach new servers only after an additional GPU
has no compute process and enough free memory for consecutive monitor checks.
Failed shards are retried under a bounded, backoff-controlled policy; every
attempt remains recorded in SQLite and the append-only event log.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER_ENDPOINT_MAX_CONSECUTIVE_FAILURES = 3
WORKER_ENDPOINT_RETRY_BACKOFF_SECONDS = 2.0
REBOOT_INHIBIT_FILENAME = "PRE_REBOOT_CHECKPOINT.lock"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_temporal_kg
from src.llm_cache import LLMEvidenceCache, cache_file_sha256, canonical_json, target_blind_query_key


UTC = dt.timezone.utc
STOP_EVENT = threading.Event()


def utc_now() -> str:
    return dt.datetime.now(UTC).isoformat()


def parse_int_list(value: str) -> List[int]:
    result = []
    for item in value.split(","):
        item = item.strip()
        if item:
            result.append(int(item))
    if len(result) != len(set(result)):
        raise ValueError(f"duplicate integer in list: {value}")
    return result


def parse_endpoint_map(values: Sequence[str]) -> Dict[int, str]:
    endpoints: Dict[int, str] = {}
    for value in values:
        gpu_text, separator, endpoint = value.partition("=")
        if not separator:
            raise ValueError(f"endpoint must be GPU=URL: {value}")
        gpu_id = int(gpu_text)
        endpoint = endpoint.rstrip("/")
        if not endpoint.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise ValueError(f"only loopback endpoints are permitted: {endpoint}")
        endpoints[gpu_id] = endpoint
    return endpoints


def sha256_json(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def json_dump_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_event(path: Path, event: Mapping[str, object]) -> None:
    payload = dict(event)
    payload.setdefault("at_utc", utc_now())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def endpoint_ready(base_url: str, timeout: float = 3.0) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{base_url.rstrip('/')}/models", timeout=timeout) as response:
            return 200 <= int(response.status) < 300
    except Exception:
        return False


def connect_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path), timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def initialize_database(path: Path, args: argparse.Namespace) -> int:
    connection = connect_database(path)
    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS run_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                split TEXT NOT NULL,
                shot INTEGER NOT NULL,
                shard_id INTEGER NOT NULL,
                num_shards INTEGER NOT NULL,
                output_path TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                worker_id TEXT,
                started_at_utc TEXT,
                finished_at_utc TEXT,
                records INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                next_eligible_at_epoch REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS workers (
                worker_id TEXT PRIMARY KEY,
                gpu_id INTEGER NOT NULL,
                slot INTEGER NOT NULL,
                endpoint TEXT NOT NULL,
                managed_server INTEGER NOT NULL,
                status TEXT NOT NULL,
                task_id INTEGER,
                updated_at_utc TEXT NOT NULL,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS task_attempts (
                task_id INTEGER NOT NULL,
                attempt INTEGER NOT NULL,
                worker_id TEXT,
                gpu_id INTEGER,
                endpoint TEXT,
                started_at_utc TEXT NOT NULL,
                finished_at_utc TEXT,
                status TEXT NOT NULL,
                exit_code INTEGER,
                records_before INTEGER NOT NULL DEFAULT 0,
                records_after INTEGER,
                error TEXT,
                PRIMARY KEY(task_id, attempt)
            );
            """
        )
        # SQLite's CREATE TABLE IF NOT EXISTS does not add columns when a
        # controller is upgraded in place. Keep old state directories
        # resumable without discarding their audit trail.
        task_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(tasks)")
        }
        if "attempts" not in task_columns:
            connection.execute(
                "ALTER TABLE tasks ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0"
            )
        if "next_eligible_at_epoch" not in task_columns:
            connection.execute(
                "ALTER TABLE tasks ADD COLUMN next_eligible_at_epoch REAL NOT NULL DEFAULT 0"
            )
        configuration = {
            "data_dir": str(args.data_dir.resolve()),
            "cache_dir": str(args.cache_dir.resolve()),
            "split": args.split,
            "shots": args.shots,
            "num_shards": args.num_shards,
            "workers_per_gpu": args.workers_per_gpu,
            "max_tokens": args.max_tokens,
            "retry_max_tokens": args.retry_max_tokens,
            "max_retries": args.max_retries,
            "task_max_attempts": args.task_max_attempts,
            "task_retry_backoff_seconds": args.task_retry_backoff_seconds,
            "history_protocol": args.history_protocol,
            "seed": args.seed,
            "model": args.model,
        }
        config_hash = sha256_json(configuration)
        existing = connection.execute(
            "SELECT value FROM run_meta WHERE key='configuration_sha256'"
        ).fetchone()
        if existing is not None and existing["value"] != config_hash:
            raise ValueError("state directory belongs to a different generation configuration")
        connection.execute(
            "INSERT OR IGNORE INTO run_meta(key,value) VALUES('configuration_sha256',?)",
            (config_hash,),
        )
        connection.execute(
            "INSERT OR IGNORE INTO run_meta(key,value) VALUES('configuration_json',?)",
            (canonical_json(configuration),),
        )
        connection.execute(
            "INSERT OR IGNORE INTO run_meta(key,value) VALUES('started_at_utc',?)",
            (utc_now(),),
        )
        for shard_id in range(args.num_shards):
            for shot in args.shots:
                output = (
                    args.cache_dir
                    / "parts"
                    / f"{args.split}_s{shot}"
                    / f"part_{shard_id:04d}-of-{args.num_shards:04d}.jsonl"
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO tasks(
                        split,shot,shard_id,num_shards,output_path,status
                    ) VALUES(?,?,?,?,?,'pending')
                    """,
                    (args.split, shot, shard_id, args.num_shards, str(output)),
                )
        connection.commit()

        # If a controller was externally terminated, its subprocess attempt is
        # no longer trustworthy even when the part file contains some durable
        # records. Mark that attempt interrupted and let --resume continue
        # from those validated records under the same configuration.
        interrupted_rows = list(
            connection.execute("SELECT * FROM tasks WHERE status='running' ORDER BY id")
        )
        recovered = 0
        recovered_at = utc_now()
        for row in interrupted_rows:
            attempt = max(1, int(row["attempts"]))
            records = count_jsonl_records(Path(str(row["output_path"])))
            worker = str(row["worker_id"] or "unknown")
            connection.execute(
                """
                INSERT INTO task_attempts(
                    task_id,attempt,worker_id,started_at_utc,finished_at_utc,status,
                    records_before,records_after,error
                ) VALUES(?,?,?,?,?,'interrupted',?,?,?)
                ON CONFLICT(task_id,attempt) DO UPDATE SET
                    finished_at_utc=excluded.finished_at_utc,
                    status='interrupted',
                    records_after=excluded.records_after,
                    error=excluded.error
                """,
                (
                    int(row["id"]),
                    attempt,
                    worker,
                    str(row["started_at_utc"] or recovered_at),
                    recovered_at,
                    int(row["records"]),
                    records,
                    "controller restart recovered an interrupted shard attempt",
                ),
            )
            retry_allowed = attempt < args.task_max_attempts
            connection.execute(
                """
                UPDATE tasks SET status=?,worker_id=NULL,records=?,attempts=?,
                    started_at_utc=NULL,finished_at_utc=?,next_eligible_at_epoch=0,error=?
                WHERE id=?
                """,
                (
                    "pending" if retry_allowed else "failed",
                    records,
                    attempt,
                    None if retry_allowed else recovered_at,
                    "recovered after interrupted controller"
                    if retry_allowed
                    else "retry budget exhausted by interrupted controller attempts",
                    int(row["id"]),
                ),
            )
            recovered += int(retry_allowed)
        connection.commit()
        return recovered
    finally:
        connection.close()


def claim_task(
    connection: sqlite3.Connection,
    worker_id: str,
    gpu_id: int,
    endpoint: str,
) -> sqlite3.Row | None:
    connection.execute("BEGIN IMMEDIATE")
    row = connection.execute(
        """
        SELECT * FROM tasks
        WHERE status='pending' AND next_eligible_at_epoch<=?
        ORDER BY CASE WHEN attempts>0 THEN 0 ELSE 1 END,
                 next_eligible_at_epoch,shard_id,shot
        LIMIT 1
        """,
        (time.time(),),
    ).fetchone()
    if row is None:
        connection.commit()
        return None
    connection.execute(
        """
        UPDATE tasks SET status='running',worker_id=?,started_at_utc=?,
            finished_at_utc=NULL,error=NULL,attempts=attempts+1
        WHERE id=? AND status='pending'
        """,
        (worker_id, utc_now(), row["id"]),
    )
    claimed = connection.execute("SELECT * FROM tasks WHERE id=?", (row["id"],)).fetchone()
    connection.execute(
        """
        INSERT INTO task_attempts(
            task_id,attempt,worker_id,gpu_id,endpoint,started_at_utc,status,records_before
        ) VALUES(?,?,?,?,?,?,'running',?)
        """,
        (
            int(claimed["id"]),
            int(claimed["attempts"]),
            worker_id,
            gpu_id,
            endpoint,
            str(claimed["started_at_utc"]),
            int(claimed["records"]),
        ),
    )
    connection.commit()
    return claimed


def pending_wait_seconds(connection: sqlite3.Connection, cap: float = 5.0) -> float | None:
    row = connection.execute(
        "SELECT MIN(next_eligible_at_epoch) AS next_at FROM tasks WHERE status='pending'"
    ).fetchone()
    if row is None or row["next_at"] is None:
        return None
    return min(cap, max(0.1, float(row["next_at"]) - time.time()))


def set_worker(
    connection: sqlite3.Connection,
    worker_id: str,
    gpu_id: int,
    slot: int,
    endpoint: str,
    managed: bool,
    status: str,
    task_id: int | None = None,
    error: str | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO workers(
            worker_id,gpu_id,slot,endpoint,managed_server,status,task_id,updated_at_utc,error
        ) VALUES(?,?,?,?,?,?,?,?,?)
        ON CONFLICT(worker_id) DO UPDATE SET
            status=excluded.status,
            task_id=excluded.task_id,
            updated_at_utc=excluded.updated_at_utc,
            error=excluded.error
        """,
        (worker_id, gpu_id, slot, endpoint, int(managed), status, task_id, utc_now(), error),
    )
    connection.commit()


def finish_task(
    connection: sqlite3.Connection,
    task_id: int,
    succeeded: bool,
    records: int,
    exit_code: int,
    error: str | None,
    task_max_attempts: int,
    retry_backoff_seconds: float,
) -> Dict[str, object]:
    connection.execute("BEGIN IMMEDIATE")
    task = connection.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    if task is None:
        connection.rollback()
        raise KeyError(f"unknown task id: {task_id}")
    attempt = int(task["attempts"])
    finished_at = utc_now()
    if succeeded:
        final_status = "complete"
        delay = 0.0
        next_eligible = 0.0
    elif attempt < task_max_attempts:
        final_status = "pending"
        delay = retry_backoff_seconds * (2 ** max(0, attempt - 1))
        next_eligible = time.time() + delay
    else:
        final_status = "failed"
        delay = 0.0
        next_eligible = 0.0
    connection.execute(
        """
        UPDATE tasks SET status=?,worker_id=?,records=?,error=?,finished_at_utc=?,
            next_eligible_at_epoch=? WHERE id=?
        """,
        (
            final_status,
            None if final_status == "pending" else task["worker_id"],
            records,
            error,
            None if final_status == "pending" else finished_at,
            next_eligible,
            task_id,
        ),
    )
    attempt_status = (
        "complete" if succeeded else "retry_scheduled" if final_status == "pending" else "failed"
    )
    connection.execute(
        """
        UPDATE task_attempts SET finished_at_utc=?,status=?,exit_code=?,records_after=?,error=?
        WHERE task_id=? AND attempt=?
        """,
        (finished_at, attempt_status, exit_code, records, error, task_id, attempt),
    )
    connection.commit()
    return {
        "status": final_status,
        "attempt": attempt,
        "max_attempts": task_max_attempts,
        "retry_delay_seconds": delay,
    }


def requeue_failed_tasks(
    connection: sqlite3.Connection,
    task_ids: Sequence[int] | None = None,
    *,
    apply: bool = False,
) -> List[Dict[str, object]]:
    """Plan or apply a one-more-attempt requeue of terminal failed tasks.

    The previous ``attempts`` value is deliberately retained.  Consequently,
    ``claim_task`` allocates attempt N+1 and inserts a new ``task_attempts``
    primary key instead of overwriting or colliding with the terminal attempt.
    This helper never edits historical attempt rows or part files.
    """

    requested = None if task_ids is None else sorted({int(task_id) for task_id in task_ids})
    connection.execute("BEGIN IMMEDIATE")
    try:
        if requested:
            placeholders = ",".join("?" for _ in requested)
            rows = list(
                connection.execute(
                    f"SELECT * FROM tasks WHERE id IN ({placeholders}) ORDER BY id",
                    requested,
                )
            )
            found = {int(row["id"]) for row in rows}
            missing = sorted(set(requested).difference(found))
            if missing:
                raise KeyError(f"unknown task ids: {missing}")
            not_failed = [int(row["id"]) for row in rows if row["status"] != "failed"]
            if not_failed:
                raise ValueError(f"refusing to requeue non-failed task ids: {not_failed}")
        elif requested == []:
            rows = []
        else:
            rows = list(connection.execute("SELECT * FROM tasks WHERE status='failed' ORDER BY id"))

        plan: List[Dict[str, object]] = []
        for row in rows:
            task_id = int(row["id"])
            attempt = int(row["attempts"])
            if attempt < 1:
                raise ValueError(f"failed task {task_id} has no recorded attempt")
            terminal = connection.execute(
                "SELECT status FROM task_attempts WHERE task_id=? AND attempt=?",
                (task_id, attempt),
            ).fetchone()
            if terminal is None or terminal["status"] != "failed":
                observed = None if terminal is None else terminal["status"]
                raise ValueError(
                    f"failed task {task_id} lacks matching terminal attempt {attempt}: {observed!r}"
                )
            plan.append(
                {
                    "task_id": task_id,
                    "shot": int(row["shot"]),
                    "shard_id": int(row["shard_id"]),
                    "records": int(row["records"]),
                    "previous_attempts": attempt,
                    "next_attempt": attempt + 1,
                    "output_path": str(row["output_path"]),
                }
            )

        if apply:
            for item in plan:
                cursor = connection.execute(
                    """
                    UPDATE tasks SET status='pending',worker_id=NULL,started_at_utc=NULL,
                        finished_at_utc=NULL,next_eligible_at_epoch=0
                    WHERE id=? AND status='failed' AND attempts=?
                    """,
                    (item["task_id"], item["previous_attempts"]),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        f"failed task changed during requeue: {item['task_id']}"
                    )
            connection.commit()
        else:
            connection.rollback()
        return plan
    except Exception:
        connection.rollback()
        raise


@dataclass
class Server:
    gpu_id: int
    endpoint: str
    state_dir: Path | None
    managed: bool


class DynamicPool:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.database_path = args.state_dir / "queue.sqlite3"
        self.events_path = args.state_dir / "events.jsonl"
        self.status_path = args.state_dir / "status.json"
        self.controller_pid_path = args.state_dir / "controller.pid"
        self.servers: Dict[int, Server] = {}
        self.worker_threads: List[threading.Thread] = []
        self.idle_confirmations: Dict[int, int] = {}
        self.startup_retry_after: Dict[int, float] = {}
        self.monitored_gpus = sorted(set(args.initial_gpus).union(args.candidate_gpus))
        self.event_lock = threading.Lock()
        self.started_monotonic = time.monotonic()
        self.adopt_endpoints = parse_endpoint_map(args.adopt_endpoint)

    def log_event(self, kind: str, **fields: object) -> None:
        with self.event_lock:
            append_event(self.events_path, {"event": kind, **fields})

    def register_pid(self) -> None:
        self.args.state_dir.mkdir(parents=True, exist_ok=True)
        if self.controller_pid_path.exists():
            text = self.controller_pid_path.read_text(encoding="utf-8").strip()
            if text.isdigit() and Path(f"/proc/{text}").exists():
                raise RuntimeError(f"controller PID already appears live: {text}")
        self.controller_pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")

    def query_gpu_state(self) -> Dict[int, Dict[str, object]]:
        states: Dict[int, Dict[str, object]] = {}
        for requested_gpu_id in self.monitored_gpus:
            gpu_command = [
                "nvidia-smi",
                "-i",
                str(requested_gpu_id),
                "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
            process_command = [
                "nvidia-smi",
                "-i",
                str(requested_gpu_id),
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
            try:
                gpu_output = subprocess.check_output(gpu_command, text=True, timeout=15)
                parsed_state: Dict[str, object] | None = None
                for line in gpu_output.splitlines():
                    fields = [item.strip() for item in line.split(",")]
                    if len(fields) != 5:
                        continue
                    index = int(fields[0])
                    if index != requested_gpu_id:
                        continue
                    parsed_state = {
                        "uuid": fields[1],
                        "memory_used_mib": int(fields[2]),
                        "memory_free_mib": int(fields[3]),
                        "utilization_gpu": int(fields[4]),
                        "compute_pids": [],
                    }
                    break
                if parsed_state is None:
                    raise ValueError(
                        f"nvidia-smi returned no state for GPU {requested_gpu_id}"
                    )

                # Query compute processes on the same physical card.  If this
                # query is unavailable, omit the card instead of treating it as
                # idle and risking a server launch over an unknown workload.
                process_output = subprocess.check_output(
                    process_command, text=True, timeout=15
                )
                for line in process_output.splitlines():
                    fields = [item.strip() for item in line.split(",")]
                    if not fields:
                        continue
                    try:
                        pid = int(fields[0])
                    except ValueError:
                        continue
                    parsed_state["compute_pids"].append(pid)
                states[requested_gpu_id] = parsed_state
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                self.log_event(
                    "gpu_query_failed",
                    gpu_id=requested_gpu_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
        return states

    def server_environment(self, gpu_id: int, port: int, state_dir: Path, shared: bool) -> Dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "LOCAL_QWEN_ENV_FILE": "/dev/null",
                "LOCAL_QWEN_GPU_ID": str(gpu_id),
                "LOCAL_QWEN_PORT": str(port),
                "LOCAL_QWEN_STATE_DIR": str(state_dir),
                "LOCAL_QWEN_MODEL_DIR": str(self.args.model_dir),
                "LOCAL_QWEN_MODEL": self.args.model,
                "LOCAL_QWEN_MAX_MODEL_LEN": str(self.args.max_model_len),
                "LOCAL_QWEN_GPU_MEMORY_UTILIZATION": str(self.args.gpu_memory_utilization),
                "LOCAL_QWEN_MAX_NUM_SEQS": str(self.args.workers_per_gpu),
                "LOCAL_QWEN_MIN_FREE_MIB": str(
                    self.args.shared_min_free_mib if shared else self.args.additional_min_free_mib
                ),
                "LOCAL_QWEN_ENFORCE_EAGER": "YES",
                "LOCAL_QWEN_DISABLE_FRONTEND_MULTIPROCESSING": "YES",
                "LOCAL_QWEN_GUIDED_DECODING_BACKEND": "lm-format-enforcer",
                "LOCAL_QWEN_QUANTIZATION": "awq_marlin",
                "ALLOW_SHARED_GPU": "YES" if shared else "NO",
            }
        )
        return environment

    def start_managed_server(self, gpu_id: int, shared: bool) -> Server | None:
        if time.time() < self.startup_retry_after.get(gpu_id, 0.0):
            return None
        port = self.args.port_base + gpu_id
        state_dir = self.args.state_dir / "servers" / f"gpu{gpu_id}"
        endpoint = f"http://127.0.0.1:{port}/v1"
        environment = self.server_environment(gpu_id, port, state_dir, shared)
        log_path = self.args.state_dir / "servers" / f"gpu{gpu_id}_startup.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_event("server_start_requested", gpu_id=gpu_id, endpoint=endpoint, shared=shared)
        with log_path.open("a", encoding="utf-8") as log_handle:
            result = subprocess.run(
                ["bash", "scripts/start_local_qwen_server.sh"],
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                timeout=self.args.server_start_timeout,
                check=False,
            )
        if result.returncode != 0 or not endpoint_ready(endpoint):
            self.startup_retry_after[gpu_id] = (
                time.time() + self.args.server_retry_cooldown_seconds
            )
            self.log_event(
                "server_start_failed",
                gpu_id=gpu_id,
                endpoint=endpoint,
                exit_code=result.returncode,
                log=str(log_path),
            )
            return None
        self.startup_retry_after.pop(gpu_id, None)
        server = Server(gpu_id=gpu_id, endpoint=endpoint, state_dir=state_dir, managed=True)
        self.log_event("server_ready", gpu_id=gpu_id, endpoint=endpoint, managed=True)
        return server

    def start_managed_servers_parallel(
        self,
        requests: Sequence[Tuple[int, bool]],
    ) -> List[Server]:
        """Start independent per-GPU servers concurrently.

        Model loading takes roughly 30-40 seconds per card. Serial startup made
        the final card wait once for every earlier card, even though the GPUs
        and ports are independent.
        """

        if not requests:
            return []
        servers: List[Server] = []
        with ThreadPoolExecutor(
            max_workers=len(requests),
            thread_name_prefix="qwen-server-start",
        ) as executor:
            futures = {
                executor.submit(self.start_managed_server, gpu_id, shared): gpu_id
                for gpu_id, shared in requests
            }
            for future in as_completed(futures):
                gpu_id = futures[future]
                try:
                    server = future.result()
                except Exception as exc:
                    self.startup_retry_after[gpu_id] = (
                        time.time() + self.args.server_retry_cooldown_seconds
                    )
                    self.log_event(
                        "server_start_exception",
                        gpu_id=gpu_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    continue
                if server is not None:
                    servers.append(server)
        return sorted(servers, key=lambda item: item.gpu_id)

    def add_server(self, server: Server) -> None:
        if server.gpu_id in self.servers:
            return
        if not endpoint_ready(server.endpoint):
            self.log_event(
                "server_rejected_unhealthy",
                gpu_id=server.gpu_id,
                endpoint=server.endpoint,
            )
            return
        self.servers[server.gpu_id] = server
        for slot in range(self.args.workers_per_gpu):
            thread = threading.Thread(
                target=self.worker_loop,
                args=(server, slot),
                name=f"qwen-gpu{server.gpu_id}-slot{slot}",
                daemon=True,
            )
            self.worker_threads.append(thread)
            thread.start()
        self.log_event(
            "workers_started",
            gpu_id=server.gpu_id,
            endpoint=server.endpoint,
            count=self.args.workers_per_gpu,
            managed=server.managed,
        )

    def generator_command(self, task: sqlite3.Row) -> List[str]:
        return [
            str(self.args.python_bin),
            "scripts/stlp_generate_candidates.py",
            "--data-dir",
            str(self.args.data_dir),
            "--split",
            str(task["split"]),
            "--shot",
            str(task["shot"]),
            "--seed",
            str(self.args.seed),
            "--history-protocol",
            self.args.history_protocol,
            "--provider",
            "local_qwen",
            "--max-tokens",
            str(self.args.max_tokens),
            "--retry-max-tokens",
            str(self.args.retry_max_tokens),
            "--timeout",
            str(self.args.request_timeout),
            "--max-retries",
            str(self.args.max_retries),
            "--request-interval",
            "0",
            "--resume",
            "--num-shards",
            str(task["num_shards"]),
            "--shard-id",
            str(task["shard_id"]),
            "--progress-every",
            str(self.args.progress_every),
            "--output",
            str(task["output_path"]),
        ]

    def worker_loop(self, server: Server, slot: int) -> None:
        connection = connect_database(self.database_path)
        worker_id = f"gpu{server.gpu_id}-slot{slot}"
        log_path = self.args.state_dir / "workers" / f"{worker_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            set_worker(
                connection,
                worker_id,
                server.gpu_id,
                slot,
                server.endpoint,
                server.managed,
                "ready",
            )
            consecutive_endpoint_failures = 0
            while not STOP_EVENT.is_set():
                if not endpoint_ready(server.endpoint):
                    consecutive_endpoint_failures += 1
                    if (
                        consecutive_endpoint_failures
                        < WORKER_ENDPOINT_MAX_CONSECUTIVE_FAILURES
                    ):
                        set_worker(
                            connection,
                            worker_id,
                            server.gpu_id,
                            slot,
                            server.endpoint,
                            server.managed,
                            "retry_wait",
                            error=(
                                "transient local Qwen endpoint health-check failure "
                                f"{consecutive_endpoint_failures}/"
                                f"{WORKER_ENDPOINT_MAX_CONSECUTIVE_FAILURES}"
                            ),
                        )
                        self.log_event(
                            "worker_endpoint_check_retry",
                            worker_id=worker_id,
                            gpu_id=server.gpu_id,
                            consecutive_failures=consecutive_endpoint_failures,
                            required_failures=WORKER_ENDPOINT_MAX_CONSECUTIVE_FAILURES,
                            retry_delay_seconds=WORKER_ENDPOINT_RETRY_BACKOFF_SECONDS,
                        )
                        STOP_EVENT.wait(WORKER_ENDPOINT_RETRY_BACKOFF_SECONDS)
                        continue
                    message = (
                        "local Qwen endpoint became unhealthy after "
                        f"{consecutive_endpoint_failures} consecutive checks; "
                        "no task was claimed or retried"
                    )
                    set_worker(
                        connection,
                        worker_id,
                        server.gpu_id,
                        slot,
                        server.endpoint,
                        server.managed,
                        "failed",
                        error=message,
                    )
                    self.log_event(
                        "worker_endpoint_unhealthy",
                        worker_id=worker_id,
                        gpu_id=server.gpu_id,
                        consecutive_failures=consecutive_endpoint_failures,
                    )
                    return
                consecutive_endpoint_failures = 0
                task = claim_task(connection, worker_id, server.gpu_id, server.endpoint)
                if task is None:
                    wait_seconds = pending_wait_seconds(connection)
                    if wait_seconds is not None:
                        set_worker(
                            connection,
                            worker_id,
                            server.gpu_id,
                            slot,
                            server.endpoint,
                            server.managed,
                            "retry_wait",
                        )
                        STOP_EVENT.wait(wait_seconds)
                        continue
                    set_worker(
                        connection,
                        worker_id,
                        server.gpu_id,
                        slot,
                        server.endpoint,
                        server.managed,
                        "idle",
                    )
                    return
                set_worker(
                    connection,
                    worker_id,
                    server.gpu_id,
                    slot,
                    server.endpoint,
                    server.managed,
                    "running",
                    task_id=int(task["id"]),
                )
                output_path = Path(str(task["output_path"]))
                output_path.parent.mkdir(parents=True, exist_ok=True)
                command = self.generator_command(task)
                environment = os.environ.copy()
                environment.update(
                    {
                        "LOCAL_QWEN_BASE_URL": server.endpoint,
                        "LOCAL_QWEN_MODEL": self.args.model,
                        "LOCAL_QWEN_MODEL_DIR": str(self.args.model_dir),
                        "LOCAL_QWEN_GPU_ID": str(server.gpu_id),
                        "LOCAL_QWEN_QUANTIZATION": "awq_marlin",
                    }
                )
                self.log_event(
                    "task_started",
                    task_id=int(task["id"]),
                    worker_id=worker_id,
                    gpu_id=server.gpu_id,
                    shot=int(task["shot"]),
                    shard_id=int(task["shard_id"]),
                    attempt=int(task["attempts"]),
                    max_attempts=self.args.task_max_attempts,
                    command=command,
                )
                with log_path.open("a", encoding="utf-8") as log_handle:
                    log_handle.write(
                        f"\n[{utc_now()}] task={task['id']} attempt={task['attempts']}"
                        f"/{self.args.task_max_attempts} command={json.dumps(command)}\n"
                    )
                    log_handle.flush()
                    result = subprocess.run(
                        command,
                        cwd=PROJECT_ROOT,
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                records = count_jsonl_records(output_path)
                if result.returncode == 0:
                    outcome = finish_task(
                        connection,
                        int(task["id"]),
                        True,
                        records,
                        result.returncode,
                        None,
                        self.args.task_max_attempts,
                        self.args.task_retry_backoff_seconds,
                    )
                    self.log_event(
                        "task_complete",
                        task_id=int(task["id"]),
                        worker_id=worker_id,
                        records=records,
                        attempt=outcome["attempt"],
                    )
                else:
                    message = f"generator exited with code {result.returncode}"
                    outcome = finish_task(
                        connection,
                        int(task["id"]),
                        False,
                        records,
                        result.returncode,
                        message,
                        self.args.task_max_attempts,
                        self.args.task_retry_backoff_seconds,
                    )
                    if outcome["status"] == "pending":
                        self.log_event(
                            "task_retry_scheduled",
                            task_id=int(task["id"]),
                            worker_id=worker_id,
                            records=records,
                            error=message,
                            attempt=outcome["attempt"],
                            max_attempts=outcome["max_attempts"],
                            retry_delay_seconds=outcome["retry_delay_seconds"],
                            log=str(log_path),
                        )
                        # Keep this worker available for the retry instead of
                        # immediately consuming another fresh shard. After the
                        # bounded backoff, claim_task prioritizes tasks with
                        # prior attempts, while another free worker may still
                        # take the retry sooner.
                        STOP_EVENT.wait(float(outcome["retry_delay_seconds"]))
                    else:
                        self.log_event(
                            "task_failed",
                            task_id=int(task["id"]),
                            worker_id=worker_id,
                            records=records,
                            error=message,
                            attempt=outcome["attempt"],
                            max_attempts=outcome["max_attempts"],
                            retry_exhausted=True,
                            log=str(log_path),
                        )
                set_worker(
                    connection,
                    worker_id,
                    server.gpu_id,
                    slot,
                    server.endpoint,
                    server.managed,
                    "ready",
                )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            try:
                set_worker(
                    connection,
                    worker_id,
                    server.gpu_id,
                    slot,
                    server.endpoint,
                    server.managed,
                    "failed",
                    error=message,
                )
            finally:
                self.log_event("worker_failed", worker_id=worker_id, error=message)
        finally:
            connection.close()

    def task_summary(self) -> Dict[str, object]:
        connection = connect_database(self.database_path)
        try:
            counts = {
                row["status"]: int(row["count"])
                for row in connection.execute(
                    "SELECT status,COUNT(*) AS count FROM tasks GROUP BY status"
                )
            }
            rows = list(
                connection.execute(
                    "SELECT output_path,status,records,attempts,next_eligible_at_epoch FROM tasks"
                )
            )
            workers = [dict(row) for row in connection.execute("SELECT * FROM workers ORDER BY gpu_id,slot")]
            failures = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT id,shot,shard_id,worker_id,error,attempts
                    FROM tasks WHERE status='failed' ORDER BY id
                    """
                )
            ]
            attempt_summary = dict(
                connection.execute(
                    """
                    SELECT COUNT(*) AS total_attempts,
                           COALESCE(SUM(CASE WHEN attempt>1 THEN 1 ELSE 0 END),0)
                               AS retry_attempts,
                           COUNT(DISTINCT CASE WHEN attempt>1 THEN task_id END)
                               AS retried_tasks
                    FROM task_attempts
                    """
                ).fetchone()
            )
        finally:
            connection.close()
        records_written = 0
        for row in rows:
            if row["status"] != "held":
                records_written += count_jsonl_records(Path(str(row["output_path"])))
        elapsed = max(0.001, time.monotonic() - self.started_monotonic)
        rate = records_written / elapsed
        remaining = max(0, self.args.expected_records - records_written)
        eta_seconds = remaining / rate if rate > 0 else None
        return {
            "task_counts": counts,
            "records_written": records_written,
            "expected_records": self.args.expected_records,
            "progress": records_written / max(1, self.args.expected_records),
            "elapsed_seconds": elapsed,
            "records_per_second": rate,
            "eta_seconds": eta_seconds,
            "workers": workers,
            "failed_tasks": failures,
            "retry_waiting_tasks": sum(
                1
                for row in rows
                if row["status"] == "pending"
                and float(row["next_eligible_at_epoch"]) > time.time()
            ),
            "attempt_summary": attempt_summary,
        }

    def write_status(self, phase: str, gpu_state: Mapping[int, Mapping[str, object]] | None = None) -> Dict[str, object]:
        summary = self.task_summary()
        status = {
            "schema_version": 2,
            "phase": phase,
            "updated_at_utc": utc_now(),
            "controller_pid": os.getpid(),
            "state_dir": str(self.args.state_dir),
            "cache_dir": str(self.args.cache_dir),
            "active_servers": [
                {
                    "gpu_id": server.gpu_id,
                    "endpoint": server.endpoint,
                    "managed": server.managed,
                }
                for server in sorted(self.servers.values(), key=lambda item: item.gpu_id)
            ],
            "gpu_state": gpu_state or {},
            "candidate_gpu_monitor": {
                "candidate_gpus": self.args.candidate_gpus,
                "monitored_gpus": self.monitored_gpus,
                "idle_confirmations": {
                    str(gpu_id): int(self.idle_confirmations.get(gpu_id, 0))
                    for gpu_id in self.monitored_gpus
                },
                "startup_retry_after_epoch": {
                    str(gpu_id): float(retry_at)
                    for gpu_id, retry_at in sorted(self.startup_retry_after.items())
                    if retry_at > time.time()
                },
                "required_consecutive_checks": self.args.idle_checks,
                "minimum_free_memory_mib": self.args.additional_min_free_mib,
                "maximum_utilization_percent": self.args.additional_max_utilization,
                "poll_seconds": self.args.poll_seconds,
            },
            **summary,
        }
        json_dump_atomic(self.status_path, status)
        return status

    def add_initial_servers(self, gpu_state: Mapping[int, Mapping[str, object]]) -> None:
        managed_requests: List[Tuple[int, bool]] = []
        for gpu_id in self.args.initial_gpus:
            if gpu_id in self.adopt_endpoints:
                server = Server(gpu_id, self.adopt_endpoints[gpu_id], None, False)
                if endpoint_ready(server.endpoint):
                    self.log_event(
                        "server_adopted",
                        gpu_id=gpu_id,
                        endpoint=server.endpoint,
                    )
                    self.add_server(server)
                else:
                    self.log_event(
                        "server_adoption_failed",
                        gpu_id=gpu_id,
                        endpoint=server.endpoint,
                    )
                continue
            state = gpu_state.get(gpu_id)
            if state is None or int(state["memory_free_mib"]) < self.args.shared_min_free_mib:
                self.log_event(
                    "initial_gpu_rejected",
                    gpu_id=gpu_id,
                    state=state,
                    required_free_mib=self.args.shared_min_free_mib,
                )
                continue
            managed_requests.append((gpu_id, True))
        for server in self.start_managed_servers_parallel(managed_requests):
            self.add_server(server)

    def maybe_add_idle_servers(self, gpu_state: Mapping[int, Mapping[str, object]]) -> None:
        managed_requests: List[Tuple[int, bool]] = []
        for gpu_id in self.monitored_gpus:
            if gpu_id in self.servers:
                continue
            if time.time() < self.startup_retry_after.get(gpu_id, 0.0):
                continue
            state = gpu_state.get(gpu_id)
            idle = bool(
                state is not None
                and not state["compute_pids"]
                and int(state["memory_free_mib"]) >= self.args.additional_min_free_mib
                and int(state["utilization_gpu"]) <= self.args.additional_max_utilization
            )
            self.idle_confirmations[gpu_id] = self.idle_confirmations.get(gpu_id, 0) + 1 if idle else 0
            if self.idle_confirmations[gpu_id] < self.args.idle_checks:
                continue
            self.log_event(
                "idle_gpu_confirmed",
                gpu_id=gpu_id,
                state=state,
                confirmations=self.idle_confirmations[gpu_id],
            )
            managed_requests.append((gpu_id, False))
        for server in self.start_managed_servers_parallel(managed_requests):
            self.add_server(server)

    def expected_locator_count(self) -> int:
        kg = load_temporal_kg(str(self.args.data_dir))
        rows = {"valid": kg.valid, "test": kg.test}[self.args.split]
        locators = {(s, r, t) for s, r, _o, t in rows}
        locators.update((o, r + kg.num_relations, t) for s, r, o, t in rows)
        return len(locators)

    def merge_shot(self, shot: int) -> None:
        connection = connect_database(self.database_path)
        try:
            tasks = list(
                connection.execute(
                    "SELECT * FROM tasks WHERE shot=? ORDER BY shard_id", (shot,)
                )
            )
        finally:
            connection.close()
        if len(tasks) != self.args.num_shards or any(task["status"] != "complete" for task in tasks):
            raise RuntimeError(f"shot {shot} does not have a complete shard set")

        final_path = self.args.cache_dir / f"{self.args.split}_s{shot}.jsonl"
        final_meta_path = Path(str(final_path) + ".meta.json")
        if final_path.exists() or final_meta_path.exists():
            raise FileExistsError(f"refusing to overwrite formal cache: {final_path}")
        records: Dict[Tuple[int, int, int], Mapping[str, object]] = {}
        common_metadata: Mapping[str, object] | None = None
        part_manifest: List[Dict[str, object]] = []
        for task in tasks:
            part_path = Path(str(task["output_path"]))
            meta_path = Path(str(part_path) + ".meta.json")
            if not part_path.is_file() or not meta_path.is_file():
                raise FileNotFoundError(f"missing completed shard artifact: {part_path}")
            with meta_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            partition = metadata.get("partition")
            if partition != {
                "num_shards": self.args.num_shards,
                "shard_id": int(task["shard_id"]),
                "assignment": "canonical_unique_locator_ordinal_mod",
            }:
                raise ValueError(f"partition metadata mismatch: {meta_path}")
            invariant = {
                key: value
                for key, value in metadata.items()
                if key not in {"generation_audit", "partition"}
            }
            if common_metadata is None:
                common_metadata = invariant
            elif invariant != common_metadata:
                raise ValueError(f"scientific metadata differs across shards: {meta_path}")
            with part_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    query = record.get("query", {})
                    if record.get("query_key") != target_blind_query_key(query):
                        raise ValueError(f"invalid query key at {part_path}:{line_number}")
                    if query.get("split") != self.args.split or int(query.get("shot", -1)) != shot:
                        raise ValueError(f"query protocol mismatch at {part_path}:{line_number}")
                    locator = (
                        int(query["known_entity_id"]),
                        int(query["oriented_relation_id"]),
                        int(query["timestamp"]),
                    )
                    if locator in records:
                        raise ValueError(f"duplicate locator across shards: {locator}")
                    records[locator] = record
            part_manifest.append(
                {
                    "shard_id": int(task["shard_id"]),
                    "records": int(task["records"]),
                    "jsonl": os.path.relpath(part_path, PROJECT_ROOT),
                    "jsonl_sha256": cache_file_sha256(str(part_path)),
                    "metadata": os.path.relpath(meta_path, PROJECT_ROOT),
                    "metadata_sha256": cache_file_sha256(str(meta_path)),
                }
            )

        expected = self.expected_locator_count()
        if len(records) != expected:
            raise ValueError(f"shot {shot} cache has {len(records)} locators; expected {expected}")
        assert common_metadata is not None
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = final_path.with_name(f".{final_path.name}.tmp.{os.getpid()}")
        with temporary.open("x", encoding="utf-8") as handle:
            for locator in sorted(records, key=lambda item: (item[2], item[0], item[1])):
                handle.write(canonical_json(records[locator]) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, final_path)

        manifest_path = final_path.with_suffix(final_path.suffix + ".parts.json")
        json_dump_atomic(
            manifest_path,
            {
                "schema_version": 1,
                "final_cache": os.path.relpath(final_path, PROJECT_ROOT),
                "records": len(records),
                "parts": part_manifest,
            },
        )
        connection = connect_database(self.database_path)
        try:
            started_at = connection.execute(
                "SELECT value FROM run_meta WHERE key='started_at_utc'"
            ).fetchone()["value"]
        finally:
            connection.close()
        final_metadata = dict(common_metadata)
        final_metadata["generation_audit"] = {
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "mode": "dynamic_multi_gpu_sharded_generation",
            "controller_command_argv": [sys.executable, *sys.argv],
            "hostname": os.uname().nodename,
            "num_shards": self.args.num_shards,
            "workers_per_gpu": self.args.workers_per_gpu,
            "task_retry_policy": {
                "task_max_attempts": self.args.task_max_attempts,
                "task_retry_backoff_seconds": self.args.task_retry_backoff_seconds,
                "per_query_max_retries": self.args.max_retries,
                "initial_max_tokens": self.args.max_tokens,
                "retry_max_tokens": self.args.retry_max_tokens,
            },
            "attempt_summary": self.task_summary()["attempt_summary"],
            "part_manifest": os.path.relpath(manifest_path, PROJECT_ROOT),
            "part_manifest_sha256": cache_file_sha256(str(manifest_path)),
            "gpu_endpoints": [
                {"gpu_id": item.gpu_id, "endpoint": item.endpoint, "managed": item.managed}
                for item in sorted(self.servers.values(), key=lambda server: server.gpu_id)
            ],
        }
        json_dump_atomic(final_meta_path, final_metadata)
        cache = LLMEvidenceCache(
            str(final_path),
            max_candidates=10,
            expected_shot=shot,
            expected_history_protocol=self.args.history_protocol,
            expected_split=self.args.split,
            require_generation_metadata=True,
        )
        if len(cache.records) != expected:
            raise RuntimeError(f"post-merge validation count mismatch for shot {shot}")
        self.log_event(
            "cache_finalized",
            shot=shot,
            path=str(final_path),
            records=len(cache.records),
            sha256=cache.sha256,
            metadata_sha256=cache.generation_metadata_sha256,
        )

    def run_diagnostics(self) -> None:
        for shot in self.args.shots:
            output = self.args.state_dir / f"llm_only_test_s{shot}.json"
            command = [
                str(self.args.python_bin),
                "scripts/stlp_evaluate_llm_only.py",
                "--data-dir",
                str(self.args.data_dir),
                "--cache",
                str(self.args.cache_dir / f"{self.args.split}_s{shot}.jsonl"),
                "--split",
                self.args.split,
                "--shot",
                str(shot),
                "--history-protocol",
                self.args.history_protocol,
                "--ranking-mode",
                "rationale",
                "--output",
                str(output),
            ]
            log_path = self.args.state_dir / f"llm_only_test_s{shot}.log"
            with log_path.open("w", encoding="utf-8") as handle:
                result = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode != 0:
                raise RuntimeError(f"LLM-only diagnostic failed for shot {shot}; see {log_path}")

    def stop_managed_servers(self) -> None:
        for server in list(self.servers.values()):
            if not server.managed or server.state_dir is None:
                continue
            port = int(server.endpoint.rsplit(":", 1)[1].split("/", 1)[0])
            environment = self.server_environment(server.gpu_id, port, server.state_dir, shared=True)
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
            self.log_event(
                "managed_server_stopped",
                gpu_id=server.gpu_id,
                exit_code=result.returncode,
                output=result.stdout[-2000:],
            )

    def run(self) -> int:
        self.register_pid()
        recovered_tasks = initialize_database(self.database_path, self.args)
        self.log_event(
            "controller_started",
            pid=os.getpid(),
            initial_gpus=self.args.initial_gpus,
            candidate_gpus=self.args.candidate_gpus,
            expected_records=self.args.expected_records,
            task_max_attempts=self.args.task_max_attempts,
            task_retry_backoff_seconds=self.args.task_retry_backoff_seconds,
        )
        if recovered_tasks:
            self.log_event(
                "interrupted_tasks_recovered",
                recovered_tasks=recovered_tasks,
            )
        try:
            gpu_state = self.query_gpu_state()
            self.add_initial_servers(gpu_state)
            if not self.servers:
                raise RuntimeError("none of the requested initial Qwen servers is available")
            while not STOP_EVENT.is_set():
                summary = self.task_summary()
                counts = summary["task_counts"]
                pending = int(counts.get("pending", 0))
                running = int(counts.get("running", 0))
                failed = int(counts.get("failed", 0))
                if pending == 0 and running == 0:
                    break
                try:
                    gpu_state = self.query_gpu_state()
                except Exception as exc:
                    self.log_event("gpu_monitor_failed", error=f"{type(exc).__name__}: {exc}")
                    gpu_state = {}
                if pending > 0:
                    self.maybe_add_idle_servers(gpu_state)
                self.write_status("generating", gpu_state)
                time.sleep(self.args.poll_seconds)

            for thread in self.worker_threads:
                thread.join(timeout=5)
            final_summary = self.task_summary()
            failed = int(final_summary["task_counts"].get("failed", 0))
            if STOP_EVENT.is_set():
                self.write_status("stopped")
                self.log_event("controller_stopped_by_signal")
                return 130
            if failed:
                self.write_status("failed")
                self.log_event("generation_failed", failed_tasks=failed)
                return 2
            self.write_status("merging")
            for shot in self.args.shots:
                self.merge_shot(shot)
            self.run_diagnostics()
            self.write_status("complete")
            self.log_event("controller_complete")
            return 0
        except Exception as exc:
            self.log_event("controller_failed", error=f"{type(exc).__name__}: {exc}")
            try:
                self.write_status("failed")
            except Exception:
                pass
            return 1
        finally:
            self.stop_managed_servers()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dynamic multi-GPU local-Qwen cache scheduler")
    parser.add_argument("--confirm-full-generation", action="store_true")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/ICEWS14")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "cache/standard_rolling_history/qwen2.5-7b-awq",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=PROJECT_ROOT / "models/Qwen2.5-7B-Instruct-AWQ"
    )
    parser.add_argument("--model", default="Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--shots", type=parse_int_list, default=parse_int_list("5,10"))
    parser.add_argument("--initial-gpus", type=parse_int_list, default=parse_int_list("2,3,4,5"))
    parser.add_argument("--candidate-gpus", type=parse_int_list, default=parse_int_list("0,1,6"))
    parser.add_argument("--adopt-endpoint", action="append", default=[])
    parser.add_argument("--num-shards", type=int, default=256)
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--retry-max-tokens", type=int, default=768)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--task-max-attempts", type=int, default=3)
    parser.add_argument("--task-retry-backoff-seconds", type=float, default=30.0)
    parser.add_argument("--request-timeout", type=float, default=360.0)
    parser.add_argument("--history-protocol", default="standard_rolling_history")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    parser.add_argument("--shared-min-free-mib", type=int, default=9500)
    parser.add_argument("--additional-min-free-mib", type=int, default=12000)
    parser.add_argument("--additional-max-utilization", type=int, default=5)
    parser.add_argument("--idle-checks", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--port-base", type=int, default=8100)
    parser.add_argument("--server-start-timeout", type=float, default=300.0)
    parser.add_argument("--server-retry-cooldown-seconds", type=float, default=120.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--expected-records", type=int, default=26358)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.confirm_full_generation:
        raise ValueError("formal generation requires explicit --confirm-full-generation")
    reboot_inhibit = args.state_dir / REBOOT_INHIBIT_FILENAME
    if reboot_inhibit.exists():
        raise RuntimeError(
            f"pre-reboot checkpoint inhibits Qwen restart: {reboot_inhibit}; "
            "use prepare_server_reboot.py unlock only after the verified reboot"
        )
    if sorted(args.shots) != [5, 10]:
        raise ValueError("the formal test-cache run requires exactly shots 5 and 10")
    if args.split != "test":
        raise ValueError("this formal launcher is restricted to the task-book test caches")
    if args.num_shards < 1 or args.workers_per_gpu < 1:
        raise ValueError("shard and worker counts must be positive")
    if args.max_retries != 1:
        raise ValueError("formal adaptive decoding is fixed to one explicit retry")
    if args.task_max_attempts < 1:
        raise ValueError("task max attempts must be positive")
    if args.task_retry_backoff_seconds < 0:
        raise ValueError("task retry backoff must be non-negative")
    if args.server_retry_cooldown_seconds < 0:
        raise ValueError("server retry cooldown must be non-negative")
    if args.retry_max_tokens < args.max_tokens:
        raise ValueError("retry max tokens must be at least the initial max tokens")
    if args.expected_records != 26358:
        raise ValueError("the two formal ICEWS14 test caches require exactly 26,358 records")
    if set(args.initial_gpus) & set(args.candidate_gpus):
        raise ValueError("initial and candidate GPU lists must not overlap")
    for path in (args.data_dir, args.model_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if not args.python_bin.is_file():
        raise FileNotFoundError(args.python_bin)


def handle_signal(signum: int, _frame: object) -> None:
    STOP_EVENT.set()


def main() -> int:
    args = build_parser().parse_args()
    args.state_dir = args.state_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.model_dir = args.model_dir.resolve()
    args.python_bin = args.python_bin.resolve()
    validate_args(args)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    return DynamicPool(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
