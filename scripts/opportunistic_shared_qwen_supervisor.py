#!/usr/bin/env python3
"""Opportunistically attach one shared GPU to the authoritative Qwen queue.

The supervisor is intentionally narrower than ``dynamic_local_qwen_pool.py``:
it only admits a card which is already shared with another compute process,
starts one project-owned loopback server, and attaches a sidecar to one
explicitly validated controller state directory.  A separate guard tears down
that project-owned stack under pressure; this process then waits until the
start thresholds are met again and may rejoin the same queue.

No signal is sent unless the PID belongs to the current UID, has this project
as its cwd, and has the expected command line.  Other users' processes are
observed only as anonymous GPU occupancy and are never modified.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fcntl
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UTC = dt.timezone.utc
STOP_REQUESTED = False


@dataclasses.dataclass(frozen=True)
class ResourceSnapshot:
    host_available_mib: int
    gpu_free_mib: int
    gpu_used_mib: int
    compute_pids: tuple[int, ...]


def utc_now() -> str:
    return dt.datetime.now(UTC).isoformat()


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def parse_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return int(text) if text.isdigit() and int(text) > 1 else None


def process_argv(pid: int) -> list[str] | None:
    try:
        if os.stat(f"/proc/{pid}").st_uid != os.getuid():
            return None
        if Path(f"/proc/{pid}/cwd").resolve() != PROJECT_ROOT:
            return None
        payload = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return [item.decode("utf-8", errors="replace") for item in payload.split(b"\0") if item]


def process_environment(pid: int) -> dict[str, str] | None:
    if process_argv(pid) is None:
        return None
    try:
        payload = Path(f"/proc/{pid}/environ").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    result: dict[str, str] = {}
    for item in payload.split(b"\0"):
        key, separator, value = item.partition(b"=")
        if separator:
            result[key.decode("utf-8", errors="replace")] = value.decode(
                "utf-8", errors="replace"
            )
    return result


def option_value(argv: Sequence[str], option: str) -> str | None:
    for index, item in enumerate(argv):
        if item == option and index + 1 < len(argv):
            return argv[index + 1]
        prefix = f"{option}="
        if item.startswith(prefix):
            return item[len(prefix) :]
    return None


def resolved_process_path(value: str, cwd: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else cwd / path).resolve()


def controller_matches(pid: int, state_dir: Path) -> bool:
    argv = process_argv(pid)
    if argv is None or not any(Path(item).name == "dynamic_local_qwen_pool.py" for item in argv):
        return False
    state_value = option_value(argv, "--state-dir")
    return state_value is not None and resolved_process_path(state_value) == state_dir


def authoritative_controller(state_dir: Path) -> int:
    declared = parse_pid(state_dir / "controller.pid")
    if declared is None or not controller_matches(declared, state_dir):
        raise RuntimeError("controller.pid is not a live project-owned controller for this state-dir")
    matches: list[int] = []
    for proc_entry in Path("/proc").iterdir():
        if proc_entry.name.isdigit():
            pid = int(proc_entry.name)
            if controller_matches(pid, state_dir):
                matches.append(pid)
    if matches != [declared]:
        raise RuntimeError(
            f"state-dir must have exactly one authoritative controller; declared={declared}, live={matches}"
        )
    return declared


def server_matches(pid: int, port: int) -> bool:
    argv = process_argv(pid)
    if argv is None or not any(Path(item).name == "serve_local_qwen_loopback.py" for item in argv):
        return False
    return option_value(argv, "--port") == str(port)


def server_matches_gpu(pid: int, gpu_id: int) -> bool:
    argv = process_argv(pid)
    if argv is None or not any(Path(item).name == "serve_local_qwen_loopback.py" for item in argv):
        return False
    environment = process_environment(pid)
    return environment is not None and environment.get("CUDA_VISIBLE_DEVICES") == str(gpu_id)


def sidecar_matches(pid: int, state_dir: Path, gpu_id: int) -> bool:
    argv = process_argv(pid)
    if argv is None or not any(Path(item).name == "attach_shared_qwen_worker.py" for item in argv):
        return False
    state_value = option_value(argv, "--state-dir")
    return bool(
        state_value is not None
        and resolved_process_path(state_value) == state_dir
        and option_value(argv, "--gpu-id") == str(gpu_id)
    )


def guard_matches(pid: int, state_dir: Path, gpu_id: int) -> bool:
    argv = process_argv(pid)
    if argv is None:
        return False
    try:
        script_index = next(
            index for index, item in enumerate(argv) if Path(item).name == "guard_shared_qwen_worker.sh"
        )
    except StopIteration:
        return False
    if len(argv) <= script_index + 2:
        return False
    return (
        resolved_process_path(argv[script_index + 1]) == state_dir
        and argv[script_index + 2] == str(gpu_id)
    )


def shared_stack_absent(state_dir: Path, gpu_id: int, port: int) -> tuple[bool, str]:
    """Prove that no matching or suspicious live stack process exists.

    PID files are checked first and fail closed if they now identify a live
    process that does not match the expected UID/cwd/cmdline.  A full /proc
    scan then catches a matching project process even when its PID file was
    lost.  Other users' processes are never signalled or otherwise modified.
    """

    base_path = state_dir / "servers" / f"gpu{gpu_id}"
    checks = (
        (base_path / "server.pid", lambda pid: server_matches(pid, port), "server"),
        (Path(f"{base_path}_sidecar.pid"), lambda pid: sidecar_matches(pid, state_dir, gpu_id), "sidecar"),
        (Path(f"{base_path}_guard.pid"), lambda pid: guard_matches(pid, state_dir, gpu_id), "guard"),
    )
    for pid_file, matcher, label in checks:
        pid = parse_pid(pid_file)
        if pid is None or not Path(f"/proc/{pid}").exists():
            continue
        if matcher(pid):
            return False, f"live_{label}_pid_file"
        return False, f"unsafe_live_{label}_pid_file"

    for proc_entry in Path("/proc").iterdir():
        if not proc_entry.name.isdigit():
            continue
        pid = int(proc_entry.name)
        if server_matches(pid, port) or server_matches_gpu(pid, gpu_id):
            return False, "live_server_process_scan"
        if sidecar_matches(pid, state_dir, gpu_id):
            return False, "live_sidecar_process_scan"
        if guard_matches(pid, state_dir, gpu_id):
            return False, "live_guard_process_scan"
    return True, "no_live_shared_stack"


def write_pid_atomic(path: Path, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(f"{pid}\n", encoding="utf-8")
    os.replace(temporary, path)


def remove_pid_if_equal(path: Path, pid: int) -> None:
    if parse_pid(path) == pid:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def append_event(path: Path, event: str, **fields: object) -> None:
    payload = {"at_utc": utc_now(), "event": event, **fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_configuration(state_dir: Path) -> dict[str, object]:
    database = state_dir / "queue.sqlite3"
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=30)
    try:
        row = connection.execute(
            "SELECT value FROM run_meta WHERE key='configuration_json'"
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise RuntimeError("queue has no configuration_json")
    config = json.loads(str(row[0]))
    if not isinstance(config, dict):
        raise RuntimeError("queue configuration is not an object")
    if config.get("split") != "test" or sorted(config.get("shots", [])) != [5, 10]:
        raise RuntimeError("opportunistic sidecar is restricted to the formal test shot-5/10 queue")
    return config


def queue_counts(state_dir: Path) -> dict[str, int]:
    database = state_dir / "queue.sqlite3"
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=30)
    try:
        return {
            str(status): int(count)
            for status, count in connection.execute(
                "SELECT status,COUNT(*) FROM tasks GROUP BY status"
            )
        }
    finally:
        connection.close()


def active_workers(state_dir: Path, gpu_id: int) -> list[tuple[object, ...]]:
    database = state_dir / "queue.sqlite3"
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=30)
    try:
        return list(
            connection.execute(
                """
                SELECT worker_id,status,task_id,endpoint FROM workers
                WHERE gpu_id=? AND status IN ('ready','running','retry_wait')
                ORDER BY worker_id
                """,
                (gpu_id,),
            )
        )
    finally:
        connection.close()


def retire_stale_worker_rows_if_safe(
    state_dir: Path,
    gpu_id: int,
    port: int,
    event_log: Path,
) -> tuple[list[tuple[object, ...]], str]:
    """Atomically retire provably orphaned, task-free worker conflicts.

    The audit is stored atomically in each affected ``workers.error`` field in
    the same transaction as ``status='stopped'``.  The append-only supervisor
    event is additional evidence, not the sole audit record.
    """

    absent, reason = shared_stack_absent(state_dir, gpu_id, port)
    if not absent:
        return [], reason
    server_state_dir = state_dir / "servers" / f"gpu{gpu_id}"
    server_state_dir.mkdir(parents=True, exist_ok=True)
    start_lock_path = server_state_dir / "start.lock"
    lock_handle = start_lock_path.open("a+")
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return [], "server_start_lock_held"
        absent, reason = shared_stack_absent(state_dir, gpu_id, port)
        if not absent:
            return [], reason

        database = state_dir / "queue.sqlite3"
        connection = sqlite3.connect(str(database), timeout=30)
        try:
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("BEGIN IMMEDIATE")
            rows = list(
                connection.execute(
                    """
                    SELECT worker_id,status,task_id,endpoint FROM workers
                    WHERE gpu_id=? AND status IN ('ready','running','retry_wait')
                    ORDER BY worker_id
                    """,
                    (gpu_id,),
                )
            )
            if not rows:
                connection.rollback()
                return [], "no_active_worker_conflicts"
            if any(str(row[1]) == "running" or row[2] is not None for row in rows):
                connection.rollback()
                return [], "active_worker_owns_task"

            audited_at = utc_now()
            audit_reason = (
                "opportunistic supervisor retired stale worker row after proving "
                "no live sidecar, guard, or server; row had no task ownership"
            )
            recovered: list[tuple[object, ...]] = []
            for worker_id, status, task_id, endpoint in rows:
                cursor = connection.execute(
                    """
                    UPDATE workers
                    SET status='stopped',task_id=NULL,updated_at_utc=?,error=?
                    WHERE worker_id=? AND gpu_id=? AND status=? AND task_id IS NULL
                    """,
                    (audited_at, audit_reason, worker_id, gpu_id, status),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(f"worker row changed during stale recovery: {worker_id}")
                recovered.append((worker_id, status, task_id, endpoint))
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
    finally:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()

    append_event(
        event_log,
        "stale_worker_rows_stopped",
        gpu_id=gpu_id,
        rows=recovered,
        audit_reason=audit_reason,
    )
    return recovered, "stale_worker_rows_stopped"


def resource_snapshot(gpu_id: int) -> ResourceSnapshot:
    available_mib = 0
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            available_mib = int(line.split()[1]) // 1024
            break
    if available_mib <= 0:
        raise RuntimeError("could not read host MemAvailable")

    gpu_output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu_id),
            "--query-gpu=index,uuid,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    )
    gpu_uuid = None
    gpu_used = None
    gpu_free = None
    for line in gpu_output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4 and fields[0].isdigit() and int(fields[0]) == gpu_id:
            gpu_uuid, gpu_used, gpu_free = fields[1], int(fields[2]), int(fields[3])
            break
    if gpu_uuid is None or gpu_used is None or gpu_free is None:
        raise RuntimeError(f"GPU {gpu_id} was not returned by nvidia-smi")

    # Scope the process query to the same physical card.  A broken unrelated
    # GPU must not make this supervisor blind, while a failed query for the
    # requested card must fail closed instead of looking falsely idle.
    process_output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu_id),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    )
    compute_pids = []
    for line in process_output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0].isdigit():
            compute_pids.append(int(fields[0]))
    return ResourceSnapshot(
        host_available_mib=available_mib,
        gpu_free_mib=gpu_free,
        gpu_used_mib=gpu_used,
        compute_pids=tuple(sorted(set(compute_pids))),
    )


def admission_decision(
    snapshot: ResourceSnapshot,
    *,
    pending_tasks: int,
    active_worker_count: int,
    server_running: bool,
    start_gpu_free_mib: int,
    start_host_available_mib: int,
) -> str:
    """Return a side-effect-free admission decision for tests and dry-runs."""

    if pending_tasks <= 0:
        return "wait_no_pending_tasks"
    if active_worker_count:
        return "wait_gpu_already_attached"
    if server_running:
        return "wait_existing_server_ownership_unknown"
    # A fully idle card belongs to the authoritative controller's normal
    # auto-admission path.  This shared-card supervisor only fills the gap when
    # another compute PID keeps that controller from claiming the GPU.
    if not snapshot.compute_pids:
        return "defer_idle_gpu_to_controller"
    if snapshot.gpu_free_mib < start_gpu_free_mib:
        return "wait_gpu_start_headroom"
    if snapshot.host_available_mib < start_host_available_mib:
        return "wait_host_start_headroom"
    return "start_shared_stack"


def start_server(
    args: argparse.Namespace,
    config: dict[str, object],
    server_state_dir: Path,
    event_log: Path,
) -> int | None:
    environment = os.environ.copy()
    environment.update(
        {
            "LOCAL_QWEN_ENV_FILE": "/dev/null",
            "QWEN_PYTHON": str(args.qwen_python),
            "LOCAL_QWEN_GPU_ID": str(args.gpu_id),
            "LOCAL_QWEN_PORT": str(args.port),
            "LOCAL_QWEN_STATE_DIR": str(server_state_dir),
            "LOCAL_QWEN_MODEL_DIR": str(args.model_dir),
            "LOCAL_QWEN_MODEL": str(config["model"]),
            "LOCAL_QWEN_MAX_MODEL_LEN": str(args.max_model_len),
            "LOCAL_QWEN_GPU_MEMORY_UTILIZATION": str(args.gpu_memory_utilization),
            "LOCAL_QWEN_MAX_NUM_SEQS": str(config["workers_per_gpu"]),
            "LOCAL_QWEN_MIN_FREE_MIB": str(args.start_gpu_free_mib),
            "LOCAL_QWEN_ENFORCE_EAGER": "YES",
            "LOCAL_QWEN_DISABLE_FRONTEND_MULTIPROCESSING": "YES",
            "LOCAL_QWEN_GUIDED_DECODING_BACKEND": "lm-format-enforcer",
            "LOCAL_QWEN_QUANTIZATION": "awq_marlin",
            "LOCAL_QWEN_START_LOCK_TIMEOUT_SECONDS": str(args.server_start_timeout_seconds),
            "ALLOW_SHARED_GPU": "YES",
        }
    )
    result = subprocess.run(
        ["bash", "scripts/start_local_qwen_server.sh"],
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=args.server_start_timeout_seconds + 30,
        check=False,
    )
    append_event(
        event_log,
        "server_start_finished",
        exit_code=result.returncode,
        output=result.stdout[-4000:],
    )
    pid = parse_pid(server_state_dir / "server.pid")
    if result.returncode != 0 or pid is None or not server_matches(pid, args.port):
        return None
    return pid


def launch_logged(command: list[str], log_path: Path, environment: dict[str, str] | None = None) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle: IO[bytes] = log_path.open("ab")
    try:
        return subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()


def stop_owned_sidecar(pid: int, state_dir: Path, gpu_id: int, event_log: Path) -> None:
    if sidecar_matches(pid, state_dir, gpu_id):
        os.kill(pid, signal.SIGTERM)
        append_event(event_log, "sidecar_term_sent", pid=pid)


def stop_owned_server(
    args: argparse.Namespace,
    server_state_dir: Path,
    owned_server_pid_file: Path,
    event_log: Path,
) -> None:
    owned_pid = parse_pid(owned_server_pid_file)
    actual_pid = parse_pid(server_state_dir / "server.pid")
    if owned_pid is None or actual_pid != owned_pid or not server_matches(owned_pid, args.port):
        return
    environment = os.environ.copy()
    environment.update(
        {
            "LOCAL_QWEN_ENV_FILE": "/dev/null",
            "LOCAL_QWEN_STATE_DIR": str(server_state_dir),
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
    append_event(
        event_log,
        "owned_server_stop_finished",
        pid=owned_pid,
        exit_code=result.returncode,
        output=result.stdout[-2000:],
    )
    if not server_matches(owned_pid, args.port):
        remove_pid_if_equal(owned_server_pid_file, owned_pid)


def validate_paths(args: argparse.Namespace) -> None:
    args.state_dir = args.state_dir.resolve()
    args.model_dir = args.model_dir.resolve()
    args.python_bin = args.python_bin.resolve()
    args.qwen_python = args.qwen_python.resolve()
    reboot_inhibit = args.state_dir / "PRE_REBOOT_CHECKPOINT.lock"
    if reboot_inhibit.exists() and not args.dry_run:
        raise RuntimeError(
            f"pre-reboot checkpoint inhibits shared-Qwen startup: {reboot_inhibit}"
        )
    try:
        args.state_dir.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("state-dir must be inside this project") from exc
    if not (args.state_dir / "queue.sqlite3").is_file():
        raise FileNotFoundError(args.state_dir / "queue.sqlite3")
    if not (args.model_dir / "config.json").is_file():
        raise FileNotFoundError(args.model_dir / "config.json")
    for executable in (args.python_bin, args.qwen_python):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(executable)
    for name in (
        "start_local_qwen_server.sh",
        "stop_local_qwen_server.sh",
        "guard_shared_qwen_worker.sh",
        "attach_shared_qwen_worker.py",
    ):
        if not (PROJECT_ROOT / "scripts" / name).is_file():
            raise FileNotFoundError(PROJECT_ROOT / "scripts" / name)
    if args.start_gpu_free_mib < args.runtime_gpu_free_mib:
        raise ValueError("start GPU threshold must be at least the runtime GPU floor")
    if args.start_host_available_mib < args.runtime_host_available_mib:
        raise ValueError("start host threshold must be at least the runtime host floor")


def run(args: argparse.Namespace) -> int:
    global STOP_REQUESTED

    validate_paths(args)
    config = read_configuration(args.state_dir)
    server_state_dir = args.state_dir / "servers" / f"gpu{args.gpu_id}"
    base_path = args.state_dir / "servers" / f"gpu{args.gpu_id}"
    supervisor_pid_file = Path(f"{base_path}_supervisor.pid")
    sidecar_pid_file = Path(f"{base_path}_sidecar.pid")
    guard_pid_file = Path(f"{base_path}_guard.pid")
    owned_server_pid_file = Path(f"{base_path}_opportunistic_server.pid")
    event_log = Path(f"{base_path}_supervisor.jsonl")

    if args.dry_run:
        controller_pid = authoritative_controller(args.state_dir)
        counts = queue_counts(args.state_dir)
        conflicts = active_workers(args.state_dir, args.gpu_id)
        snapshot = resource_snapshot(args.gpu_id)
        server_pid = parse_pid(server_state_dir / "server.pid")
        server_running = server_pid is not None and server_matches(server_pid, args.port)
        decision = admission_decision(
            snapshot,
            pending_tasks=counts.get("pending", 0),
            active_worker_count=len(conflicts),
            server_running=server_running,
            start_gpu_free_mib=args.start_gpu_free_mib,
            start_host_available_mib=args.start_host_available_mib,
        )
        print(
            json.dumps(
                {
                    "active_workers": conflicts,
                    "controller_pid": controller_pid,
                    "decision": decision,
                    "dry_run": True,
                    "gpu_id": args.gpu_id,
                    "queue_counts": counts,
                    "snapshot": dataclasses.asdict(snapshot),
                    "state_dir": str(args.state_dir),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.confirm_opportunistic_supervisor:
        raise ValueError("live supervision requires --confirm-opportunistic-supervisor")

    lock_path = Path(f"{base_path}_supervisor.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(f"another supervisor already holds {lock_path}") from exc
    write_pid_atomic(supervisor_pid_file, os.getpid())
    append_event(
        event_log,
        "supervisor_started",
        pid=os.getpid(),
        gpu_id=args.gpu_id,
        start_gpu_free_mib=args.start_gpu_free_mib,
        start_host_available_mib=args.start_host_available_mib,
        runtime_gpu_free_mib=args.runtime_gpu_free_mib,
        runtime_host_available_mib=args.runtime_host_available_mib,
    )

    def handle_stop(_signum: int, _frame: object) -> None:
        global STOP_REQUESTED
        STOP_REQUESTED = True

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    try:
        while not STOP_REQUESTED:
            try:
                controller_pid = authoritative_controller(args.state_dir)
                counts = queue_counts(args.state_dir)
                if counts.get("pending", 0) == 0 and counts.get("running", 0) == 0:
                    append_event(event_log, "queue_terminal", queue_counts=counts)
                    return 0
                conflicts = active_workers(args.state_dir, args.gpu_id)
                if conflicts:
                    recovered_rows, recovery_status = retire_stale_worker_rows_if_safe(
                        args.state_dir,
                        args.gpu_id,
                        args.port,
                        event_log,
                    )
                    if recovered_rows:
                        conflicts = active_workers(args.state_dir, args.gpu_id)
                snapshot = resource_snapshot(args.gpu_id)
                server_pid = parse_pid(server_state_dir / "server.pid")
                server_running = server_pid is not None and server_matches(server_pid, args.port)
                decision = admission_decision(
                    snapshot,
                    pending_tasks=counts.get("pending", 0),
                    active_worker_count=len(conflicts),
                    server_running=server_running,
                    start_gpu_free_mib=args.start_gpu_free_mib,
                    start_host_available_mib=args.start_host_available_mib,
                )
                if decision != "start_shared_stack":
                    append_event(
                        event_log,
                        "admission_wait",
                        decision=decision,
                        controller_pid=controller_pid,
                        queue_counts=counts,
                        host_available_mib=snapshot.host_available_mib,
                        gpu_free_mib=snapshot.gpu_free_mib,
                        compute_pid_count=len(snapshot.compute_pids),
                        stale_recovery_status=recovery_status if conflicts else None,
                    )
                    time.sleep(args.poll_seconds)
                    continue

                # Revalidate immediately before startup.  The shared-process
                # requirement makes the controller's idle-card admission false;
                # start_local_qwen_server.sh adds a per-state startup lock for
                # the remaining sub-second race.
                authoritative_controller(args.state_dir)
                if active_workers(args.state_dir, args.gpu_id):
                    time.sleep(args.poll_seconds)
                    continue
                server_pid = start_server(args, config, server_state_dir, event_log)
                if server_pid is None:
                    time.sleep(args.retry_seconds)
                    continue
                write_pid_atomic(owned_server_pid_file, server_pid)
                if STOP_REQUESTED:
                    stop_owned_server(args, server_state_dir, owned_server_pid_file, event_log)
                    return 130

                # If the controller won the startup lock and attached workers,
                # yield without claiming or stopping its service.
                authoritative_controller(args.state_dir)
                post_start_conflicts = active_workers(args.state_dir, args.gpu_id)
                if post_start_conflicts:
                    remove_pid_if_equal(owned_server_pid_file, server_pid)
                    append_event(
                        event_log,
                        "yielded_server_to_controller",
                        server_pid=server_pid,
                        active_workers=post_start_conflicts,
                    )
                    time.sleep(args.poll_seconds)
                    continue

                endpoint = f"http://127.0.0.1:{args.port}/v1"
                sidecar_command = [
                    str(args.python_bin),
                    "scripts/attach_shared_qwen_worker.py",
                    "--confirm-shared-worker",
                    "--state-dir",
                    str(args.state_dir),
                    "--gpu-id",
                    str(args.gpu_id),
                    "--endpoint",
                    endpoint,
                    "--model-dir",
                    str(args.model_dir),
                    "--python-bin",
                    str(args.python_bin),
                    "--request-timeout",
                    str(args.request_timeout),
                    "--stop-server-on-exit",
                ]
                sidecar = launch_logged(
                    sidecar_command,
                    Path(f"{base_path}_sidecar.log"),
                )
                write_pid_atomic(sidecar_pid_file, sidecar.pid)
                time.sleep(2)
                if sidecar.poll() is not None or not sidecar_matches(
                    sidecar.pid, args.state_dir, args.gpu_id
                ):
                    append_event(
                        event_log,
                        "sidecar_start_failed",
                        pid=sidecar.pid,
                        exit_code=sidecar.poll(),
                    )
                    remove_pid_if_equal(sidecar_pid_file, sidecar.pid)
                    stop_owned_server(args, server_state_dir, owned_server_pid_file, event_log)
                    time.sleep(args.retry_seconds)
                    continue

                guard_environment = os.environ.copy()
                guard_environment.update(
                    {
                        "QWEN_GUARD_MIN_GPU_FREE_MIB": str(args.runtime_gpu_free_mib),
                        "QWEN_GUARD_CHECK_SECONDS": str(args.guard_check_seconds),
                        "QWEN_GUARD_CONSECUTIVE_LIMIT": str(args.guard_consecutive_limit),
                    }
                )
                guard_command = [
                    "bash",
                    "scripts/guard_shared_qwen_worker.sh",
                    str(args.state_dir),
                    str(args.gpu_id),
                    str(args.runtime_host_available_mib),
                ]
                guard = launch_logged(
                    guard_command,
                    Path(f"{base_path}_guard.stdout.log"),
                    guard_environment,
                )
                write_pid_atomic(guard_pid_file, guard.pid)
                append_event(
                    event_log,
                    "shared_stack_started",
                    controller_pid=controller_pid,
                    server_pid=server_pid,
                    sidecar_pid=sidecar.pid,
                    guard_pid=guard.pid,
                    endpoint=endpoint,
                )

                while not STOP_REQUESTED and sidecar.poll() is None and guard.poll() is None:
                    try:
                        authoritative_controller(args.state_dir)
                    except RuntimeError as exc:
                        append_event(event_log, "controller_lost", error=str(exc))
                        stop_owned_sidecar(sidecar.pid, args.state_dir, args.gpu_id, event_log)
                        break
                    time.sleep(min(args.poll_seconds, 10.0))
                if STOP_REQUESTED and sidecar.poll() is None:
                    stop_owned_sidecar(sidecar.pid, args.state_dir, args.gpu_id, event_log)
                if guard.poll() is not None and sidecar.poll() is None:
                    # Never leave an unguarded shared worker running.
                    stop_owned_sidecar(sidecar.pid, args.state_dir, args.gpu_id, event_log)
                try:
                    sidecar.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    append_event(event_log, "sidecar_stop_timeout", pid=sidecar.pid)
                try:
                    guard.wait(timeout=args.guard_shutdown_timeout_seconds)
                except subprocess.TimeoutExpired:
                    append_event(event_log, "guard_stop_timeout", pid=guard.pid)
                if sidecar.poll() is not None:
                    remove_pid_if_equal(sidecar_pid_file, sidecar.pid)
                if guard.poll() is not None:
                    remove_pid_if_equal(guard_pid_file, guard.pid)
                stop_owned_server(args, server_state_dir, owned_server_pid_file, event_log)
                append_event(
                    event_log,
                    "shared_stack_finished",
                    sidecar_exit_code=sidecar.poll(),
                    guard_exit_code=guard.poll(),
                )
                if not STOP_REQUESTED:
                    time.sleep(args.retry_seconds)
            except (OSError, RuntimeError, sqlite3.Error, subprocess.SubprocessError) as exc:
                append_event(
                    event_log,
                    "supervisor_wait_after_error",
                    error=f"{type(exc).__name__}: {exc}",
                )
                # If an error happened after this supervisor had started a
                # stack, retire only its validated sidecar.  The guard then
                # performs the normal pressure-safe server shutdown.  With no
                # live sidecar, stop only a server whose PID is also present in
                # this supervisor's ownership marker.
                owned_sidecar_pid = parse_pid(sidecar_pid_file)
                if owned_sidecar_pid is not None and sidecar_matches(
                    owned_sidecar_pid, args.state_dir, args.gpu_id
                ):
                    stop_owned_sidecar(
                        owned_sidecar_pid,
                        args.state_dir,
                        args.gpu_id,
                        event_log,
                    )
                stop_owned_server(
                    args,
                    server_state_dir,
                    owned_server_pid_file,
                    event_log,
                )
                time.sleep(args.retry_seconds)
        return 130
    finally:
        remove_pid_if_equal(supervisor_pid_file, os.getpid())
        append_event(event_log, "supervisor_stopped", pid=os.getpid())
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Opportunistically add one shared GPU to a live local-Qwen queue"
    )
    parser.add_argument("--confirm-opportunistic-supervisor", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="inspect once without writes or processes")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "models/Qwen2.5-7B-Instruct-AWQ",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument(
        "--qwen-python",
        type=Path,
        default=Path(os.environ.get("QWEN_PYTHON", sys.executable)),
    )
    parser.add_argument(
        "--start-gpu-free-mib",
        type=int,
        default=env_int("QWEN_OPPORTUNISTIC_START_GPU_FREE_MIB", 11000),
    )
    parser.add_argument(
        "--start-host-available-mib",
        type=int,
        default=env_int("QWEN_OPPORTUNISTIC_START_HOST_AVAILABLE_MIB", 10000),
    )
    parser.add_argument(
        "--runtime-gpu-free-mib",
        type=int,
        default=env_int("QWEN_OPPORTUNISTIC_RUNTIME_GPU_FREE_MIB", 1024),
    )
    parser.add_argument(
        "--runtime-host-available-mib",
        type=int,
        default=env_int("QWEN_OPPORTUNISTIC_RUNTIME_HOST_AVAILABLE_MIB", 2048),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--retry-seconds", type=float, default=60.0)
    parser.add_argument("--guard-check-seconds", type=int, default=20)
    parser.add_argument("--guard-consecutive-limit", type=int, default=3)
    parser.add_argument("--guard-shutdown-timeout-seconds", type=float, default=75.0)
    parser.add_argument("--server-start-timeout-seconds", type=int, default=300)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.port is None:
        args.port = 8100 + args.gpu_id
    if args.gpu_id < 0 or not (1 <= args.port <= 65535):
        raise ValueError("GPU id and loopback port are invalid")
    for value in (
        args.start_gpu_free_mib,
        args.start_host_available_mib,
        args.runtime_gpu_free_mib,
        args.runtime_host_available_mib,
        args.poll_seconds,
        args.retry_seconds,
    ):
        if value <= 0:
            raise ValueError("resource thresholds and intervals must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
