#!/usr/bin/env python3
"""Stop one authoritative scale-4 Qwen controller on a critical health fault.

This guard is deliberately fail-closed and narrow.  It never starts a process,
restarts a service, edits the queue database, or signals a Qwen server directly.
On one of the configured critical conditions it validates the controller PID
against the PID file, UID, cwd, command line, exact ``--state-dir`` and
``status.json`` before sending that controller SIGTERM.  It then leaves a
durable marker for a separate, human-reviewed rollback/restart step.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROC_ROOT = Path("/proc")
UTC = dt.timezone.utc
ENGINE_FAILURE_RE = re.compile(
    r"engine\s+iteration\s+timed\s+out|AsyncEngineDeadError",
    flags=re.IGNORECASE,
)


class ControllerValidationError(RuntimeError):
    """The declared controller is absent, ambiguous, or does not match scope."""


@dataclasses.dataclass(frozen=True)
class CriticalFault:
    reason: str
    observations: dict[str, object]


@dataclasses.dataclass
class ProgressTracker:
    last_records: int
    last_change_monotonic: float

    def observe(
        self,
        records: int,
        now_monotonic: float,
        stall_seconds: float,
        generating: bool,
    ) -> CriticalFault | None:
        if records != self.last_records:
            self.last_records = records
            self.last_change_monotonic = now_monotonic
            return None
        stalled_for = max(0.0, now_monotonic - self.last_change_monotonic)
        if generating and stalled_for >= stall_seconds:
            return CriticalFault(
                "no_progress",
                {
                    "records_written": records,
                    "stalled_for_seconds": round(stalled_for, 3),
                    "stall_threshold_seconds": stall_seconds,
                },
            )
        return None


@dataclasses.dataclass
class LogCursor:
    inode: int
    offset: int


def utc_now() -> str:
    return dt.datetime.now(UTC).isoformat()


def parse_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return int(text) if text.isdigit() and int(text) > 1 else None


def option_value(argv: Sequence[str], option: str) -> str | None:
    for index, item in enumerate(argv):
        if item == option and index + 1 < len(argv):
            return argv[index + 1]
        prefix = f"{option}="
        if item.startswith(prefix):
            return item[len(prefix) :]
    return None


def resolved_process_path(value: str, cwd: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else cwd / path).resolve()


def process_argv(
    pid: int,
    project_root: Path = PROJECT_ROOT,
    proc_root: Path = PROC_ROOT,
    expected_uid: int | None = None,
) -> list[str] | None:
    expected_uid = os.getuid() if expected_uid is None else expected_uid
    process_dir = proc_root / str(pid)
    try:
        if process_dir.stat().st_uid != expected_uid:
            return None
        if (process_dir / "cwd").resolve() != project_root.resolve():
            return None
        payload = (process_dir / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return [item.decode("utf-8", errors="replace") for item in payload.split(b"\0") if item]


def controller_matches(
    pid: int,
    state_dir: Path,
    project_root: Path = PROJECT_ROOT,
    proc_root: Path = PROC_ROOT,
    expected_uid: int | None = None,
) -> bool:
    argv = process_argv(pid, project_root, proc_root, expected_uid)
    if argv is None or not any(Path(item).name == "dynamic_local_qwen_pool.py" for item in argv):
        return False
    state_value = option_value(argv, "--state-dir")
    return bool(
        state_value is not None
        and resolved_process_path(state_value, project_root) == state_dir.resolve()
    )


def read_validated_status(state_dir: Path, controller_pid: int) -> dict[str, object]:
    status_path = state_dir / "status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ControllerValidationError(f"cannot read valid status.json: {exc}") from exc
    if not isinstance(status, dict):
        raise ControllerValidationError("status.json must contain an object")
    try:
        status_state_dir = Path(str(status["state_dir"])).resolve()
        status_pid = int(status["controller_pid"])
        records = int(status["records_written"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ControllerValidationError(f"status.json lacks required controller fields: {exc}") from exc
    if status_state_dir != state_dir.resolve():
        raise ControllerValidationError(
            f"status state_dir mismatch: {status_state_dir} != {state_dir.resolve()}"
        )
    if status_pid != controller_pid:
        raise ControllerValidationError(
            f"status controller_pid mismatch: {status_pid} != {controller_pid}"
        )
    if records < 0:
        raise ControllerValidationError("records_written cannot be negative")
    return status


def authoritative_controller(
    state_dir: Path,
    project_root: Path = PROJECT_ROOT,
    proc_root: Path = PROC_ROOT,
    expected_uid: int | None = None,
) -> tuple[int, dict[str, object]]:
    state_dir = state_dir.resolve()
    declared = parse_pid(state_dir / "controller.pid")
    if declared is None or not controller_matches(
        declared, state_dir, project_root, proc_root, expected_uid
    ):
        raise ControllerValidationError(
            "controller.pid is not a live project-owned controller for this state-dir"
        )
    matches: list[int] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        raise ControllerValidationError(f"cannot enumerate process table: {exc}") from exc
    for entry in entries:
        if entry.name.isdigit():
            pid = int(entry.name)
            if controller_matches(pid, state_dir, project_root, proc_root, expected_uid):
                matches.append(pid)
    matches.sort()
    if matches != [declared]:
        raise ControllerValidationError(
            f"state-dir must have exactly one authoritative controller; "
            f"declared={declared}, live={matches}"
        )
    return declared, read_validated_status(state_dir, declared)


def host_available_mib(meminfo_path: Path = Path("/proc/meminfo")) -> int:
    for line in meminfo_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdigit():
                return int(fields[1]) // 1024
    raise RuntimeError(f"MemAvailable is missing from {meminfo_path}")


def gpu_free_mib(gpu_id: int) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_id),
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()[-500:]}")
    text = result.stdout.strip()
    if not text.isdigit():
        raise RuntimeError(f"unexpected nvidia-smi memory value: {text!r}")
    return int(text)


def initial_log_cursors(state_dir: Path) -> dict[Path, LogCursor]:
    cursors: dict[Path, LogCursor] = {}
    for path in sorted((state_dir / "servers").glob("gpu*/server.log")):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        cursors[path] = LogCursor(stat.st_ino, stat.st_size)
    return cursors


def scan_new_engine_failures(
    state_dir: Path,
    cursors: dict[Path, LogCursor],
) -> CriticalFault | None:
    paths = set(cursors).union((state_dir / "servers").glob("gpu*/server.log"))
    for path in sorted(paths):
        try:
            stat = path.stat()
        except FileNotFoundError:
            cursors.pop(path, None)
            continue
        cursor = cursors.get(path)
        offset = cursor.offset if cursor and cursor.inode == stat.st_ino and stat.st_size >= cursor.offset else 0
        with path.open("rb") as handle:
            handle.seek(offset)
            payload = handle.read()
            next_offset = handle.tell()
        cursors[path] = LogCursor(stat.st_ino, next_offset)
        if not payload:
            continue
        text = payload.decode("utf-8", errors="replace")
        match = ENGINE_FAILURE_RE.search(text)
        if match:
            line = text[text.rfind("\n", 0, match.start()) + 1 : text.find("\n", match.end())]
            if not line:
                line = match.group(0)
            return CriticalFault(
                "engine_timeout",
                {
                    "log_path": str(path),
                    "matched_line": line[:2000],
                    "pattern": match.group(0),
                },
            )
    return None


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def append_event(path: Path, event: str, **fields: object) -> None:
    payload = {"at_utc": utc_now(), "event": event, **fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def stop_authoritative_controller(
    state_dir: Path,
    fault: CriticalFault,
    marker_path: Path,
    event_log: Path,
    project_root: Path = PROJECT_ROOT,
    proc_root: Path = PROC_ROOT,
    expected_uid: int | None = None,
    kill_fn: Callable[[int, int], None] = os.kill,
) -> int:
    """Validate again, persist the reason, and SIGTERM only that controller."""

    controller_pid, status = authoritative_controller(
        state_dir, project_root, proc_root, expected_uid
    )
    marker = {
        "schema_version": 1,
        "triggered_at_utc": utc_now(),
        "reason": fault.reason,
        "observations": fault.observations,
        "controller_pid": controller_pid,
        "controller_signal": "SIGTERM",
        "signal_sent": False,
        "state_dir": str(state_dir.resolve()),
        "status_phase": status.get("phase"),
        "records_written": status.get("records_written"),
        "action_required": (
            "Inspect logs and perform a separate worker-count rollback/restart if appropriate; "
            "this guard does not restart processes or modify queue.sqlite3."
        ),
    }
    atomic_json(marker_path, marker)
    append_event(
        event_log,
        "critical_fault_validated",
        reason=fault.reason,
        controller_pid=controller_pid,
        marker=str(marker_path),
        observations=fault.observations,
    )
    try:
        # Close the validation-to-signal gap as far as practical without
        # obtaining broader process authority: the exact identity is checked a
        # second time immediately before SIGTERM.
        revalidated_pid, _ = authoritative_controller(
            state_dir, project_root, proc_root, expected_uid
        )
        if revalidated_pid != controller_pid:
            raise ControllerValidationError(
                f"controller changed before signal: {controller_pid} -> {revalidated_pid}"
            )
        kill_fn(controller_pid, signal.SIGTERM)
    except Exception as exc:
        marker["signal_error"] = f"{type(exc).__name__}: {exc}"
        marker["updated_at_utc"] = utc_now()
        atomic_json(marker_path, marker)
        append_event(
            event_log,
            "controller_term_refused_or_failed",
            controller_pid=controller_pid,
            error=marker["signal_error"],
        )
        raise
    marker["signal_sent"] = True
    marker["signal_sent_at_utc"] = utc_now()
    atomic_json(marker_path, marker)
    append_event(
        event_log,
        "controller_term_sent",
        controller_pid=controller_pid,
        reason=fault.reason,
    )
    return controller_pid


def validate_paths(args: argparse.Namespace) -> None:
    args.state_dir = args.state_dir.resolve()
    args.marker = (args.marker or args.state_dir / "SCALE4_ROLLBACK_REQUIRED.json").resolve()
    args.event_log = (args.event_log or args.state_dir / "scale4_health_guard.jsonl").resolve()
    args.pid_file = (args.pid_file or args.state_dir / "scale4_health_guard.pid").resolve()
    try:
        args.state_dir.relative_to(PROJECT_ROOT)
        args.marker.relative_to(args.state_dir)
        args.event_log.relative_to(args.state_dir)
        args.pid_file.relative_to(args.state_dir)
    except ValueError as exc:
        raise ValueError("state and guard output paths must stay inside the project state-dir") from exc
    if not args.state_dir.is_dir():
        raise FileNotFoundError(args.state_dir)
    if args.check_seconds < 5:
        raise ValueError("check-seconds must be at least 5")
    if args.stall_seconds < 60:
        raise ValueError("stall-seconds must be at least 60")
    if args.gpu_free_floor_mib <= 0 or args.host_free_floor_mib <= 0:
        raise ValueError("resource floors must be positive")


def check_fault(
    args: argparse.Namespace,
    status: dict[str, object],
    progress: ProgressTracker,
    cursors: dict[Path, LogCursor],
    now_monotonic: float,
) -> CriticalFault | None:
    engine_fault = scan_new_engine_failures(args.state_dir, cursors)
    if engine_fault is not None:
        return engine_fault
    available = host_available_mib()
    if available < args.host_free_floor_mib:
        return CriticalFault(
            "host_memory_pressure",
            {
                "host_available_mib": available,
                "host_free_floor_mib": args.host_free_floor_mib,
            },
        )
    free = gpu_free_mib(args.gpu_id)
    if free < args.gpu_free_floor_mib:
        return CriticalFault(
            "gpu_memory_pressure",
            {
                "gpu_id": args.gpu_id,
                "gpu_free_mib": free,
                "gpu_free_floor_mib": args.gpu_free_floor_mib,
            },
        )
    phase = str(status.get("phase", ""))
    task_counts = status.get("task_counts")
    active_work = True
    if isinstance(task_counts, dict):
        active_work = int(task_counts.get("pending", 0)) + int(task_counts.get("running", 0)) > 0
    generating = phase == "generating" and active_work
    return progress.observe(
        int(status["records_written"]),
        now_monotonic,
        args.stall_seconds,
        generating,
    )


def run(args: argparse.Namespace) -> int:
    validate_paths(args)
    controller_pid, status = authoritative_controller(args.state_dir)
    if args.marker.exists() and not args.dry_run:
        raise RuntimeError(f"refusing to run while an unresolved marker exists: {args.marker}")
    progress = ProgressTracker(int(status["records_written"]), time.monotonic())
    cursors = initial_log_cursors(args.state_dir)

    if args.dry_run:
        fault = check_fault(args, status, progress, cursors, time.monotonic())
        print(
            json.dumps(
                {
                    "controller_pid": controller_pid,
                    "dry_run": True,
                    "fault": dataclasses.asdict(fault) if fault else None,
                    "host_available_mib": host_available_mib(),
                    "gpu_free_mib": gpu_free_mib(args.gpu_id),
                    "records_written": status["records_written"],
                    "state_dir": str(args.state_dir),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 2 if fault else 0

    lock_path = args.state_dir / "scale4_health_guard.lock"
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(f"another health guard already holds {lock_path}") from exc
    atomic_json(args.pid_file, {"pid": os.getpid(), "started_at_utc": utc_now()})
    append_event(
        args.event_log,
        "guard_started",
        pid=os.getpid(),
        controller_pid=controller_pid,
        state_dir=str(args.state_dir),
        gpu_id=args.gpu_id,
        gpu_free_floor_mib=args.gpu_free_floor_mib,
        host_free_floor_mib=args.host_free_floor_mib,
        stall_seconds=args.stall_seconds,
        check_seconds=args.check_seconds,
    )
    try:
        while True:
            try:
                current_pid, status = authoritative_controller(args.state_dir)
            except ControllerValidationError as exc:
                append_event(args.event_log, "controller_no_longer_authoritative", error=str(exc))
                return 0
            if current_pid != controller_pid:
                append_event(
                    args.event_log,
                    "controller_identity_changed",
                    original_pid=controller_pid,
                    current_pid=current_pid,
                )
                return 3
            phase = str(status.get("phase", ""))
            if phase in {"complete", "failed", "stopped"}:
                append_event(args.event_log, "guard_finished_with_controller", phase=phase)
                return 0
            try:
                fault = check_fault(args, status, progress, cursors, time.monotonic())
            except Exception as exc:
                append_event(
                    args.event_log,
                    "health_check_unavailable",
                    error=f"{type(exc).__name__}: {exc}",
                )
                fault = None
            if fault is not None:
                stop_authoritative_controller(
                    args.state_dir,
                    fault,
                    args.marker,
                    args.event_log,
                )
                return 2
            time.sleep(args.check_seconds)
    finally:
        try:
            payload = json.loads(args.pid_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            payload = {}
        if payload.get("pid") == os.getpid():
            args.pid_file.unlink(missing_ok=True)
        append_event(args.event_log, "guard_stopped", pid=os.getpid())
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stop the authoritative scale-4 Qwen controller on a critical health fault"
    )
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=3)
    parser.add_argument("--gpu-free-floor-mib", type=int, default=2048)
    parser.add_argument("--host-free-floor-mib", type=int, default=4096)
    parser.add_argument("--stall-seconds", type=float, default=600.0)
    parser.add_argument("--check-seconds", type=float, default=30.0)
    parser.add_argument("--marker", type=Path)
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="inspect once; never signal or write")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.gpu_id < 0:
        raise ValueError("gpu-id must be non-negative")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
