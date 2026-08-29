"""Cross-process gate for short-lived local-Qwen queue/cache mutators."""

from __future__ import annotations

import contextlib
import fcntl
import os
import stat
from pathlib import Path
from typing import Iterator, TextIO


MUTATION_GATE_FILENAME = "PRE_REBOOT_MUTATION_GATE.lock"
REBOOT_INHIBIT_FILENAME = "PRE_REBOOT_CHECKPOINT.lock"


@contextlib.contextmanager
def queue_mutation_gate(
    state_dir: Path,
    *,
    exclusive: bool,
    reject_reboot_inhibit: bool,
) -> Iterator[TextIO]:
    """Hold a stable queue-mutation lock for the entire caller operation.

    Checkpoint creation takes the exclusive side.  Short-lived administrative
    writers take the shared side and, after acquiring it, reject the persistent
    reboot inhibit.  This closes the check-then-BEGIN race around SQLite.
    """

    state_dir = state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    gate_path = state_dir / MUTATION_GATE_FILENAME
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(gate_path, flags, 0o600)
    handle = os.fdopen(descriptor, "r+", encoding="utf-8")
    try:
        gate_stat = os.fstat(handle.fileno())
        if (
            gate_stat.st_uid != os.getuid()
            or not stat.S_ISREG(gate_stat.st_mode)
            or stat.S_IMODE(gate_stat.st_mode) != 0o600
        ):
            raise RuntimeError(f"unsafe local-Qwen mutation gate: {gate_path}")
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(handle.fileno(), operation | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            role = "checkpoint" if exclusive else "administrative writer"
            raise RuntimeError(f"local-Qwen mutation gate is busy for {role}: {gate_path}") from exc
        try:
            path_stat = gate_path.stat()
        except FileNotFoundError as exc:
            raise RuntimeError("local-Qwen mutation gate path disappeared after locking") from exc
        if (path_stat.st_dev, path_stat.st_ino) != (gate_stat.st_dev, gate_stat.st_ino):
            raise RuntimeError("local-Qwen mutation gate path was replaced after locking")
        if reject_reboot_inhibit and (state_dir / REBOOT_INHIBIT_FILENAME).exists():
            raise RuntimeError(
                "pre-reboot checkpoint inhibits local-Qwen queue/cache mutation: "
                f"{state_dir / REBOOT_INHIBIT_FILENAME}"
            )
        yield handle
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
