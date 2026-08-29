import importlib.util
import json
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "guard_qwen_scale4_controller.py"
SPEC = importlib.util.spec_from_file_location("guard_qwen_scale4_controller", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


class GuardFixture:
    def __init__(self, root: Path, pids=(101,)):
        self.root = root
        self.project = root / "project"
        self.state = self.project / "logs" / "formal"
        self.proc = root / "proc"
        self.project.mkdir(parents=True)
        self.state.mkdir(parents=True)
        self.proc.mkdir()
        for pid in pids:
            self.add_controller(pid)
        if pids:
            (self.state / "controller.pid").write_text(f"{pids[0]}\n", encoding="utf-8")
            self.write_status(pids[0], records=40)

    def add_controller(self, pid: int, state_value: str | None = None) -> None:
        process = self.proc / str(pid)
        process.mkdir()
        (process / "cwd").symlink_to(self.project, target_is_directory=True)
        argv = [
            "/usr/bin/python",
            "scripts/dynamic_local_qwen_pool.py",
            "--state-dir",
            state_value or str(self.state),
        ]
        (process / "cmdline").write_bytes(b"\0".join(item.encode() for item in argv) + b"\0")

    def write_status(self, pid: int, records: int, state_dir: Path | None = None) -> None:
        (self.state / "status.json").write_text(
            json.dumps(
                {
                    "controller_pid": pid,
                    "state_dir": str(state_dir or self.state),
                    "records_written": records,
                    "phase": "generating",
                    "task_counts": {"pending": 10, "running": 4},
                }
            ),
            encoding="utf-8",
        )


class ControllerValidationTests(unittest.TestCase):
    def test_requires_exact_uid_cwd_command_and_state_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = GuardFixture(Path(directory))
            pid, status = guard.authoritative_controller(
                fixture.state,
                fixture.project,
                fixture.proc,
                os.getuid(),
            )
            self.assertEqual(pid, 101)
            self.assertEqual(status["records_written"], 40)
            fixture.write_status(101, 40, fixture.project / "wrong")
            with self.assertRaises(guard.ControllerValidationError):
                guard.authoritative_controller(
                    fixture.state,
                    fixture.project,
                    fixture.proc,
                    os.getuid(),
                )

    def test_refuses_two_matching_controllers(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = GuardFixture(Path(directory), pids=(101, 202))
            with self.assertRaisesRegex(guard.ControllerValidationError, "exactly one"):
                guard.authoritative_controller(
                    fixture.state,
                    fixture.project,
                    fixture.proc,
                    os.getuid(),
                )

    def test_refuses_wrong_state_dir_in_command_line(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = GuardFixture(Path(directory), pids=())
            fixture.add_controller(101, str(fixture.project / "other"))
            (fixture.state / "controller.pid").write_text("101\n", encoding="utf-8")
            fixture.write_status(101, 40)
            with self.assertRaisesRegex(guard.ControllerValidationError, "controller.pid"):
                guard.authoritative_controller(
                    fixture.state,
                    fixture.project,
                    fixture.proc,
                    os.getuid(),
                )


class DetectionTests(unittest.TestCase):
    def test_progress_stall_and_reset(self):
        tracker = guard.ProgressTracker(last_records=10, last_change_monotonic=100.0)
        self.assertIsNone(tracker.observe(10, 699.9, 600.0, True))
        fault = tracker.observe(10, 700.0, 600.0, True)
        self.assertEqual(fault.reason, "no_progress")
        self.assertIsNone(tracker.observe(11, 701.0, 600.0, True))
        self.assertIsNone(tracker.observe(11, 1400.0, 600.0, False))

    def test_log_scanner_ignores_history_then_detects_new_engine_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            log = state / "servers" / "gpu3" / "server.log"
            log.parent.mkdir(parents=True)
            log.write_text("old AsyncEngineDeadError\n", encoding="utf-8")
            cursors = guard.initial_log_cursors(state)
            self.assertIsNone(guard.scan_new_engine_failures(state, cursors))
            with log.open("a", encoding="utf-8") as handle:
                handle.write("Engine iteration timed out after 900 seconds\n")
            fault = guard.scan_new_engine_failures(state, cursors)
            self.assertEqual(fault.reason, "engine_timeout")
            self.assertIn("timed out", fault.observations["matched_line"])

    def test_stop_writes_marker_and_sends_only_sigterm_after_revalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = GuardFixture(Path(directory))
            marker = fixture.state / "SCALE4_ROLLBACK_REQUIRED.json"
            event_log = fixture.state / "guard.jsonl"
            signals = []
            pid = guard.stop_authoritative_controller(
                fixture.state,
                guard.CriticalFault("gpu_memory_pressure", {"gpu_free_mib": 1000}),
                marker,
                event_log,
                fixture.project,
                fixture.proc,
                os.getuid(),
                lambda target, sig: signals.append((target, sig)),
            )
            self.assertEqual(pid, 101)
            self.assertEqual(signals, [(101, signal.SIGTERM)])
            payload = json.loads(marker.read_text(encoding="utf-8"))
            self.assertTrue(payload["signal_sent"])
            self.assertEqual(payload["reason"], "gpu_memory_pressure")
            self.assertFalse((fixture.state / "queue.sqlite3").exists())


if __name__ == "__main__":
    unittest.main()
