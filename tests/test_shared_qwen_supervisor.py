import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.attach_shared_qwen_worker import (
    active_worker_conflict,
    mark_sidecar_workers_stopped,
)
from scripts.opportunistic_shared_qwen_supervisor import (
    ResourceSnapshot,
    admission_decision,
    build_parser,
    resource_snapshot,
    retire_stale_worker_rows_if_safe,
)


class SharedSidecarCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.state_dir = Path(self.temporary.name)
        connection = sqlite3.connect(str(self.state_dir / "queue.sqlite3"))
        connection.execute(
            """
            CREATE TABLE workers (
                worker_id TEXT PRIMARY KEY,
                gpu_id INTEGER NOT NULL,
                slot INTEGER NOT NULL,
                endpoint TEXT NOT NULL,
                managed_server INTEGER NOT NULL,
                status TEXT NOT NULL,
                task_id INTEGER,
                updated_at_utc TEXT NOT NULL,
                error TEXT
            )
            """
        )
        rows = [
            ("gpu5-slot0", 5, 0, "http://127.0.0.1:8105/v1", 0, "ready", None),
            ("gpu5-slot1", 5, 1, "http://127.0.0.1:8105/v1", 0, "retry_wait", None),
            ("gpu5-slot2", 5, 2, "http://127.0.0.1:8105/v1", 0, "running", 9),
            ("gpu5-slot3", 5, 3, "http://127.0.0.1:9999/v1", 0, "ready", None),
            ("gpu4-slot0", 4, 0, "http://127.0.0.1:8104/v1", 0, "ready", None),
        ]
        connection.executemany(
            """
            INSERT INTO workers(
                worker_id,gpu_id,slot,endpoint,managed_server,status,task_id,
                updated_at_utc,error
            ) VALUES(?,?,?,?,?,?,?,'old',NULL)
            """,
            rows,
        )
        connection.commit()
        connection.close()

    def tearDown(self):
        self.temporary.cleanup()

    def test_cleanup_retires_only_exact_non_running_sidecar_rows(self):
        changed, remaining = mark_sidecar_workers_stopped(
            self.state_dir,
            5,
            "http://127.0.0.1:8105/v1",
            4,
        )
        self.assertEqual(changed, 2)
        self.assertEqual(remaining, [("gpu5-slot2", "running", 9)])

        connection = sqlite3.connect(str(self.state_dir / "queue.sqlite3"))
        observed = dict(
            connection.execute("SELECT worker_id,status FROM workers ORDER BY worker_id")
        )
        connection.close()
        self.assertEqual(observed["gpu5-slot0"], "stopped")
        self.assertEqual(observed["gpu5-slot1"], "stopped")
        self.assertEqual(observed["gpu5-slot2"], "running")
        self.assertEqual(observed["gpu5-slot3"], "ready")
        self.assertEqual(observed["gpu4-slot0"], "ready")

    def test_stopped_rows_no_longer_block_reconnect(self):
        connection = sqlite3.connect(str(self.state_dir / "queue.sqlite3"))
        connection.execute("DELETE FROM workers WHERE worker_id IN ('gpu5-slot2','gpu5-slot3')")
        connection.commit()
        connection.close()
        self.assertEqual(len(active_worker_conflict(self.state_dir, 5)), 2)
        mark_sidecar_workers_stopped(
            self.state_dir,
            5,
            "http://127.0.0.1:8105/v1",
            4,
        )
        self.assertEqual(active_worker_conflict(self.state_dir, 5), [])

    def test_orphan_task_free_conflicts_are_atomically_audited_and_stopped(self):
        connection = sqlite3.connect(str(self.state_dir / "queue.sqlite3"))
        connection.execute("DELETE FROM workers WHERE worker_id='gpu5-slot2'")
        connection.commit()
        connection.close()
        event_log = self.state_dir / "supervisor.jsonl"
        with mock.patch(
            "scripts.opportunistic_shared_qwen_supervisor.shared_stack_absent",
            return_value=(True, "no_live_shared_stack"),
        ):
            recovered, status = retire_stale_worker_rows_if_safe(
                self.state_dir, 5, 8105, event_log
            )
        self.assertEqual(status, "stale_worker_rows_stopped")
        self.assertEqual(len(recovered), 3)

        connection = sqlite3.connect(str(self.state_dir / "queue.sqlite3"))
        observed = list(
            connection.execute(
                "SELECT status,task_id,error FROM workers WHERE gpu_id=5 ORDER BY worker_id"
            )
        )
        connection.close()
        self.assertTrue(all(row[0] == "stopped" for row in observed))
        self.assertTrue(all(row[1] is None for row in observed))
        self.assertTrue(all("no live sidecar" in row[2] for row in observed))
        self.assertIn("stale_worker_rows_stopped", event_log.read_text(encoding="utf-8"))

    def test_running_or_task_owned_conflict_is_never_recovered(self):
        event_log = self.state_dir / "supervisor.jsonl"
        with mock.patch(
            "scripts.opportunistic_shared_qwen_supervisor.shared_stack_absent",
            return_value=(True, "no_live_shared_stack"),
        ):
            recovered, status = retire_stale_worker_rows_if_safe(
                self.state_dir, 5, 8105, event_log
            )
        self.assertEqual(recovered, [])
        self.assertEqual(status, "active_worker_owns_task")
        self.assertFalse(event_log.exists())

        connection = sqlite3.connect(str(self.state_dir / "queue.sqlite3"))
        connection.execute("DELETE FROM workers WHERE worker_id='gpu5-slot2'")
        connection.execute("UPDATE workers SET task_id=77 WHERE worker_id='gpu5-slot0'")
        connection.commit()
        connection.close()
        with mock.patch(
            "scripts.opportunistic_shared_qwen_supervisor.shared_stack_absent",
            return_value=(True, "no_live_shared_stack"),
        ):
            recovered, status = retire_stale_worker_rows_if_safe(
                self.state_dir, 5, 8105, event_log
            )
        self.assertEqual(recovered, [])
        self.assertEqual(status, "active_worker_owns_task")

    def test_live_stack_blocks_stale_row_recovery(self):
        event_log = self.state_dir / "supervisor.jsonl"
        with mock.patch(
            "scripts.opportunistic_shared_qwen_supervisor.shared_stack_absent",
            return_value=(False, "live_sidecar_pid_file"),
        ):
            recovered, status = retire_stale_worker_rows_if_safe(
                self.state_dir, 5, 8105, event_log
            )
        self.assertEqual(recovered, [])
        self.assertEqual(status, "live_sidecar_pid_file")
        self.assertFalse(event_log.exists())


class SharedAdmissionTests(unittest.TestCase):
    def decision(self, snapshot, **overrides):
        values = {
            "pending_tasks": 10,
            "active_worker_count": 0,
            "server_running": False,
            "start_gpu_free_mib": 11000,
            "start_host_available_mib": 10000,
        }
        values.update(overrides)
        return admission_decision(snapshot, **values)

    def test_shared_card_starts_only_with_both_start_floors(self):
        snapshot = ResourceSnapshot(10000, 11000, 5380, (1234,))
        self.assertEqual(self.decision(snapshot), "start_shared_stack")
        self.assertEqual(
            self.decision(ResourceSnapshot(9999, 11000, 5380, (1234,))),
            "wait_host_start_headroom",
        )
        self.assertEqual(
            self.decision(ResourceSnapshot(10000, 10999, 5381, (1234,))),
            "wait_gpu_start_headroom",
        )

    def test_idle_card_is_left_to_authoritative_controller(self):
        snapshot = ResourceSnapshot(20000, 15000, 1000, ())
        self.assertEqual(self.decision(snapshot), "defer_idle_gpu_to_controller")

    def test_existing_worker_or_server_is_never_duplicated(self):
        snapshot = ResourceSnapshot(20000, 12000, 4000, (1234,))
        self.assertEqual(
            self.decision(snapshot, active_worker_count=1),
            "wait_gpu_already_attached",
        )
        self.assertEqual(
            self.decision(snapshot, server_running=True),
            "wait_existing_server_ownership_unknown",
        )

    def test_start_floors_can_be_deliberately_lowered_by_environment(self):
        with mock.patch.dict(
            "os.environ",
            {
                "QWEN_OPPORTUNISTIC_START_GPU_FREE_MIB": "9500",
                "QWEN_OPPORTUNISTIC_START_HOST_AVAILABLE_MIB": "8000",
            },
        ):
            args = build_parser().parse_args(
                ["--state-dir", "/tmp/formal-state", "--gpu-id", "5"]
            )
        self.assertEqual(args.start_gpu_free_mib, 9500)
        self.assertEqual(args.start_host_available_mib, 8000)

    def test_resource_snapshot_queries_only_requested_physical_gpu(self):
        outputs = [
            "5, GPU-5, 5564, 10518\n",
            "4142991\n4144601\n",
        ]
        with (
            mock.patch.object(
                Path,
                "read_text",
                return_value="MemTotal: 65536000 kB\nMemAvailable: 36700160 kB\n",
            ),
            mock.patch(
                "scripts.opportunistic_shared_qwen_supervisor.subprocess.check_output",
                side_effect=outputs,
            ) as check_output,
        ):
            snapshot = resource_snapshot(5)

        self.assertEqual(
            snapshot,
            ResourceSnapshot(
                host_available_mib=35840,
                gpu_free_mib=10518,
                gpu_used_mib=5564,
                compute_pids=(4142991, 4144601),
            ),
        )
        self.assertEqual(check_output.call_count, 2)
        for call in check_output.call_args_list:
            command = call.args[0]
            self.assertEqual(command[:3], ["nvidia-smi", "-i", "5"])
        self.assertIn(
            "--query-compute-apps=pid",
            check_output.call_args_list[1].args[0],
        )


if __name__ == "__main__":
    unittest.main()
