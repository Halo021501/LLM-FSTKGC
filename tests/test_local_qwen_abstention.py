import contextlib
import io
import json
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import stlp_generate_candidates as generator
from scripts import dynamic_local_qwen_pool as dynamic_pool
from src.llm_cache import LLMEvidenceCache, PROHIBITED_QUERY_FIELDS, target_blind_query_key


class LocalQwenAbstentionOfflineTests(unittest.TestCase):
    """Regression tests for local generation; network access is always forbidden."""

    def setUp(self):
        self._network_guard = mock.patch.object(
            socket,
            "create_connection",
            side_effect=AssertionError("offline local-Qwen test attempted network access"),
        )
        self._network_guard.start()
        self.addCleanup(self._network_guard.stop)

    @staticmethod
    def _dataset(root: Path) -> Path:
        data_dir = root / "tiny_tkg"
        data_dir.mkdir()
        (data_dir / "stat.txt").write_text("4\t1\t0\n", encoding="utf-8")
        (data_dir / "entity2id.txt").write_text(
            "Public_Known\t0\n"
            "Hidden_Target_SENTINEL\t1\n"
            "Past_Candidate\t2\n"
            "Other_Candidate\t3\n",
            encoding="utf-8",
        )
        (data_dir / "relation2id.txt").write_text(
            "Public_Relation\t0\n", encoding="utf-8"
        )
        (data_dir / "train.txt").write_text("0\t0\t2\t1\n", encoding="utf-8")
        (data_dir / "valid.txt").write_text("3\t0\t2\t2\n", encoding="utf-8")
        (data_dir / "test.txt").write_text("0\t0\t1\t3\n", encoding="utf-8")
        return data_dir

    @staticmethod
    def _args(data_dir: Path, output: Path):
        return generator.build_parser().parse_args(
            [
                "--data-dir",
                str(data_dir),
                "--split",
                "test",
                "--output",
                str(output),
                "--shot",
                "5",
                "--provider",
                "local_qwen",
                "--max-tokens",
                "512",
                "--retry-max-tokens",
                "768",
                "--max-retries",
                "1",
                "--timeout",
                "360",
                "--resume",
                "--num-shards",
                "2",
                "--shard-id",
                "0",
                "--progress-every",
                "100",
            ]
        )

    def _run_failure_case(self, error: Exception, expected_code: str) -> None:
        class FailingClient:
            model = "Qwen2.5-7B-Instruct-AWQ"

            def __init__(self):
                self.max_tokens = []

            def complete_json(self, _prompt, *, max_tokens):
                self.max_tokens.append(max_tokens)
                raise error

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = self._dataset(root)
            output = root / "part.jsonl"
            args = self._args(data_dir, output)
            client = FailingClient()
            fixed_provenance = {
                "model_alias": client.model,
                "model_revision": "offline-test-revision",
            }
            with (
                mock.patch.object(
                    generator.LocalQwenClient, "from_environment", return_value=client
                ),
                mock.patch.object(
                    generator, "provider_provenance", return_value=fixed_provenance
                ),
                mock.patch.object(generator.time, "sleep", return_value=None),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                generator.run(args)

            self.assertEqual(client.max_tokens, [512, 768])
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["candidates"], [])
            self.assertEqual(row["query_key"], target_blind_query_key(row["query"]))
            self.assertTrue(PROHIBITED_QUERY_FIELDS.isdisjoint(row["query"]))
            self.assertNotIn("Hidden_Target_SENTINEL", json.dumps(row, ensure_ascii=False))
            self.assertEqual(
                row["provider_abstention"],
                {
                    "schema_version": 1,
                    "policy": generator.LOCAL_QWEN_ABSTENTION_POLICY,
                    "code": expected_code,
                    "attempts": 2,
                    "error_type": type(error).__name__,
                },
            )
            self.assertEqual(row["diagnostics"]["provider_abstention"], 1.0)
            self.assertEqual(row["diagnostics"]["provider_abstention_attempts"], 2.0)
            self.assertEqual(
                row["diagnostics"][f"provider_abstention_{expected_code}"], 1.0
            )
            self.assertEqual(
                row["token_usage"],
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

            metadata_path = Path(str(output) + ".meta.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["decoding"],
                {
                    "max_candidates": 10,
                    "max_tokens": 512,
                    "retry_max_tokens": 768,
                    "max_retries": 1,
                },
            )
            cache = LLMEvidenceCache(
                str(output),
                expected_shot=5,
                expected_history_protocol="standard_rolling_history",
                expected_split="test",
                expected_dataset_fingerprint=metadata["dataset_fingerprint"],
                require_generation_metadata=True,
            )
            self.assertEqual(len(cache.records), 1)

            original_cache = output.read_bytes()
            original_metadata = metadata_path.read_bytes()
            resumed_client = mock.Mock(model=client.model)
            resumed_client.complete_json.side_effect = AssertionError(
                "resume called the provider for an already recorded abstention"
            )
            with (
                mock.patch.object(
                    generator.LocalQwenClient,
                    "from_environment",
                    return_value=resumed_client,
                ),
                mock.patch.object(
                    generator, "provider_provenance", return_value=fixed_provenance
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                generator.run(args)
            resumed_client.complete_json.assert_not_called()
            self.assertEqual(output.read_bytes(), original_cache)
            self.assertEqual(metadata_path.read_bytes(), original_metadata)

    def test_exhausted_length_invalid_json_and_transport_become_resumable_abstentions(self):
        cases = [
            (
                RuntimeError(
                    "Local Qwen returned invalid JSON content "
                    "(finish_reason='length', completion_tokens=768, max_tokens=768)"
                ),
                "length_exhausted",
            ),
            (
                RuntimeError(
                    "Local Qwen returned invalid JSON content "
                    "(finish_reason='stop', completion_tokens=21, max_tokens=768)"
                ),
                "invalid_json_exhausted",
            ),
            (TimeoutError("timed out"), "transport_exhausted"),
        ]
        for error, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                self._run_failure_case(error, expected_code)

    def test_unrelated_programming_error_is_not_silently_abstained(self):
        self.assertIsNone(generator.local_qwen_abstention(ValueError("bug"), attempts=2))

    def test_failed_task_requeue_preserves_attempts_and_claims_next_audit_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                data_dir=root / "data",
                cache_dir=root / "cache",
                split="test",
                shots=[5],
                num_shards=1,
                workers_per_gpu=1,
                max_tokens=512,
                retry_max_tokens=768,
                max_retries=1,
                task_max_attempts=3,
                task_retry_backoff_seconds=0.0,
                history_protocol="standard_rolling_history",
                seed=42,
                model="Qwen2.5-7B-Instruct-AWQ",
            )
            database = root / "queue.sqlite3"
            dynamic_pool.initialize_database(database, args)
            connection = dynamic_pool.connect_database(database)
            try:
                task_id = None
                for attempt in range(1, 4):
                    task = dynamic_pool.claim_task(
                        connection,
                        "gpu2-slot0",
                        2,
                        "http://127.0.0.1:8102/v1",
                    )
                    self.assertIsNotNone(task)
                    task_id = int(task["id"])
                    self.assertEqual(int(task["attempts"]), attempt)
                    outcome = dynamic_pool.finish_task(
                        connection,
                        task_id,
                        False,
                        attempt,
                        1,
                        "generator exited",
                        args.task_max_attempts,
                        args.task_retry_backoff_seconds,
                    )
                self.assertEqual(outcome["status"], "failed")

                dry_run = dynamic_pool.requeue_failed_tasks(connection, apply=False)
                self.assertEqual(dry_run[0]["previous_attempts"], 3)
                self.assertEqual(dry_run[0]["next_attempt"], 4)
                unchanged = connection.execute(
                    "SELECT status,attempts,records FROM tasks WHERE id=?", (task_id,)
                ).fetchone()
                self.assertEqual(
                    (unchanged["status"], unchanged["attempts"], unchanged["records"]),
                    ("failed", 3, 3),
                )

                applied = dynamic_pool.requeue_failed_tasks(
                    connection, [task_id], apply=True
                )
                self.assertEqual(applied, dry_run)
                requeued = connection.execute(
                    "SELECT status,attempts,records FROM tasks WHERE id=?", (task_id,)
                ).fetchone()
                self.assertEqual(
                    (requeued["status"], requeued["attempts"], requeued["records"]),
                    ("pending", 3, 3),
                )

                fourth = dynamic_pool.claim_task(
                    connection,
                    "gpu4-slot0",
                    4,
                    "http://127.0.0.1:8104/v1",
                )
                self.assertEqual(int(fourth["attempts"]), 4)
                dynamic_pool.finish_task(
                    connection,
                    task_id,
                    True,
                    52,
                    0,
                    None,
                    args.task_max_attempts,
                    args.task_retry_backoff_seconds,
                )
                attempts = list(
                    connection.execute(
                        "SELECT attempt,status FROM task_attempts "
                        "WHERE task_id=? ORDER BY attempt",
                        (task_id,),
                    )
                )
                self.assertEqual(
                    [(row["attempt"], row["status"]) for row in attempts],
                    [
                        (1, "retry_scheduled"),
                        (2, "retry_scheduled"),
                        (3, "failed"),
                        (4, "complete"),
                    ],
                )
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
