import importlib.util
import json
import os
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from src.data import HistoryIndex
from src.llm_cache import LLMEvidenceCache, target_blind_query_key
from src.model import NineFuseTKG
from src.stlp import (
    DeepSeekClient,
    LocalQwenClient,
    TargetBlindQuery,
    build_query_metadata,
    build_stlp_prompt,
)
from scripts import dynamic_local_qwen_pool as dynamic_pool


class AlterEgoV5LLMInvariantTests(unittest.TestCase):
    @staticmethod
    def _pool_args(directory):
        root = Path(directory)
        return SimpleNamespace(
            data_dir=root / "data",
            cache_dir=root / "cache",
            split="test",
            shots=[5, 10],
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

    @staticmethod
    def _candidate_generator_module():
        path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "scripts", "stlp_generate_candidates.py")
        )
        spec = importlib.util.spec_from_file_location("stlp_generate_candidates_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def _model(self, llm_mode="off", **kwargs):
        return NineFuseTKG(
            8,
            4,
            6,
            dim=32,
            history_len=3,
            channels=4,
            dropout=0.0,
            alterego_candidate_k=6,
            alterego_tournament_rank=8,
            llm_mode=llm_mode,
            llm_max_candidates=3,
            **kwargs,
        )

    def _features(self, target=1):
        return HistoryIndex(
            [(0, 0, 2, 0), (0, 1, 3, 1), (3, 0, 4, 1), (0, 0, 5, 2)],
            num_entities=8,
            num_relations_total=4,
            history_len=3,
        ).build([(0, 0, target, 3)])

    def _llm_features(self, target=1):
        features = self._features(target)
        features.update(
            {
                "llm_candidate_ids": torch.tensor([[6, 7, 0]]),
                "llm_candidate_mask": torch.tensor([[True, True, False]]),
                "llm_confidence": torch.tensor([[0.9, 0.6, 0.0]]),
                "llm_mapping_score": torch.tensor([[1.0, 0.95, 0.0]]),
                "llm_template_agreement": torch.tensor([[0.8, 0.5, 0.0]]),
                "llm_temporal_score": torch.tensor([[0.7, 0.4, 0.0]]),
                "llm_rank_prior": torch.tensor([[1.0, 0.5, 0.0]]),
                "llm_cache_hit": torch.tensor([True]),
            }
        )
        return features

    def test_off_mode_ignores_llm_tensors_and_keeps_four_experts(self):
        torch.manual_seed(11)
        model = self._model(llm_mode="off")
        model.eval()
        query = torch.tensor([[0, 0, 1, 3]])
        support = torch.tensor([[[0, 0, 2, 0], [0, 0, 5, 2]]])
        with torch.no_grad():
            plain, plain_aux = model(query, support, self._features())
            injected, injected_aux = model(query, support, self._llm_features())
        self.assertTrue(torch.equal(plain, injected))
        self.assertTrue(torch.equal(plain_aux["expert_logps"], injected_aux["expert_logps"]))
        self.assertEqual(injected_aux["expert_logps"].shape[1], 4)
        self.assertEqual(injected_aux["llm_bonus"].abs().sum().item(), 0.0)

    def test_candidate_mode_is_not_a_score_expert(self):
        model = self._model(llm_mode="candidate")
        model.eval()
        query = torch.tensor([[0, 0, 1, 3]])
        support = torch.tensor([[[0, 0, 2, 0], [0, 0, 5, 2]]])
        with torch.no_grad():
            _, aux = model(query, support, self._llm_features())
        self.assertEqual(aux["expert_logps"].shape[1], 4)
        self.assertEqual(aux["llm_bonus"].abs().sum().item(), 0.0)
        tournament_ids = set(aux["alterego_candidate_ids"][0].tolist())
        self.assertTrue({6, 7}.issubset(tournament_ids))

    def test_score_mode_has_bounded_sparse_bonus(self):
        model = self._model(llm_mode="score", llm_max_delta=0.35)
        model.eval()
        model.set_alterego_runtime_enabled(False)
        query = torch.tensor([[0, 0, 1, 3]])
        support = torch.tensor([[[0, 0, 2, 0], [0, 0, 5, 2]]])
        features = self._llm_features()
        with torch.no_grad():
            model.set_llm_runtime_mode("off")
            base, base_aux = model(query, support, features)
            model.set_llm_runtime_mode("score")
            adjusted, adjusted_aux = model(query, support, features)
        bonus = adjusted_aux["llm_bonus"]
        self.assertGreater(bonus[0, 6].item(), 0.0)
        self.assertGreater(bonus[0, 7].item(), 0.0)
        self.assertLessEqual(bonus.max().item(), 0.35 + 1e-7)
        self.assertEqual(torch.count_nonzero(bonus).item(), 2)
        self.assertFalse(torch.equal(base, adjusted))
        self.assertTrue(torch.equal(base_aux["expert_logps"], adjusted_aux["expert_logps"]))

    def test_confidence_ablation_keeps_candidates_and_other_features(self):
        model = self._model(llm_mode="rationale", llm_disable_confidence=True)
        prepared = model._prepare_llm_features(self._llm_features(), 1, torch.device("cpu"))
        candidate_ids, candidate_mask, confidence, mapping, template, temporal, rank_prior = prepared
        self.assertEqual(candidate_ids.tolist(), [[6, 7, 0]])
        self.assertEqual(candidate_mask.tolist(), [[True, True, False]])
        self.assertEqual(torch.count_nonzero(confidence).item(), 0)
        self.assertGreater(torch.count_nonzero(mapping).item(), 0)
        self.assertGreater(torch.count_nonzero(template).item(), 0)
        self.assertGreater(torch.count_nonzero(temporal).item(), 0)
        self.assertGreater(torch.count_nonzero(rank_prior).item(), 0)

    def test_cache_lookup_and_key_are_target_blind(self):
        query = {
            "dataset_fingerprint": "dataset",
            "split": "test",
            "direction": "tail",
            "known_entity_id": 0,
            "oriented_relation_id": 0,
            "timestamp": 3,
            "shot": 5,
            "seed": 42,
            "history_protocol": "standard_rolling_history",
            "support_digest": "support",
            "history_digest": "history",
            "prompt_template_version": "stlp-deepseek-v1",
        }
        record = {
            "schema_version": 1,
            "query_key": target_blind_query_key(query),
            "query": query,
            "prompt_hash": "a" * 64,
            "candidates": [
                {
                    "mapped_entity_id": 6,
                    "confidence": 0.9,
                    "mapping_score": 1.0,
                    "template_agreement": 0.8,
                    "temporal_score": 0.7,
                }
            ],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", encoding="utf-8", delete=False) as handle:
            handle.write(json.dumps(record) + "\n")
            path = handle.name
        try:
            cache = LLMEvidenceCache(path, max_candidates=3, expected_shot=5, expected_split="test")
            first = cache.augment_features([(0, 0, 1, 3)], self._features(1))
            second = cache.augment_features([(0, 0, 7, 3)], self._features(7))
            for key in LLMEvidenceCache.tensor_fields:
                self.assertTrue(torch.equal(first[key], second[key]), key)
        finally:
            os.unlink(path)

        leaking = dict(query)
        leaking["target_entity_id"] = 1
        with self.assertRaises(ValueError):
            target_blind_query_key(leaking)

    def test_formal_cache_requires_schema_v2_generation_provenance(self):
        query = {
            "dataset_fingerprint": "dataset",
            "split": "test",
            "direction": "tail",
            "known_entity_id": 0,
            "oriented_relation_id": 0,
            "timestamp": 3,
            "shot": 5,
            "seed": 42,
            "history_protocol": "standard_rolling_history",
            "support_digest": "support",
            "history_digest": "history",
            "prompt_template_version": "stlp-qwen2.5-local-v1",
        }
        record = {
            "schema_version": 1,
            "query_key": target_blind_query_key(query),
            "query": query,
            "prompt_hash": "a" * 64,
            "provider": "local_qwen",
            "model": "Qwen2.5-7B-Instruct-AWQ",
            "candidates": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "cache.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            with self.assertRaisesRegex(FileNotFoundError, "requires cache generation metadata"):
                LLMEvidenceCache(path, require_generation_metadata=True)
            metadata = {
                "schema_version": 2,
                "shot": 5,
                "history_protocol": "standard_rolling_history",
                "split": "test",
                "dataset_fingerprint": "dataset",
                "query_key_excludes_target": True,
                "provider_provenance": {"model_revision": "fixed"},
                "generation_audit": {"started_at_utc": "fixed"},
            }
            with open(path + ".meta.json", "w", encoding="utf-8") as handle:
                json.dump(metadata, handle)
            cache = LLMEvidenceCache(
                path,
                expected_shot=5,
                expected_history_protocol="standard_rolling_history",
                expected_split="test",
                expected_dataset_fingerprint="dataset",
                require_generation_metadata=True,
            )
            self.assertEqual(cache.metadata()["generation_metadata"]["schema_version"], 2)
            self.assertEqual(len(cache.metadata()["generation_metadata_sha256"]), 64)

    def test_prompt_contract_has_no_target_argument_or_serialized_label(self):
        query = TargetBlindQuery("test", "tail", 0, 0, 3)
        support = [(0, 0, 2, 1)]
        metadata = build_query_metadata(
            query,
            shot=5,
            seed=42,
            history_protocol="standard_rolling_history",
            support=support,
            history=[(0, 1, 3, 2)],
            dataset_fingerprint="dataset",
        )
        prompt = build_stlp_prompt(
            query,
            support,
            [(0, 1, 3, 2)],
            ["known", "hidden_target", "past_candidate", "history_candidate"],
            ["relation", "history_relation"],
            2,
        )
        self.assertNotIn("target", {key.lower() for key in metadata})
        self.assertNotIn("hidden_target", prompt)
        self.assertEqual(target_blind_query_key(metadata), target_blind_query_key(dict(metadata)))

    def test_missing_deepseek_key_fails_before_network(self):
        with self.assertRaises(ValueError):
            DeepSeekClient("")

    def test_local_qwen_client_is_loopback_only_and_openai_compatible(self):
        with self.assertRaises(ValueError):
            LocalQwenClient(base_url="https://example.com/v1")

        response_body = {
            "id": "local-smoke",
            "model": "Qwen2.5-7B-Instruct-AWQ",
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"candidates":[{"e":"Iran","c":0.8,'
                            '"r":"Recent statements support continuity","t":0.7}]}'
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return json.dumps(response_body).encode("utf-8")

        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        fake_opener = mock.Mock()
        fake_opener.open.side_effect = fake_urlopen
        with mock.patch("urllib.request.build_opener", return_value=fake_opener) as build_opener:
            client = LocalQwenClient(timeout_seconds=7)
            result = client.complete_json("target-blind prompt", max_tokens=128)
        proxy_handler = build_opener.call_args.args[0]
        self.assertEqual(proxy_handler.proxies, {})
        self.assertEqual(captured["url"], "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(captured["payload"]["temperature"], 0.0)
        self.assertEqual(captured["payload"]["seed"], 0)
        self.assertEqual(
            captured["payload"]["guided_decoding_backend"], "lm-format-enforcer"
        )
        self.assertEqual(
            captured["payload"]["guided_json"]["required"], ["candidates"]
        )
        wire_item = captured["payload"]["guided_json"]["properties"]["candidates"]["items"]
        self.assertEqual(wire_item["required"], ["e", "c", "r", "t"])
        self.assertNotIn("response_format", captured["payload"])
        self.assertNotIn("thinking", captured["payload"])
        self.assertEqual(captured["timeout"], 7)
        self.assertEqual(
            json.loads(result["content"]),
            {
                "candidates": [
                    {
                        "entity_name": "Iran",
                        "confidence": 0.8,
                        "temporal_rationale": "Recent statements support continuity",
                        "temporal_consistency": 0.7,
                    }
                ]
            },
        )

        response_body["choices"][0]["message"]["content"] = '{"candidates":[],"unexpected":true}'
        with self.assertRaisesRegex(RuntimeError, "unexpected top-level keys"):
            client.complete_json("target-blind prompt", max_tokens=128)

    def test_resume_requires_exact_invariant_metadata(self):
        generator = self._candidate_generator_module()
        invariant = {
            "schema_version": 2,
            "shot": 5,
            "provider": "local_qwen",
            "provider_provenance": {"model_revision": "fixed-revision"},
        }
        audit = {
            "started_at_utc": "2026-08-08T00:00:00+00:00",
            "command_argv": ["python", "generate.py"],
            "hostname": "test-host",
            "physical_gpu_id": "2",
        }
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "cache.jsonl")
            generator.ensure_cache_metadata(output, invariant, audit, resume=True)
            generator.ensure_cache_metadata(output, invariant, audit, resume=True)
            changed = dict(invariant)
            changed["shot"] = 10
            with self.assertRaisesRegex(ValueError, "differing fields:.*shot"):
                generator.ensure_cache_metadata(output, changed, audit, resume=True)
            with open(output, "w", encoding="utf-8") as handle:
                handle.write("{}\n")
            os.unlink(output + ".meta.json")
            with self.assertRaisesRegex(FileNotFoundError, "without metadata"):
                generator.ensure_cache_metadata(output, invariant, audit, resume=True)

    def test_dynamic_pool_retries_failed_shard_and_audits_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "queue.sqlite3"
            args = self._pool_args(directory)
            self.assertEqual(dynamic_pool.initialize_database(database, args), 0)
            connection = dynamic_pool.connect_database(database)
            try:
                first = dynamic_pool.claim_task(
                    connection, "gpu3-slot0", 3, "http://127.0.0.1:8103/v1"
                )
                self.assertIsNotNone(first)
                outcome = dynamic_pool.finish_task(
                    connection,
                    int(first["id"]),
                    False,
                    4,
                    1,
                    "invalid JSON",
                    args.task_max_attempts,
                    args.task_retry_backoff_seconds,
                )
                self.assertEqual(outcome["status"], "pending")
                self.assertEqual(outcome["attempt"], 1)

                second = dynamic_pool.claim_task(
                    connection, "gpu5-slot0", 5, "http://127.0.0.1:8105/v1"
                )
                self.assertEqual(int(second["id"]), int(first["id"]))
                outcome = dynamic_pool.finish_task(
                    connection,
                    int(second["id"]),
                    True,
                    52,
                    0,
                    None,
                    args.task_max_attempts,
                    args.task_retry_backoff_seconds,
                )
                self.assertEqual(outcome["status"], "complete")
                attempts = list(
                    connection.execute(
                        "SELECT attempt,status,records_after FROM task_attempts "
                        "WHERE task_id=? ORDER BY attempt",
                        (int(first["id"]),),
                    )
                )
                self.assertEqual(
                    [(row["attempt"], row["status"], row["records_after"]) for row in attempts],
                    [(1, "retry_scheduled", 4), (2, "complete", 52)],
                )
            finally:
                connection.close()

    def test_dynamic_pool_recovers_interrupted_attempt_without_losing_records(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "queue.sqlite3"
            args = self._pool_args(directory)
            dynamic_pool.initialize_database(database, args)
            connection = dynamic_pool.connect_database(database)
            task = dynamic_pool.claim_task(
                connection, "gpu3-slot0", 3, "http://127.0.0.1:8103/v1"
            )
            connection.close()
            output = Path(str(task["output_path"]))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('{"durable":true}\n', encoding="utf-8")

            self.assertEqual(dynamic_pool.initialize_database(database, args), 1)
            connection = dynamic_pool.connect_database(database)
            try:
                recovered = connection.execute(
                    "SELECT status,attempts,records FROM tasks WHERE id=?", (int(task["id"]),)
                ).fetchone()
                attempt = connection.execute(
                    "SELECT status,records_after FROM task_attempts "
                    "WHERE task_id=? AND attempt=1",
                    (int(task["id"]),),
                ).fetchone()
                self.assertEqual(
                    (recovered["status"], recovered["attempts"], recovered["records"]),
                    ("pending", 1, 1),
                )
                self.assertEqual((attempt["status"], attempt["records_after"]), ("interrupted", 1))
            finally:
                connection.close()

    def test_dynamic_pool_starts_independent_gpu_servers_in_parallel(self):
        pool = object.__new__(dynamic_pool.DynamicPool)
        pool.args = SimpleNamespace(server_retry_cooldown_seconds=120.0)
        pool.startup_retry_after = {}
        pool.log_event = mock.Mock()
        barrier = threading.Barrier(3)

        def fake_start(gpu_id, shared):
            barrier.wait(timeout=1.0)
            return dynamic_pool.Server(
                gpu_id=gpu_id,
                endpoint=f"http://127.0.0.1:{8100 + gpu_id}/v1",
                state_dir=None,
                managed=True,
            )

        pool.start_managed_server = fake_start
        servers = pool.start_managed_servers_parallel([(3, True), (4, True), (5, True)])
        self.assertEqual([server.gpu_id for server in servers], [3, 4, 5])

    def test_dynamic_pool_keeps_rejected_initial_gpus_under_idle_monitoring(self):
        pool = object.__new__(dynamic_pool.DynamicPool)
        pool.args = SimpleNamespace(
            additional_min_free_mib=12000,
            additional_max_utilization=5,
            idle_checks=1,
        )
        pool.monitored_gpus = [3, 5]
        pool.servers = {}
        pool.startup_retry_after = {}
        pool.idle_confirmations = {}
        pool.log_event = mock.Mock()
        captured = []
        pool.start_managed_servers_parallel = lambda requests: captured.extend(requests) or []
        idle_state = {
            "compute_pids": [],
            "memory_free_mib": 16000,
            "utilization_gpu": 0,
        }
        pool.maybe_add_idle_servers({3: idle_state, 5: idle_state})
        self.assertEqual(captured, [(3, False), (5, False)])

    def test_dynamic_pool_gpu_query_isolates_failed_physical_card(self):
        pool = object.__new__(dynamic_pool.DynamicPool)
        pool.monitored_gpus = [2, 4, 6]
        pool.log_event = mock.Mock()

        def fake_check_output(command, **_kwargs):
            self.assertEqual(command[0], "nvidia-smi")
            self.assertEqual(command[1], "-i")
            gpu_id = int(command[2])
            if gpu_id == 6:
                raise dynamic_pool.subprocess.CalledProcessError(255, command)
            if any(item.startswith("--query-gpu=") for item in command):
                return f"{gpu_id}, GPU-{gpu_id}, 7000, 9000, 12\n"
            if gpu_id == 4:
                raise dynamic_pool.subprocess.CalledProcessError(255, command)
            self.assertIn("--query-compute-apps=pid,used_gpu_memory", command)
            return "1234, 7000\n"

        with mock.patch.object(
            dynamic_pool.subprocess,
            "check_output",
            side_effect=fake_check_output,
        ) as check_output:
            states = pool.query_gpu_state()

        self.assertEqual(set(states), {2})
        self.assertEqual(states[2]["uuid"], "GPU-2")
        self.assertEqual(states[2]["compute_pids"], [1234])
        self.assertEqual(check_output.call_count, 5)
        failures = [call.kwargs for call in pool.log_event.call_args_list]
        self.assertEqual([item["gpu_id"] for item in failures], [4, 6])
        self.assertTrue(all(call.args[0] == "gpu_query_failed" for call in pool.log_event.call_args_list))

    def test_dynamic_pool_transient_endpoint_failure_does_not_duplicate_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._pool_args(directory)
            args.shots = [5]
            args.state_dir = root / "state"
            args.model_dir = root / "model"
            args.python_bin = root / "python"
            args.request_timeout = 1.0
            args.progress_every = 1
            args.state_dir.mkdir(parents=True)
            database = args.state_dir / "queue.sqlite3"
            self.assertEqual(dynamic_pool.initialize_database(database, args), 0)

            pool = object.__new__(dynamic_pool.DynamicPool)
            pool.args = args
            pool.database_path = database
            pool.log_event = mock.Mock()
            server = dynamic_pool.Server(
                gpu_id=2,
                endpoint="http://127.0.0.1:8102/v1",
                state_dir=None,
                managed=True,
            )

            def fake_run(command, **_kwargs):
                output = Path(command[command.index("--output") + 1])
                output.write_text('{"record":1}\n', encoding="utf-8")
                return dynamic_pool.subprocess.CompletedProcess(command, 0)

            dynamic_pool.STOP_EVENT.clear()
            with (
                mock.patch.object(
                    dynamic_pool,
                    "endpoint_ready",
                    side_effect=[False, True, True],
                ) as endpoint_ready,
                mock.patch.object(
                    dynamic_pool.subprocess,
                    "run",
                    side_effect=fake_run,
                ) as generator_run,
                mock.patch.object(
                    dynamic_pool,
                    "WORKER_ENDPOINT_RETRY_BACKOFF_SECONDS",
                    0.0,
                ),
            ):
                pool.worker_loop(server, 0)

            connection = dynamic_pool.connect_database(database)
            try:
                task = connection.execute("SELECT * FROM tasks").fetchone()
                attempts = list(connection.execute("SELECT * FROM task_attempts"))
                worker = connection.execute(
                    "SELECT status,error FROM workers WHERE worker_id='gpu2-slot0'"
                ).fetchone()
            finally:
                connection.close()
                dynamic_pool.STOP_EVENT.clear()

            self.assertEqual(endpoint_ready.call_count, 3)
            self.assertEqual(generator_run.call_count, 1)
            self.assertEqual((task["status"], task["attempts"], task["records"]), ("complete", 1, 1))
            self.assertEqual(len(attempts), 1)
            self.assertEqual((attempts[0]["attempt"], attempts[0]["status"]), (1, "complete"))
            self.assertEqual((worker["status"], worker["error"]), ("idle", None))
            self.assertEqual(
                [call.args[0] for call in pool.log_event.call_args_list].count(
                    "worker_endpoint_check_retry"
                ),
                1,
            )
            self.assertNotIn(
                "worker_endpoint_unhealthy",
                [call.args[0] for call in pool.log_event.call_args_list],
            )

    def test_dynamic_pool_requires_three_endpoint_failures_before_worker_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._pool_args(directory)
            args.shots = [5]
            args.state_dir = root / "state"
            args.state_dir.mkdir(parents=True)
            database = args.state_dir / "queue.sqlite3"
            self.assertEqual(dynamic_pool.initialize_database(database, args), 0)

            pool = object.__new__(dynamic_pool.DynamicPool)
            pool.args = args
            pool.database_path = database
            pool.log_event = mock.Mock()
            server = dynamic_pool.Server(
                gpu_id=2,
                endpoint="http://127.0.0.1:8102/v1",
                state_dir=None,
                managed=True,
            )

            dynamic_pool.STOP_EVENT.clear()
            with (
                mock.patch.object(dynamic_pool, "endpoint_ready", return_value=False) as ready,
                mock.patch.object(
                    dynamic_pool,
                    "WORKER_ENDPOINT_RETRY_BACKOFF_SECONDS",
                    0.0,
                ),
                mock.patch.object(dynamic_pool.subprocess, "run") as generator_run,
            ):
                pool.worker_loop(server, 0)

            connection = dynamic_pool.connect_database(database)
            try:
                task = connection.execute("SELECT * FROM tasks").fetchone()
                attempt_count = connection.execute(
                    "SELECT COUNT(*) FROM task_attempts"
                ).fetchone()[0]
                worker = connection.execute(
                    "SELECT status,error FROM workers WHERE worker_id='gpu2-slot0'"
                ).fetchone()
            finally:
                connection.close()
                dynamic_pool.STOP_EVENT.clear()

            self.assertEqual(ready.call_count, 3)
            generator_run.assert_not_called()
            self.assertEqual((task["status"], task["attempts"], task["records"]), ("pending", 0, 0))
            self.assertEqual(attempt_count, 0)
            self.assertEqual(worker["status"], "failed")
            self.assertIn("3 consecutive checks", worker["error"])
            events = [call.args[0] for call in pool.log_event.call_args_list]
            self.assertEqual(events.count("worker_endpoint_check_retry"), 2)
            self.assertEqual(events.count("worker_endpoint_unhealthy"), 1)

    def test_local_cache_provenance_has_exact_model_and_runtime_identity(self):
        generator = self._candidate_generator_module()
        release = json.loads((Path(__file__).resolve().parents[1] / "LLM_EXTENSION_PROVENANCE.json").read_text())
        model_dir = Path(os.environ.get("LOCAL_QWEN_MODEL_DIR", release["local_model_directory"]))
        if not model_dir.is_absolute():
            model_dir = Path(__file__).resolve().parents[1] / model_dir
        if not (model_dir / "MODEL_MANIFEST.sha256").is_file():
            self.skipTest("optional local-Qwen weights are not present in this package")
        client = mock.Mock(model="Qwen2.5-7B-Instruct-AWQ")
        provenance = generator.provider_provenance("local_qwen", client)
        self.assertEqual(provenance["model_revision"], "b25037543e9394b818fdfca67ab2a00ecc7dd641")
        self.assertEqual(len(provenance["model_manifest_sha256"]), 64)
        self.assertEqual(len(provenance["runtime_lock_sha256"]), 64)
        self.assertEqual(provenance["serving_profile"]["quantization_kernel"], "awq_marlin")

    def test_external_v5_reference_is_bit_exact_when_available(self):
        reference_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "NineFuseTKG_version_1.7.0alterego_v5", "src", "model.py")
        )
        if not os.path.exists(reference_path):
            self.skipTest("sibling v5 reference is not present in this package")
        spec = importlib.util.spec_from_file_location("v5_reference_model", reference_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        kwargs = dict(
            num_entities=8,
            num_relations_total=4,
            num_times=6,
            dim=32,
            history_len=3,
            channels=4,
            dropout=0.0,
            alterego_candidate_k=6,
            alterego_tournament_rank=8,
        )
        torch.manual_seed(123)
        reference = module.NineFuseTKG(**kwargs)
        torch.manual_seed(123)
        current = NineFuseTKG(**kwargs, llm_mode="off")
        for key, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, current.state_dict()[key]), key)
        reference.eval()
        current.eval()
        query = torch.tensor([[0, 0, 1, 3]])
        support = torch.tensor([[[0, 0, 2, 0], [0, 0, 5, 2]]])
        with torch.no_grad():
            expected, _ = reference(query, support, self._features())
            actual, aux = current(query, support, self._features())
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(aux["expert_logps"].shape[1], 4)


if __name__ == "__main__":
    unittest.main()
