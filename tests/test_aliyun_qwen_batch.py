import contextlib
import io
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import stlp_aliyun_qwen_batch as batch_cli
from src.aliyun_qwen_batch import (
    AliyunQwenBatchClient,
    DEFAULT_BATCH_MODEL,
    PROVIDER_NAME,
    jsonl_bytes,
    sha256_file,
    validate_batch_requests,
    write_bytes_atomic,
)
from src.llm_cache import LLMEvidenceCache, PROHIBITED_QUERY_FIELDS, target_blind_query_key


class AliyunQwenBatchOfflineTests(unittest.TestCase):
    """No test in this class is allowed to open a network connection."""

    def setUp(self):
        self._network_guards = [
            mock.patch(
                "urllib.request.urlopen",
                side_effect=AssertionError("offline Batch test attempted urllib network access"),
            ),
            mock.patch.object(
                socket,
                "create_connection",
                side_effect=AssertionError("offline Batch test attempted socket network access"),
            ),
        ]
        for guard in self._network_guards:
            guard.start()
            self.addCleanup(guard.stop)

    @staticmethod
    def _request(custom_id="a" * 64, *, prompt="Return JSON only.", model=DEFAULT_BATCH_MODEL):
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Return valid JSON only from causal public context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
                "enable_thinking": False,
                "temperature": 0.0,
            },
        }

    @staticmethod
    def _write_tiny_dataset(directory):
        data_dir = Path(directory) / "tiny_tkg"
        data_dir.mkdir()
        (data_dir / "stat.txt").write_text("4\t1\t0\n", encoding="utf-8")
        (data_dir / "entity2id.txt").write_text(
            "Public_Known\t0\n"
            "Hidden_Target_SENTINEL\t1\n"
            "Past_Candidate\t2\n"
            "Other_Candidate\t3\n",
            encoding="utf-8",
        )
        (data_dir / "relation2id.txt").write_text("Public_Relation\t0\n", encoding="utf-8")
        (data_dir / "train.txt").write_text("0\t0\t2\t1\n", encoding="utf-8")
        (data_dir / "valid.txt").write_text("3\t0\t2\t2\n", encoding="utf-8")
        (data_dir / "test.txt").write_text("0\t0\t1\t3\n", encoding="utf-8")
        return data_dir

    @staticmethod
    def _prepare_args(
        data_dir,
        job_dir,
        *,
        limit,
        shot=5,
        omit_support=False,
        permute_support_order=False,
        replace_entity_names=False,
    ):
        argv = [
                "prepare",
                "--job-dir",
                str(job_dir),
                "--data-dir",
                str(data_dir),
                "--split",
                "test",
                "--shot",
                str(shot),
                "--model",
                DEFAULT_BATCH_MODEL,
                "--limit",
                str(limit),
            ]
        if omit_support:
            argv.append("--omit-support")
        if permute_support_order:
            argv.append("--permute-support-order")
        if replace_entity_names:
            argv.append("--replace-entity-names")
        return batch_cli.build_parser().parse_args(argv)

    @staticmethod
    def _collect_args(data_dir, job_dir, result_file, output):
        return batch_cli.build_parser().parse_args(
            [
                "collect",
                "--job-dir",
                str(job_dir),
                "--data-dir",
                str(data_dir),
                "--result-file",
                str(result_file),
                "--output",
                str(output),
                "--allow-incomplete-cache",
            ]
        )

    @staticmethod
    def _result(custom_id, entity_name, ordinal, *, status_code=200, error=None, content=None):
        if content is None:
            content = json.dumps(
                {
                    "candidates": [
                        {
                            "entity_name": entity_name,
                            "confidence": 0.8,
                            "temporal_rationale": "Earlier causal support",
                            "temporal_consistency": 0.7,
                        }
                    ]
                }
            )
        return {
            "id": f"batch-row-{ordinal}",
            "custom_id": custom_id,
            "response": {
                "status_code": status_code,
                "request_id": f"request-{ordinal}",
                "body": {
                    "id": f"chat-{ordinal}",
                    "model": DEFAULT_BATCH_MODEL,
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": content},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20 + ordinal,
                        "completion_tokens": 5,
                        "total_tokens": 25 + ordinal,
                    },
                },
            },
            "error": error,
        }

    @staticmethod
    def _forbidden_fields(value):
        found = set()
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in PROHIBITED_QUERY_FIELDS:
                    found.add(str(key).lower())
                found.update(AliyunQwenBatchOfflineTests._forbidden_fields(child))
        elif isinstance(value, (list, tuple)):
            for child in value:
                found.update(AliyunQwenBatchOfflineTests._forbidden_fields(child))
        return found

    def test_request_jsonl_is_deterministic_offline_and_contains_no_secret(self):
        secret = "sk-offline-test-secret-must-never-reach-disk"
        row = self._request(prompt="Public query only. Return JSON.")
        first = jsonl_bytes([row])
        reordered = {
            "body": dict(reversed(list(row["body"].items()))),
            "url": row["url"],
            "method": row["method"],
            "custom_id": row["custom_id"],
        }
        second = jsonl_bytes([reordered])
        self.assertEqual(first, second)
        self.assertNotIn(secret.encode(), first)
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"DASHSCOPE_API_KEY": secret}, clear=False
        ):
            path = Path(directory) / "requests.jsonl"
            write_bytes_atomic(path, first)
            summary = validate_batch_requests(path)
            self.assertEqual(summary["model"], DEFAULT_BATCH_MODEL)
            self.assertEqual(summary["request_count"], 1)
            self.assertEqual(summary["sha256"], sha256_file(path))
            self.assertNotIn(secret, path.read_text(encoding="utf-8"))

    def test_request_validator_rejects_duplicate_ids_and_secret_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            duplicate = self._request()
            path.write_bytes(jsonl_bytes([duplicate, duplicate]))
            with self.assertRaisesRegex(ValueError, "duplicate custom_id"):
                validate_batch_requests(path)

            leaking = self._request(custom_id="b" * 64)
            leaking["body"]["api_key"] = "not-a-real-secret"
            path.write_bytes(jsonl_bytes([leaking]))
            with self.assertRaisesRegex(ValueError, "credential-like"):
                validate_batch_requests(path)

    def test_cancel_uses_the_official_batch_cancel_endpoint(self):
        client = object.__new__(AliyunQwenBatchClient)
        client._json_request = mock.Mock(return_value={"status": "cancelling"})
        result = client.cancel_batch("batch-safe_id")
        self.assertEqual(result["status"], "cancelling")
        client._json_request.assert_called_once_with(
            "POST", "batches/batch-safe_id/cancel"
        )

    def test_atomic_writer_never_overwrites_existing_artifact_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "formal-cache.jsonl"
            original = b"existing-valid-cache\n"
            path.write_bytes(original)
            with self.assertRaises(FileExistsError):
                write_bytes_atomic(path, b"partial-or-invalid\n")
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob(f".{path.name}.tmp.*")), [])

    def test_prepare_is_deterministic_target_blind_and_never_loads_credentials(self):
        secret = "sk-prepare-sentinel-must-not-be-serialized"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = self._write_tiny_dataset(root)
            job_a = root / "job_a"
            job_b = root / "job_b"
            stdout = io.StringIO()
            with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": secret}, clear=False), mock.patch.object(
                batch_cli.AliyunQwenBatchClient,
                "from_environment",
                side_effect=AssertionError("offline prepare tried to construct an API client"),
            ), contextlib.redirect_stdout(stdout):
                first = batch_cli.prepare_job(self._prepare_args(data_dir, job_a, limit=1))
            with mock.patch.dict(os.environ, {}, clear=True):
                second = batch_cli.prepare_job(self._prepare_args(data_dir, job_b, limit=1))

            self.assertFalse(first["network_called"])
            self.assertFalse(second["network_called"])
            self.assertNotIn(secret, stdout.getvalue())
            for filename in (
                batch_cli.REQUEST_FILENAME,
                batch_cli.INDEX_FILENAME,
                batch_cli.PLAN_FILENAME,
            ):
                bytes_a = (job_a / filename).read_bytes()
                bytes_b = (job_b / filename).read_bytes()
                self.assertEqual(bytes_a, bytes_b, filename)
                self.assertNotIn(secret.encode(), bytes_a, filename)

            requests = batch_cli.read_jsonl(job_a / batch_cli.REQUEST_FILENAME)
            indexes = batch_cli.read_jsonl(job_a / batch_cli.INDEX_FILENAME)
            plan = batch_cli.read_json(job_a / batch_cli.PLAN_FILENAME)
            self.assertEqual(len(requests), 1)
            self.assertEqual(len(indexes), 1)
            custom_id = requests[0]["custom_id"]
            self.assertEqual(custom_id, indexes[0]["query_key"])
            self.assertEqual(custom_id, target_blind_query_key(indexes[0]["query"]))
            self.assertEqual(len(custom_id), 64)
            self.assertEqual(self._forbidden_fields(requests + indexes + [plan]), set())
            serialized = jsonl_bytes(requests) + jsonl_bytes(indexes) + json.dumps(plan).encode()
            self.assertNotIn(b"Hidden_Target_SENTINEL", serialized)
            body = requests[0]["body"]
            self.assertEqual(body["model"], DEFAULT_BATCH_MODEL)
            self.assertIs(body["enable_thinking"], False)
            self.assertEqual(body["response_format"], {"type": "json_object"})
            self.assertNotIn("max_tokens", body)
            self.assertEqual(plan["provider"], PROVIDER_NAME)
            self.assertTrue(plan["query_key_excludes_target"])
            self.assertFalse(plan["staging_contains_hidden_targets"])

    def test_extended_one_and_three_shot_plans_are_target_blind_and_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = self._write_tiny_dataset(root)
            custom_ids = []
            for shot in (1, 3):
                job_dir = root / f"job_s{shot}"
                with contextlib.redirect_stdout(io.StringIO()):
                    result = batch_cli.prepare_job(
                        self._prepare_args(data_dir, job_dir, limit=1, shot=shot)
                    )
                self.assertFalse(result["network_called"])
                plan = batch_cli.read_json(job_dir / batch_cli.PLAN_FILENAME)
                request = batch_cli.read_jsonl(job_dir / batch_cli.REQUEST_FILENAME)[0]
                index = batch_cli.read_jsonl(job_dir / batch_cli.INDEX_FILENAME)[0]
                self.assertEqual(plan["shot"], shot)
                self.assertEqual(index["query"]["shot"], shot)
                self.assertEqual(request["custom_id"], target_blind_query_key(index["query"]))
                payload = (
                    (job_dir / batch_cli.REQUEST_FILENAME).read_bytes()
                    + (job_dir / batch_cli.INDEX_FILENAME).read_bytes()
                )
                self.assertNotIn(b"Hidden_Target_SENTINEL", payload)
                custom_ids.append(request["custom_id"])
            self.assertEqual(len(set(custom_ids)), 2)

    def test_no_support_is_strictly_prompt_only_for_template_agreement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = self._write_tiny_dataset(root)
            job_dir = root / "job_no_support"
            with contextlib.redirect_stdout(io.StringIO()):
                batch_cli.prepare_job(
                    self._prepare_args(
                        data_dir, job_dir, limit=1, shot=1, omit_support=True
                    )
                )
            index = batch_cli.read_jsonl(job_dir / batch_cli.INDEX_FILENAME)[0]
            request = batch_cli.read_jsonl(job_dir / batch_cli.REQUEST_FILENAME)[0]
            plan = batch_cli.read_json(job_dir / batch_cli.PLAN_FILENAME)
            prompt = request["body"]["messages"][1]["content"]
            self.assertIn(
                "Few-shot support facts, all strictly earlier than the query:\n(none)",
                prompt,
            )
            self.assertEqual(index["support_candidate_names"], ["Past_Candidate"])
            self.assertTrue(plan["prompt_ablation"]["omit_support"])
            self.assertFalse(plan["prompt_ablation"]["omit_history"])

    def test_support_order_control_changes_order_but_preserves_fact_multiset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = self._write_tiny_dataset(root)
            (data_dir / "train.txt").write_text(
                "0\t0\t2\t0\n3\t0\t2\t1\n2\t0\t3\t2\n",
                encoding="utf-8",
            )
            standard_dir = root / "standard"
            permuted_dir = root / "permuted"
            with contextlib.redirect_stdout(io.StringIO()):
                batch_cli.prepare_job(
                    self._prepare_args(data_dir, standard_dir, limit=1, shot=3)
                )
                batch_cli.prepare_job(
                    self._prepare_args(
                        data_dir,
                        permuted_dir,
                        limit=1,
                        shot=3,
                        permute_support_order=True,
                    )
                )
            standard_index = batch_cli.read_jsonl(standard_dir / batch_cli.INDEX_FILENAME)[0]
            permuted_index = batch_cli.read_jsonl(permuted_dir / batch_cli.INDEX_FILENAME)[0]
            plan = batch_cli.read_json(permuted_dir / batch_cli.PLAN_FILENAME)
            self.assertNotEqual(
                standard_index["query"]["support_digest"],
                permuted_index["query"]["support_digest"],
            )
            self.assertEqual(plan["support_order_permutation"]["eligible_queries"], 1)
            self.assertEqual(plan["support_order_permutation"]["changed_queries"], 1)
            self.assertTrue(plan["support_order_permutation"]["fact_multiset_preserved"])

    def test_entity_name_replacement_is_target_blind_and_inverse_mapped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = self._write_tiny_dataset(root)
            job_dir = root / "entity_replaced"
            with contextlib.redirect_stdout(io.StringIO()):
                batch_cli.prepare_job(
                    self._prepare_args(
                        data_dir,
                        job_dir,
                        limit=1,
                        shot=1,
                        replace_entity_names=True,
                    )
                )
            plan = batch_cli.read_json(job_dir / batch_cli.PLAN_FILENAME)
            request = batch_cli.read_jsonl(job_dir / batch_cli.REQUEST_FILENAME)[0]
            index = batch_cli.read_jsonl(job_dir / batch_cli.INDEX_FILENAME)[0]
            payload = json.dumps(request, ensure_ascii=False)
            for original_name in (
                "Public_Known",
                "Hidden_Target_SENTINEL",
                "Past_Candidate",
                "Other_Candidate",
            ):
                self.assertNotIn(original_name, payload)
            self.assertNotIn("Hidden Target SENTINEL", payload)
            self.assertTrue(plan["prompt_ablation"]["replace_entity_names"])
            self.assertFalse(
                plan["entity_name_replacement"]["original_names_sent_to_provider"]
            )
            aliases = batch_cli.deterministic_entity_placeholders(
                ["Public_Known", "Hidden_Target_SENTINEL", "Past_Candidate", "Other_Candidate"],
                seed=42,
            )
            result_file = root / "result.jsonl"
            result_file.write_bytes(
                jsonl_bytes([self._result(index["custom_id"], aliases[2], 0)])
            )
            output = root / "entity_replaced_cache.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                batch_cli.collect_job(
                    self._collect_args(data_dir, job_dir, result_file, output)
                )
            record = batch_cli.read_jsonl(output)[0]
            candidate = record["candidates"][0]
            self.assertEqual(candidate["mapped_entity_id"], 2)
            self.assertEqual(candidate["mapped_entity_name"], "Past_Candidate")
            self.assertEqual(candidate["mapping_method"], "placeholder_exact")

    def _prepared_two_query_job(self, root):
        data_dir = self._write_tiny_dataset(root)
        job_dir = root / "job"
        batch_cli.prepare_job(self._prepare_args(data_dir, job_dir, limit=2))
        indexes = batch_cli.read_jsonl(job_dir / batch_cli.INDEX_FILENAME)
        self.assertEqual(len(indexes), 2)
        return data_dir, job_dir, indexes

    def test_collect_joins_by_custom_id_not_result_order_and_loads_formal_cache(self):
        secret = "sk-collect-sentinel-must-not-be-serialized"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir, job_dir, indexes = self._prepared_two_query_job(root)
            custom_ids = [str(row["custom_id"]) for row in indexes]
            by_id = {
                custom_ids[0]: self._result(custom_ids[0], "Past_Candidate", 0),
                custom_ids[1]: self._result(custom_ids[1], "Other_Candidate", 1),
            }
            forward = root / "forward.jsonl"
            reversed_path = root / "reversed.jsonl"
            forward.write_bytes(jsonl_bytes([by_id[custom_ids[0]], by_id[custom_ids[1]]]))
            reversed_path.write_bytes(jsonl_bytes([by_id[custom_ids[1]], by_id[custom_ids[0]]]))
            output_a = root / "aliyun_cache_a.jsonl"
            output_b = root / "aliyun_cache_b.jsonl"
            with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": secret}, clear=False), mock.patch.object(
                batch_cli.AliyunQwenBatchClient,
                "from_environment",
                side_effect=AssertionError("offline collect tried to construct an API client"),
            ):
                result_a = batch_cli.collect_job(
                    self._collect_args(data_dir, job_dir, forward, output_a)
                )
                result_b = batch_cli.collect_job(
                    self._collect_args(data_dir, job_dir, reversed_path, output_b)
                )

            self.assertFalse(result_a["network_called"])
            self.assertFalse(result_b["network_called"])
            self.assertEqual(output_a.read_bytes(), output_b.read_bytes())
            records = batch_cli.read_jsonl(output_a)
            self.assertEqual([row["query_key"] for row in records], custom_ids)
            self.assertEqual(records[0]["candidates"][0]["mapped_entity_id"], 2)
            self.assertEqual(records[1]["candidates"][0]["mapped_entity_id"], 3)

            plan = batch_cli.read_json(job_dir / batch_cli.PLAN_FILENAME)
            cache = LLMEvidenceCache(
                str(output_a),
                expected_shot=5,
                expected_history_protocol="standard_rolling_history",
                expected_split="test",
                expected_dataset_fingerprint=str(plan["dataset_fingerprint"]),
                require_generation_metadata=True,
            )
            metadata = cache.metadata()["generation_metadata"]
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["provider"], PROVIDER_NAME)
            self.assertEqual(metadata["model"], DEFAULT_BATCH_MODEL)
            self.assertTrue(metadata["provider_provenance"]["provider_managed_model"])
            self.assertFalse(metadata["provider_provenance"]["exact_weight_revision_available"])
            self.assertFalse(metadata["api_called_inside_training_or_evaluation"])
            for path in (output_a, Path(str(output_a) + ".meta.json")):
                self.assertNotIn(secret, path.read_text(encoding="utf-8"))

    def test_collect_strictly_rejects_duplicate_missing_unknown_and_error_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir, job_dir, indexes = self._prepared_two_query_job(root)
            first_id, second_id = [str(row["custom_id"]) for row in indexes]
            good_first = self._result(first_id, "Past_Candidate", 0)
            good_second = self._result(second_id, "Other_Candidate", 1)
            scenarios = {
                "duplicate": [good_first, good_first, good_second],
                "missing": [good_first],
                "unknown": [good_first, self._result("f" * 64, "Other_Candidate", 9)],
                "provider_error": [
                    good_first,
                    self._result(second_id, "Other_Candidate", 1, error={"code": "failed"}),
                ],
                "non_200": [
                    good_first,
                    self._result(second_id, "Other_Candidate", 1, status_code=500),
                ],
                "invalid_json": [
                    good_first,
                    self._result(second_id, "Other_Candidate", 1, content="not-json"),
                ],
            }
            for name, rows in scenarios.items():
                with self.subTest(name=name):
                    result_path = root / f"{name}.jsonl"
                    result_path.write_bytes(jsonl_bytes(rows))
                    output = root / f"{name}_formal_cache.jsonl"
                    with self.assertRaises(ValueError):
                        batch_cli.collect_job(
                            self._collect_args(data_dir, job_dir, result_path, output)
                        )
                    self.assertFalse(output.exists())
                    self.assertFalse(Path(str(output) + ".meta.json").exists())

    def test_collect_refuses_to_overwrite_existing_cache_or_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir, job_dir, indexes = self._prepared_two_query_job(root)
            result_path = root / "results.jsonl"
            result_path.write_bytes(
                jsonl_bytes(
                    [
                        self._result(str(indexes[0]["custom_id"]), "Past_Candidate", 0),
                        self._result(str(indexes[1]["custom_id"]), "Other_Candidate", 1),
                    ]
                )
            )
            output = root / "protected_cache.jsonl"
            sentinel = b"user-owned-cache-must-remain\n"
            output.write_bytes(sentinel)
            with self.assertRaises(FileExistsError):
                batch_cli.collect_job(self._collect_args(data_dir, job_dir, result_path, output))
            self.assertEqual(output.read_bytes(), sentinel)
            self.assertFalse(Path(str(output) + ".meta.json").exists())

            output_with_protected_meta = root / "cache_with_protected_meta.jsonl"
            protected_meta = Path(str(output_with_protected_meta) + ".meta.json")
            protected_meta.write_bytes(b"user-owned-metadata-must-remain\n")
            with self.assertRaises(FileExistsError):
                batch_cli.collect_job(
                    self._collect_args(
                        data_dir,
                        job_dir,
                        result_path,
                        output_with_protected_meta,
                    )
                )
            self.assertFalse(output_with_protected_meta.exists())
            self.assertEqual(protected_meta.read_bytes(), b"user-owned-metadata-must-remain\n")


if __name__ == "__main__":
    unittest.main()
