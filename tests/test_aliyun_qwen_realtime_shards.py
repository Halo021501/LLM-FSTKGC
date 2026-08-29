import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts import merge_aliyun_qwen_realtime_shards as merge_cli
from scripts import prepare_aliyun_qwen_realtime_shards as prepare_cli
from scripts import stlp_aliyun_qwen_batch as batch_cli
from src.aliyun_qwen_batch import jsonl_bytes, sha256_bytes, write_bytes_atomic, write_json_atomic
from src.llm_cache import LLMEvidenceCache


class RealtimeShardOfflineTests(unittest.TestCase):
    @staticmethod
    def _dataset(root: Path) -> Path:
        data = root / "data"
        data.mkdir()
        (data / "stat.txt").write_text("6\t1\t0\n", encoding="utf-8")
        (data / "entity2id.txt").write_text(
            "A\t0\nB\t1\nC\t2\nD\t3\nE\t4\nF\t5\n", encoding="utf-8"
        )
        (data / "relation2id.txt").write_text("R\t0\n", encoding="utf-8")
        (data / "train.txt").write_text("0\t0\t1\t0\n", encoding="utf-8")
        (data / "valid.txt").write_text("2\t0\t3\t1\n", encoding="utf-8")
        (data / "test.txt").write_text(
            "0\t0\t2\t2\n1\t0\t3\t2\n4\t0\t5\t3\n", encoding="utf-8"
        )
        return data

    @staticmethod
    def _part_cache(source_dir: Path, output: Path) -> None:
        plan, _, indexes = batch_cli._verify_job(source_dir)
        records = [
            {
                "schema_version": 1,
                "query_key": index["custom_id"],
                "query": index["query"],
                "prompt_hash": index["prompt_hash"],
                "provider": "aliyun_qwen_realtime",
                "model": "qwen-flash",
                "response_id": f"response-{index['ordinal']}",
                "candidates": [],
                "diagnostics": {},
                "latency_ms": 10.0,
                "token_usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                },
            }
            for index in indexes
        ]
        cache_bytes = jsonl_bytes(records)
        write_bytes_atomic(output, cache_bytes)
        write_json_atomic(
            Path(str(output) + ".meta.json"),
            {
                "schema_version": 2,
                "purpose": "target-blind STLP candidate cache",
                "split": plan["split"],
                "shot": plan["shot"],
                "seed": plan["seed"],
                "history_protocol": plan["history_protocol"],
                "provider": "aliyun_qwen_realtime",
                "model": "qwen-flash",
                "provider_provenance": {
                    "provider_managed_model": True,
                    "exact_weight_revision_available": False,
                    "requested_model": "qwen-flash",
                    "resolved_models": ["qwen-flash"],
                    "region": "cn-beijing",
                    "api_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "source_request_sha256": plan["request_sha256"],
                    "source_index_sha256": plan["index_sha256"],
                    "provider_abstention_policy": "empty_evidence",
                    "provenance_file": "fixture.json",
                    "provenance_file_sha256": "0" * 64,
                    "official_documentation": {},
                },
                "dataset_fingerprint": plan["dataset_fingerprint"],
                "query_key_excludes_target": True,
                "api_called_inside_training_or_evaluation": False,
                "formal_full_split": False,
                "prompt_ablation": plan["prompt_ablation"],
                "decoding": plan["decoding"],
                "generation_audit": {
                    "request_count": len(records),
                    "successful_response_count": len(records),
                    "provider_abstention_count": 0,
                    "provider_abstention_codes": {},
                    "token_usage": {
                        "prompt_tokens": 20 * len(records),
                        "completion_tokens": 5 * len(records),
                        "total_tokens": 25 * len(records),
                    },
                    "estimated_list_price_cny": 0.01,
                    "price_verified_at": "fixture",
                    "attempt_count": len(records),
                    "retry_count": 0,
                    "avg_latency_ms": 10.0,
                    "per_request_latency_available": True,
                    "cache_sha256": sha256_bytes(cache_bytes),
                },
            },
        )

    def test_prepare_and_merge_complete_two_shards_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = self._dataset(root)
            shard_root = root / "shards"
            args = prepare_cli.build_parser().parse_args(
                [
                    "--job-root",
                    str(shard_root),
                    "--data-dir",
                    str(data),
                    "--split",
                    "test",
                    "--shot",
                    "10",
                    "--num-shards",
                    "2",
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                prepared = prepare_cli.prepare(args)
            self.assertFalse(prepared["network_called"])
            self.assertEqual(prepared["request_count"], 6)

            sources = sorted(path for path in shard_root.iterdir() if path.is_dir())
            parts = []
            for source in sources:
                part = root / f"{source.name}.jsonl"
                self._part_cache(source, part)
                parts.append(part)
            output = root / "merged.jsonl"
            merge_args = merge_cli.build_parser().parse_args(
                [
                    "--data-dir",
                    str(data),
                    "--split",
                    "test",
                    "--shot",
                    "10",
                    "--source-job-dir",
                    str(sources[0]),
                    "--source-job-dir",
                    str(sources[1]),
                    "--part-cache",
                    str(parts[0]),
                    "--part-cache",
                    str(parts[1]),
                    "--output",
                    str(output),
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = merge_cli.merge(merge_args)
            self.assertEqual(result["records"], 6)
            cache = LLMEvidenceCache(str(output), require_generation_metadata=True)
            self.assertEqual(len(cache.records), 6)
            self.assertTrue(cache.generation_metadata["formal_full_split"])
            self.assertEqual(cache.generation_metadata["generation_audit"]["num_shards"], 2)


if __name__ == "__main__":
    unittest.main()
