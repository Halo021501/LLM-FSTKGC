import importlib.util
import json
import os
import tempfile
import unittest

import torch

from src.data import HistoryIndex
from src.llm_cache import LLMEvidenceCache, target_blind_query_key
from src.model import NineFuseTKG
from src.stlp import EntityMapper, TargetBlindQuery, build_query_metadata, build_stlp_prompt, parse_and_map_response
from src.train import choose_causal_support, group_by_relation


class AlterEgoV5LLMInvariantTests(unittest.TestCase):
    def test_causal_support_uses_bounded_recent_pool_before_subject_tie_break(self):
        rows = [(0, 0, 99, 1)] + [(subject, 0, subject + 100, 1) for subject in range(1, 40)]
        selected = choose_causal_support(group_by_relation(rows), (0, 0, 7, 2), shot=1)
        self.assertEqual(len(selected), 1)
        self.assertNotEqual(selected[0][0], 0)

    def test_empty_graph_support_is_a_label_free_structural_sentinel(self):
        self.assertEqual(choose_causal_support({}, (3, 2, 7, 5), shot=1), [(3, 2, 3, 4)])

    def test_unmapped_diagnostic_is_not_named_as_a_factuality_judgment(self):
        payload = json.dumps(
            {
                "candidates": [
                    {"entity_name": "known", "confidence": 0.8},
                    {"entity_name": "not-in-vocabulary", "confidence": 0.7},
                ]
            }
        )
        _, diagnostics = parse_and_map_response(payload, EntityMapper({"known": 0}), [])
        self.assertEqual(diagnostics["unmapped_rate"], 0.5)
        self.assertEqual(
            set(diagnostics),
            {"returned_candidates", "mapped_candidates", "mapping_rate", "unmapped_rate"},
        )

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
            "prompt_template_version": "stlp-aliyun-qwen-realtime-v1",
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
            "prompt_template_version": "stlp-aliyun-qwen-realtime-v1",
        }
        record = {
            "schema_version": 1,
            "query_key": target_blind_query_key(query),
            "query": query,
            "prompt_hash": "a" * 64,
            "provider": "aliyun_qwen_realtime",
            "model": "qwen-flash",
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
