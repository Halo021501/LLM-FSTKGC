from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import torch

from .data import Quad


SCHEMA_VERSION = 1
PROHIBITED_QUERY_FIELDS = {
    "target",
    "target_id",
    "target_entity",
    "target_entity_id",
    "gold",
    "gold_id",
    "gold_entity",
    "gold_entity_id",
    "hidden_object",
    "hidden_object_id",
    "object_id",
    "answer",
    "answer_id",
    "answer_entity_id",
    "label",
    "label_id",
}


def canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def target_blind_query_key(query: Mapping[str, object]) -> str:
    """Hash the public query context while rejecting label-like fields."""

    found = set()

    def inspect(value: object) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                lowered = str(key).lower()
                if lowered in PROHIBITED_QUERY_FIELDS:
                    found.add(lowered)
                inspect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                inspect(child)

    inspect(query)
    forbidden = sorted(found)
    if forbidden:
        raise ValueError(f"target-blind query contains prohibited fields: {forbidden}")
    required = {
        "dataset_fingerprint",
        "split",
        "direction",
        "known_entity_id",
        "oriented_relation_id",
        "timestamp",
        "shot",
        "seed",
        "history_protocol",
        "support_digest",
        "history_digest",
        "prompt_template_version",
    }
    missing = sorted(required.difference(query))
    if missing:
        raise ValueError(f"query metadata is missing required fields: {missing}")
    return hashlib.sha256(canonical_json(query).encode("utf-8")).hexdigest()


def query_locator(known_entity_id: int, oriented_relation_id: int, timestamp: int) -> Tuple[int, int, int]:
    """Runtime cache locator; deliberately has no answer/target component."""

    return int(known_entity_id), int(oriented_relation_id), int(timestamp)


def cache_file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_files_fingerprint(data_dir: str) -> str:
    """Fingerprint the exact train/valid/test bytes used for cache generation."""

    digest = hashlib.sha256()
    for split in ("train", "valid", "test"):
        candidates = [os.path.join(data_dir, split), os.path.join(data_dir, split + ".txt")]
        path = next((candidate for candidate in candidates if os.path.exists(candidate)), None)
        if path is None:
            raise FileNotFoundError(f"cannot fingerprint missing split {split} under {data_dir}")
        digest.update(split.encode("utf-8"))
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class LLMEvidenceCache:
    """Validated sparse LLM evidence loaded from an immutable JSONL cache.

    The cache is indexed by (known entity, oriented relation, timestamp).  The
    answer column is never part of lookup, cache construction metadata, or the
    tensors passed to the model.
    """

    tensor_fields = (
        "llm_candidate_ids",
        "llm_candidate_mask",
        "llm_confidence",
        "llm_mapping_score",
        "llm_template_agreement",
        "llm_temporal_score",
        "llm_rank_prior",
        "llm_cache_hit",
    )

    def __init__(
        self,
        path: str,
        max_candidates: int = 10,
        expected_shot: int | None = None,
        expected_history_protocol: str | None = None,
        expected_split: str | None = None,
        expected_dataset_fingerprint: str | None = None,
        require_generation_metadata: bool = False,
    ) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if not path:
            raise ValueError("cache path must not be empty")
        self.path = os.path.abspath(path)
        self.max_candidates = int(max_candidates)
        self.records: Dict[Tuple[int, int, int], Mapping[str, object]] = {}
        self._diagnostics = Counter()
        self.generation_metadata: Mapping[str, object] | None = None
        self.generation_metadata_sha256: str | None = None
        self.generation_metadata_path = self.path + ".meta.json"
        if os.path.exists(self.generation_metadata_path):
            with open(self.generation_metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"cache generation metadata must be a JSON object: {self.generation_metadata_path}"
                )
            if metadata.get("schema_version") not in {1, 2}:
                raise ValueError(
                    f"unsupported cache metadata schema at {self.generation_metadata_path}: "
                    f"{metadata.get('schema_version')}"
                )
            expected_fields = {
                "shot": expected_shot,
                "history_protocol": expected_history_protocol,
                "split": expected_split,
                "dataset_fingerprint": expected_dataset_fingerprint,
            }
            for field, expected in expected_fields.items():
                if expected is not None and metadata.get(field) != expected:
                    raise ValueError(
                        f"cache generation metadata {field} mismatch at "
                        f"{self.generation_metadata_path}: expected {expected}"
                    )
            if metadata.get("query_key_excludes_target") is not True:
                raise ValueError(
                    f"cache generation metadata lacks target-blind key declaration: "
                    f"{self.generation_metadata_path}"
                )
            if require_generation_metadata and metadata.get("schema_version") != 2:
                raise ValueError(
                    f"formal LLM mode requires cache metadata schema 2: {self.generation_metadata_path}"
                )
            if require_generation_metadata and (
                not isinstance(metadata.get("provider_provenance"), dict)
                or not isinstance(metadata.get("generation_audit"), dict)
            ):
                raise ValueError(
                    f"formal LLM cache lacks provider provenance or generation audit: "
                    f"{self.generation_metadata_path}"
                )
            self.generation_metadata = metadata
            self.generation_metadata_sha256 = cache_file_sha256(self.generation_metadata_path)
        elif require_generation_metadata:
            raise FileNotFoundError(
                f"formal LLM mode requires cache generation metadata: {self.generation_metadata_path}"
            )

        with open(self.path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {self.path}:{line_number}: {exc}") from exc
                self._validate_record(
                    record,
                    line_number,
                    expected_shot=expected_shot,
                    expected_history_protocol=expected_history_protocol,
                    expected_split=expected_split,
                    expected_dataset_fingerprint=expected_dataset_fingerprint,
                )
                query = record["query"]
                locator = query_locator(
                    query["known_entity_id"],
                    query["oriented_relation_id"],
                    query["timestamp"],
                )
                if locator in self.records:
                    raise ValueError(f"duplicate target-blind query locator at {self.path}:{line_number}: {locator}")
                self.records[locator] = record
                self._diagnostics["records"] += 1
                candidates = record.get("candidates", [])
                self._diagnostics["raw_candidates"] += len(candidates)
                self._diagnostics["mapped_candidates"] += sum(
                    candidate.get("mapped_entity_id") is not None for candidate in candidates
                )

        if not self.records:
            raise ValueError(f"LLM cache is empty: {self.path}")
        self.sha256 = cache_file_sha256(self.path)

    def _validate_record(
        self,
        record: MutableMapping[str, object],
        line_number: int,
        expected_shot: int | None,
        expected_history_protocol: str | None,
        expected_split: str | None,
        expected_dataset_fingerprint: str | None,
    ) -> None:
        where = f"{self.path}:{line_number}"
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported cache schema at {where}: {record.get('schema_version')}")
        query = record.get("query")
        if not isinstance(query, dict):
            raise ValueError(f"missing query object at {where}")
        computed = target_blind_query_key(query)
        if record.get("query_key") != computed:
            raise ValueError(f"query_key mismatch at {where}")
        prompt_hash = record.get("prompt_hash")
        if not isinstance(prompt_hash, str) or len(prompt_hash) != 64:
            raise ValueError(f"invalid prompt_hash at {where}")
        if expected_shot is not None and int(query["shot"]) != int(expected_shot):
            raise ValueError(f"shot mismatch at {where}: expected {expected_shot}")
        if expected_history_protocol is not None and query.get("history_protocol") != expected_history_protocol:
            raise ValueError(f"history protocol mismatch at {where}: expected {expected_history_protocol}")
        if expected_split is not None and query.get("split") != expected_split:
            raise ValueError(f"split mismatch at {where}: expected {expected_split}")
        if expected_dataset_fingerprint is not None and query.get("dataset_fingerprint") != expected_dataset_fingerprint:
            raise ValueError(f"dataset fingerprint mismatch at {where}")
        candidates = record.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"candidates must be a list at {where}")
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise ValueError(f"candidate {index} must be an object at {where}")
            mapped_id = candidate.get("mapped_entity_id")
            if mapped_id is not None and (not isinstance(mapped_id, int) or mapped_id < 0):
                raise ValueError(f"invalid mapped_entity_id for candidate {index} at {where}")
            for field in ("confidence", "mapping_score", "template_agreement", "temporal_score"):
                value = float(candidate.get(field, 0.0))
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{field} outside [0, 1] for candidate {index} at {where}")

    def metadata(self) -> Dict[str, object]:
        raw = self._diagnostics["raw_candidates"]
        mapped = self._diagnostics["mapped_candidates"]
        return {
            "path": self.path,
            "sha256": self.sha256,
            "records": self._diagnostics["records"],
            "raw_candidates": raw,
            "mapped_candidates": mapped,
            "mapping_rate": mapped / max(1, raw),
            "max_candidates": self.max_candidates,
            "lookup_contract": ["known_entity_id", "oriented_relation_id", "timestamp"],
            "target_blind": True,
            "generation_metadata_path": self.generation_metadata_path
            if self.generation_metadata is not None
            else None,
            "generation_metadata_sha256": self.generation_metadata_sha256,
            "generation_metadata": self.generation_metadata,
        }

    def augment_features(
        self,
        queries: Sequence[Quad],
        features: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Attach CPU tensors for cache hits without reading query target ids."""

        batch = len(queries)
        width = self.max_candidates
        num_entities = int(features["copy_logits"].shape[1])
        candidate_ids = torch.zeros(batch, width, dtype=torch.long)
        candidate_mask = torch.zeros(batch, width, dtype=torch.bool)
        confidence = torch.zeros(batch, width)
        mapping_score = torch.zeros(batch, width)
        template_agreement = torch.zeros(batch, width)
        temporal_score = torch.zeros(batch, width)
        rank_prior = torch.zeros(batch, width)
        cache_hit = torch.zeros(batch, dtype=torch.bool)

        for row_index, query_row in enumerate(queries):
            # Only the public columns are unpacked.  query_row[2] is never read.
            known_entity_id = int(query_row[0])
            oriented_relation_id = int(query_row[1])
            timestamp = int(query_row[3])
            record = self.records.get(query_locator(known_entity_id, oriented_relation_id, timestamp))
            if record is None:
                continue
            cache_hit[row_index] = True
            seen_ids = set()
            output_index = 0
            for source_rank, candidate in enumerate(record.get("candidates", []), start=1):
                mapped_id = candidate.get("mapped_entity_id")
                if mapped_id is None:
                    continue
                mapped_id = int(mapped_id)
                if mapped_id >= num_entities or mapped_id in seen_ids:
                    continue
                seen_ids.add(mapped_id)
                candidate_ids[row_index, output_index] = mapped_id
                candidate_mask[row_index, output_index] = True
                confidence[row_index, output_index] = float(candidate.get("confidence", 0.0))
                mapping_score[row_index, output_index] = float(candidate.get("mapping_score", 0.0))
                template_agreement[row_index, output_index] = float(candidate.get("template_agreement", 0.0))
                temporal_score[row_index, output_index] = float(candidate.get("temporal_score", 0.0))
                rank_prior[row_index, output_index] = 1.0 / float(source_rank)
                output_index += 1
                if output_index >= width:
                    break

        enriched = dict(features)
        enriched.update(
            {
                "llm_candidate_ids": candidate_ids,
                "llm_candidate_mask": candidate_mask,
                "llm_confidence": confidence,
                "llm_mapping_score": mapping_score,
                "llm_template_agreement": template_agreement,
                "llm_temporal_score": temporal_score,
                "llm_rank_prior": rank_prior,
                "llm_cache_hit": cache_hit,
            }
        )
        return enriched


def cache_coverage(cache: LLMEvidenceCache, queries: Iterable[Quad]) -> Dict[str, float]:
    total = 0
    hits = 0
    for row in queries:
        total += 1
        hits += query_locator(row[0], row[1], row[3]) in cache.records
    return {"queries": float(total), "cache_hits": float(hits), "cache_hit_rate": hits / max(1, total)}
