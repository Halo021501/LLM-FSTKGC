from __future__ import annotations

import difflib
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .data import Quad
from .llm_cache import target_blind_query_key


PROMPT_TEMPLATE_VERSION = "stlp-aliyun-qwen-realtime-v1"


def clamp01(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_id_map(path: str, expected_size: int | None = None) -> Tuple[Dict[str, int], List[str]]:
    """Read common name-to-id mappings while tolerating an optional count header."""

    name_to_id: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) == 1 and parts[0].isdigit() and not name_to_id:
                continue
            if len(parts) < 2:
                raise ValueError(f"invalid id mapping at {path}:{line_number}")
            if parts[-1].lstrip("-").isdigit():
                name = " ".join(parts[:-1]) if "\t" not in line else "\t".join(parts[:-1])
                index = int(parts[-1])
            elif parts[0].lstrip("-").isdigit():
                index = int(parts[0])
                name = " ".join(parts[1:]) if "\t" not in line else "\t".join(parts[1:])
            else:
                raise ValueError(f"mapping row has no integer id at {path}:{line_number}")
            if index < 0:
                raise ValueError(f"negative id at {path}:{line_number}")
            name_to_id[name] = index
    if not name_to_id:
        raise ValueError(f"empty id mapping: {path}")
    size = max(name_to_id.values()) + 1
    if expected_size is not None and size != expected_size:
        raise ValueError(f"mapping size mismatch for {path}: expected {expected_size}, found {size}")
    id_to_name = [str(index) for index in range(size)]
    for name, index in name_to_id.items():
        id_to_name[index] = name
    return name_to_id, id_to_name


def display_name(value: str) -> str:
    return value.replace("_", " ").strip()


def normalize_entity_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold().replace("_", " ")
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


@dataclass(frozen=True)
class MappingResult:
    raw_name: str
    mapped_entity_id: int | None
    mapped_entity_name: str | None
    mapping_score: float
    mapping_method: str


class EntityMapper:
    """Conservative exact/normalized/fuzzy mapper from LLM names to entity ids."""

    def __init__(
        self,
        name_to_id: Mapping[str, int],
        fuzzy_threshold: float = 0.90,
        fuzzy_margin: float = 0.04,
    ) -> None:
        self.name_to_id = dict(name_to_id)
        self.id_to_name = {index: name for name, index in name_to_id.items()}
        normalized: Dict[str, List[Tuple[str, int]]] = {}
        for name, index in name_to_id.items():
            normalized.setdefault(normalize_entity_name(name), []).append((name, index))
        self.normalized = normalized
        self.normalized_keys = sorted(key for key in normalized if key)
        self.fuzzy_threshold = float(fuzzy_threshold)
        self.fuzzy_margin = float(fuzzy_margin)

    def map(self, raw_name: str) -> MappingResult:
        raw_name = str(raw_name).strip()
        if raw_name in self.name_to_id:
            index = self.name_to_id[raw_name]
            return MappingResult(raw_name, index, self.id_to_name[index], 1.0, "exact")
        normalized = normalize_entity_name(raw_name)
        exact_normalized = self.normalized.get(normalized, [])
        if len(exact_normalized) == 1:
            name, index = exact_normalized[0]
            return MappingResult(raw_name, index, name, 0.98, "normalized_exact")
        if not normalized:
            return MappingResult(raw_name, None, None, 0.0, "unmapped")
        close = difflib.get_close_matches(normalized, self.normalized_keys, n=2, cutoff=self.fuzzy_threshold)
        if not close:
            return MappingResult(raw_name, None, None, 0.0, "unmapped")
        best_score = difflib.SequenceMatcher(None, normalized, close[0]).ratio()
        second_score = difflib.SequenceMatcher(None, normalized, close[1]).ratio() if len(close) > 1 else 0.0
        matches = self.normalized[close[0]]
        if len(matches) != 1 or best_score - second_score < self.fuzzy_margin:
            return MappingResult(raw_name, None, None, best_score, "ambiguous_fuzzy")
        name, index = matches[0]
        return MappingResult(raw_name, index, name, best_score, "fuzzy")


@dataclass(frozen=True)
class TargetBlindQuery:
    split: str
    direction: str
    known_entity_id: int
    oriented_relation_id: int
    timestamp: int


def support_digest(support: Sequence[Quad]) -> str:
    rows = [[int(s), int(r), int(o), int(t)] for s, r, o, t in support]
    return sha256_text(json.dumps(rows, separators=(",", ":")))


def build_query_metadata(
    query: TargetBlindQuery,
    shot: int,
    seed: int,
    history_protocol: str,
    support: Sequence[Quad],
    history: Sequence[Quad],
    dataset_fingerprint: str,
    prompt_template_version: str = PROMPT_TEMPLATE_VERSION,
) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "dataset_fingerprint": dataset_fingerprint,
        "split": query.split,
        "direction": query.direction,
        "known_entity_id": int(query.known_entity_id),
        "oriented_relation_id": int(query.oriented_relation_id),
        "timestamp": int(query.timestamp),
        "shot": int(shot),
        "seed": int(seed),
        "history_protocol": history_protocol,
        "support_digest": support_digest(support),
        "history_digest": support_digest(history),
        "prompt_template_version": prompt_template_version,
    }
    # This assertion makes accidental label injection fail at cache-build time.
    target_blind_query_key(metadata)
    return metadata


def _relation_label(oriented_relation_id: int, relation_names: Sequence[str], num_relations: int) -> str:
    base_id = oriented_relation_id % num_relations
    relation = display_name(relation_names[base_id])
    return relation if oriented_relation_id < num_relations else f"inverse of {relation}"


def _render_facts(
    facts: Sequence[Quad],
    entity_names: Sequence[str],
    relation_names: Sequence[str],
    num_relations: int,
    limit: int,
) -> str:
    if not facts:
        return "(none)"
    rendered = []
    for s, r, o, t in list(facts)[-limit:]:
        rendered.append(
            f"- time={t}: {display_name(entity_names[s])} | "
            f"{_relation_label(r, relation_names, num_relations)} | {display_name(entity_names[o])}"
        )
    return "\n".join(rendered)


def build_stlp_prompt(
    query: TargetBlindQuery,
    support: Sequence[Quad],
    history: Sequence[Quad],
    entity_names: Sequence[str],
    relation_names: Sequence[str],
    num_relations: int,
    max_candidates: int = 10,
    compact_response_keys: bool = False,
) -> str:
    """Build a prompt from public context only; no target argument exists."""

    known = display_name(entity_names[query.known_entity_id])
    relation = _relation_label(query.oriented_relation_id, relation_names, num_relations)
    prediction_role = "tail/object" if query.direction == "tail" else "head/subject"
    if compact_response_keys:
        response_contract = f"""Return one JSON object with key candidates and at most {max_candidates} objects.
Use compact fields: e=ICEWS entity name, c=confidence, r=temporal rationale, t=temporal consistency.
c and t must be numbers in [0,1]. Keep r factual and under 8 words."""
    else:
        response_contract = f"""Return one JSON object with key candidates. candidates must contain at most {max_candidates} objects.
Each candidate object must contain entity_name, confidence, temporal_rationale, and temporal_consistency.
confidence and temporal_consistency must be numbers in [0,1]. Keep temporal_rationale under 35 words."""
    return f"""You are a semantic-temporal prior for temporal knowledge graph completion.
Use only the supplied past facts. Never invent a fact from the current or future timestamp.
The entity must be an ICEWS entity name. Return JSON only.

Public query:
- known entity: {known}
- oriented relation: {relation}
- query timestamp id: {query.timestamp}
- predict: {prediction_role}

Few-shot support facts, all strictly earlier than the query:
{_render_facts(support, entity_names, relation_names, num_relations, max(1, len(support)))}

Recent causal history, all strictly earlier than the query:
{_render_facts(history, entity_names, relation_names, num_relations, 16)}

{response_contract}
Do not add markdown or any key that reveals or assumes a hidden gold answer."""


def _extract_json_object(text: str) -> Mapping[str, object]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM response contains no JSON object")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    return value


def _name_pattern_score(candidate_name: str, support_names: Sequence[str]) -> float:
    if not support_names:
        return 0.5
    candidate = str(candidate_name)
    signature = (
        int("(" in candidate or ")" in candidate),
        min(4, len(normalize_entity_name(candidate).split())),
        int(any(char.isdigit() for char in candidate)),
    )
    agreements = []
    for name in support_names:
        other = (
            int("(" in name or ")" in name),
            min(4, len(normalize_entity_name(name).split())),
            int(any(char.isdigit() for char in name)),
        )
        agreements.append(sum(a == b for a, b in zip(signature, other)) / 3.0)
    return max(agreements)


def parse_and_map_response(
    response_text: str,
    mapper: EntityMapper,
    support_candidate_names: Sequence[str],
    max_candidates: int = 10,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    value = _extract_json_object(response_text)
    raw_candidates = value.get("candidates", [])
    if not isinstance(raw_candidates, list):
        raise ValueError("LLM response candidates must be a list")
    output: List[Dict[str, object]] = []
    seen_mapped = set()
    mapped_count = 0
    for raw in raw_candidates[:max_candidates]:
        if not isinstance(raw, dict) or not str(raw.get("entity_name", "")).strip():
            continue
        mapping = mapper.map(str(raw["entity_name"]))
        if mapping.mapped_entity_id is not None:
            if mapping.mapped_entity_id in seen_mapped:
                continue
            seen_mapped.add(mapping.mapped_entity_id)
            mapped_count += 1
        rationale = str(raw.get("temporal_rationale", "")).strip()[:500]
        temporal_consistency = clamp01(raw.get("temporal_consistency"), default=0.0)
        if not rationale:
            temporal_consistency = 0.0
        output.append(
            {
                "entity_name": mapping.raw_name,
                "mapped_entity_id": mapping.mapped_entity_id,
                "mapped_entity_name": mapping.mapped_entity_name,
                "mapping_method": mapping.mapping_method,
                "confidence": clamp01(raw.get("confidence"), default=0.0),
                "mapping_score": clamp01(mapping.mapping_score),
                "template_agreement": clamp01(
                    _name_pattern_score(mapping.mapped_entity_name or mapping.raw_name, support_candidate_names)
                ),
                "temporal_score": temporal_consistency,
                "temporal_rationale": rationale,
            }
        )
    raw_count = min(len(raw_candidates), max_candidates)
    return output, {
        "returned_candidates": float(raw_count),
        "mapped_candidates": float(mapped_count),
        "mapping_rate": mapped_count / max(1, raw_count),
        "unmapped_rate": (raw_count - mapped_count) / max(1, raw_count),
    }
