from __future__ import annotations

import bisect
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


Quad = Tuple[int, int, int, int]
RawQuad = Tuple[str, str, str, str]


def _split_path(data_dir: str, split: str) -> str:
    candidates = [
        os.path.join(data_dir, split),
        os.path.join(data_dir, split + ".txt"),
    ]
    if split == "valid":
        candidates.extend(
            [
                os.path.join(data_dir, "dev"),
                os.path.join(data_dir, "dev.txt"),
            ]
        )
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Cannot find split `{split}` under {data_dir}")


def _read_raw_split(path: str) -> List[RawQuad]:
    rows: List[RawQuad] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 4:
                continue
            rows.append((parts[0], parts[1], parts[2], parts[3]))
    return rows


def _is_int_token(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


def _read_stat(data_dir: str) -> Optional[Tuple[int, int, int]]:
    path = os.path.join(data_dir, "stat.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        parts = handle.readline().strip().split()
    if len(parts) < 2 or not all(_is_int_token(x) for x in parts[:2]):
        return None
    third = int(parts[2]) if len(parts) > 2 and _is_int_token(parts[2]) else 0
    return int(parts[0]), int(parts[1]), third


def _time_sort_key(value: str):
    if _is_int_token(value):
        return (0, int(value))
    return (1, value)


def _appearance_mapping(values: Iterable[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
    return mapping


@dataclass
class TemporalKG:
    train: List[Quad]
    valid: List[Quad]
    test: List[Quad]
    num_entities: int
    num_relations: int
    num_times: int
    entity2id: Dict[str, int]
    relation2id: Dict[str, int]
    time2id: Dict[str, int]
    id2entity: List[str]
    id2relation: List[str]
    id2time: List[str]

    @property
    def train_aug(self) -> List[Quad]:
        return add_inverse(self.train, self.num_relations)

    @property
    def valid_aug(self) -> List[Quad]:
        return add_inverse(self.valid, self.num_relations)

    @property
    def test_aug(self) -> List[Quad]:
        return add_inverse(self.test, self.num_relations)


def load_temporal_kg(data_dir: str, max_train: Optional[int] = None) -> TemporalKG:
    """Load common TKGC datasets with either id-based or text-based quadruples.

    Supported rows are `s r o t` and longer variants where extra columns are ignored.
    Timestamps are compacted into chronological ids, while entity/relation ids are
    preserved when the source is numeric.
    """

    raw_train = _read_raw_split(_split_path(data_dir, "train"))
    raw_valid = _read_raw_split(_split_path(data_dir, "valid"))
    raw_test = _read_raw_split(_split_path(data_dir, "test"))
    if max_train is not None and max_train > 0:
        raw_train = raw_train[:max_train]

    all_rows = raw_train + raw_valid + raw_test
    stat = _read_stat(data_dir)
    numeric_entities = all(_is_int_token(row[0]) and _is_int_token(row[2]) for row in all_rows)
    numeric_relations = all(_is_int_token(row[1]) for row in all_rows)

    if numeric_entities:
        max_ent = max(max(int(row[0]), int(row[2])) for row in all_rows) + 1
        num_entities = max(stat[0], max_ent) if stat else max_ent
        entity2id = {str(i): i for i in range(num_entities)}
    else:
        entity2id = _appearance_mapping([row[0] for row in all_rows] + [row[2] for row in all_rows])
        num_entities = len(entity2id)

    if numeric_relations:
        max_rel = max(int(row[1]) for row in all_rows) + 1
        num_relations = max(stat[1], max_rel) if stat else max_rel
        relation2id = {str(i): i for i in range(num_relations)}
    else:
        relation2id = _appearance_mapping(row[1] for row in all_rows)
        num_relations = len(relation2id)

    raw_times = sorted({row[3] for row in all_rows}, key=_time_sort_key)
    time2id = {value: idx for idx, value in enumerate(raw_times)}

    def convert(rows: Sequence[RawQuad]) -> List[Quad]:
        converted: List[Quad] = []
        for s, r, o, t in rows:
            sid = int(s) if numeric_entities else entity2id[s]
            oid = int(o) if numeric_entities else entity2id[o]
            rid = int(r) if numeric_relations else relation2id[r]
            converted.append((sid, rid, oid, time2id[t]))
        converted.sort(key=lambda x: (x[3], x[0], x[1], x[2]))
        return converted

    id2entity = [None] * num_entities
    for name, idx in entity2id.items():
        if idx < num_entities:
            id2entity[idx] = name
    id2entity = [str(i) if value is None else value for i, value in enumerate(id2entity)]

    id2relation = [None] * num_relations
    for name, idx in relation2id.items():
        if idx < num_relations:
            id2relation[idx] = name
    id2relation = [str(i) if value is None else value for i, value in enumerate(id2relation)]

    return TemporalKG(
        train=convert(raw_train),
        valid=convert(raw_valid),
        test=convert(raw_test),
        num_entities=num_entities,
        num_relations=num_relations,
        num_times=len(time2id),
        entity2id=entity2id,
        relation2id=relation2id,
        time2id=time2id,
        id2entity=id2entity,
        id2relation=id2relation,
        id2time=raw_times,
    )


def add_inverse(quads: Sequence[Quad], num_relations: int) -> List[Quad]:
    augmented = list(quads)
    augmented.extend((o, r + num_relations, s, t) for s, r, o, t in quads)
    return augmented


class FewShotRelationSampler:
    def __init__(
        self,
        quads: Sequence[Quad],
        shot: int,
        query: int,
        seed: int = 42,
    ) -> None:
        self.by_relation: Dict[int, List[Quad]] = defaultdict(list)
        for quad in quads:
            self.by_relation[quad[1]].append(quad)
        self.relations = [r for r, rows in self.by_relation.items() if len(rows) >= 2]
        if not self.relations:
            raise ValueError("Need at least one relation with two examples for few-shot sampling")
        self.shot = shot
        self.query = query
        self.rng = random.Random(seed)

    def sample(self) -> Tuple[List[Quad], List[Quad]]:
        relation = self.rng.choice(self.relations)
        rows = self.by_relation[relation]
        count = min(len(rows), max(2, self.shot + self.query))
        picked = self.rng.sample(rows, count) if len(rows) > count else list(rows)
        support_size = min(self.shot, max(1, count - 1))
        query_size = min(self.query, count - support_size)
        return picked[:support_size], picked[support_size : support_size + query_size]


class HistoryIndex:
    """Fast enough history/retrieval features for few-shot TKGC episodes."""

    def __init__(
        self,
        facts: Sequence[Quad],
        num_entities: int,
        num_relations_total: int,
        history_len: int = 8,
        rule_topk: int = 32,
        snapshot_len: int = 24,
        decay: float = 0.12,
        adaptive_decay: bool = True,
    ) -> None:
        self.facts = list(facts)
        self.num_entities = num_entities
        self.num_relations_total = num_relations_total
        self.history_len = history_len
        self.rule_topk = rule_topk
        self.snapshot_len = snapshot_len
        self.decay = decay
        self.adaptive_decay = adaptive_decay
        self.base_time_scale = max(1.0, 1.0 / max(decay, 1e-6))
        self.events_by_subject: List[List[Tuple[int, int, int]]] = [[] for _ in range(num_entities)]
        self.events_by_subject_relation: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
        self.relation_tail_prior: Dict[int, Counter] = defaultdict(Counter)
        self.relation_prior_dense = torch.zeros(num_relations_total, num_entities)
        relation_times: Dict[int, List[int]] = defaultdict(list)
        for s, r, o, t in facts:
            self.events_by_subject[s].append((t, r, o))
            self.events_by_subject_relation[(s, r)].append((t, o))
            self.relation_tail_prior[r][o] += 1
            self.relation_prior_dense[r, o] += 1.0
            relation_times[r].append(t)
        for rows in self.events_by_subject:
            rows.sort(key=lambda x: x[0])
        for rows in self.events_by_subject_relation.values():
            rows.sort(key=lambda x: x[0])
        self.relation_time_scale = self._build_relation_time_scales(relation_times)

    def clone(self) -> "HistoryIndex":
        """Return an independent index for leakage-free rolling evaluation."""
        return HistoryIndex(
            self.facts,
            self.num_entities,
            self.num_relations_total,
            history_len=self.history_len,
            rule_topk=self.rule_topk,
            snapshot_len=self.snapshot_len,
            decay=self.decay,
            adaptive_decay=self.adaptive_decay,
        )

    def add_facts(self, facts: Sequence[Quad]) -> None:
        """Append a completed snapshot after it has been evaluated.

        Callers must add a whole timestamp at once.  This guarantees that no
        fact from the current snapshot can be used to predict another fact in
        that same snapshot.
        """
        for s, r, o, t in facts:
            subject_rows = self.events_by_subject[s]
            relation_rows = self.events_by_subject_relation[(s, r)]
            if subject_rows and t < subject_rows[-1][0]:
                raise ValueError("rolling history facts must be chronological")
            if relation_rows and t < relation_rows[-1][0]:
                raise ValueError("rolling relation history facts must be chronological")
            subject_rows.append((t, r, o))
            relation_rows.append((t, o))
            self.relation_tail_prior[r][o] += 1
            self.relation_prior_dense[r, o] += 1.0
            self.facts.append((s, r, o, t))

    @staticmethod
    def _median(values: Sequence[int]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[mid])
        return 0.5 * float(ordered[mid - 1] + ordered[mid])

    def _build_relation_time_scales(self, relation_times: Dict[int, List[int]]) -> List[float]:
        relation_gaps: Dict[int, List[int]] = defaultdict(list)
        for (_, r), rows in self.events_by_subject_relation.items():
            previous_t = None
            for t, _ in rows:
                if previous_t is not None and t > previous_t:
                    relation_gaps[r].append(t - previous_t)
                previous_t = t

        for r, times in relation_times.items():
            previous_t = None
            for t in sorted(times):
                if previous_t is not None and t > previous_t:
                    relation_gaps[r].append(t - previous_t)
                previous_t = t

        max_scale = self.base_time_scale * 8.0
        scales = [self.base_time_scale for _ in range(self.num_relations_total)]
        for r in range(self.num_relations_total):
            median_gap = self._median(relation_gaps.get(r, []))
            if median_gap > 0:
                scales[r] = min(max(1.0, median_gap), max_scale)
        return scales

    def _decay_weight(self, relation: int, age: int) -> float:
        if age <= 0:
            return 1.0
        if not self.adaptive_decay:
            return math.exp(-self.decay * age)
        scale = self.relation_time_scale[relation] if relation < len(self.relation_time_scale) else self.base_time_scale
        return math.exp(-float(age) / max(1.0, scale))

    def _normalized_relation_scale(self, relation: int) -> float:
        scale = self.relation_time_scale[relation] if relation < len(self.relation_time_scale) else self.base_time_scale
        return float(scale / (scale + self.base_time_scale))

    def _recent_subject_events(self, s: int, t: int) -> List[Tuple[int, int, int]]:
        rows = self.events_by_subject[s]
        idx = bisect.bisect_left(rows, (t, -1, -1))
        return rows[max(0, idx - self.history_len) : idx]

    def _snapshot_neighborhood(self, s: int, r: int, t: int) -> List[Tuple[int, int, int, int]]:
        """Return a causal, query-conditioned two-hop snapshot neighborhood.

        Facts at the query timestamp are excluded by ``bisect_left``.  Direct
        edges use role 0 and sampled second-hop edges use role 1.  Selection is
        score-based, but the returned rows are chronological for recurrent
        snapshot evolution in the model.
        """
        scored: Dict[Tuple[int, int, int, int], float] = {}
        direct = self._recent_subject_events(s, t)
        for ht, hr, mid in direct:
            direct_score = self._decay_weight(hr, max(0, t - ht))
            if hr == r:
                direct_score *= 1.5
            key = (mid, hr, ht, 0)
            scored[key] = max(scored.get(key, 0.0), direct_score)

            rows = self.events_by_subject[mid]
            idx = bisect.bisect_left(rows, (t, -1, -1))
            for nt, nr, tail in rows[max(0, idx - 4) : idx]:
                second_score = 0.5 * self._decay_weight(nr, max(0, t - nt))
                if nr == r:
                    second_score *= 1.25
                second_key = (tail, nr, nt, 1)
                scored[second_key] = max(scored.get(second_key, 0.0), second_score)

        selected = sorted(scored.items(), key=lambda item: item[1], reverse=True)[: self.snapshot_len]
        return sorted((row for row, _ in selected), key=lambda row: (row[2], row[3], row[1], row[0]))

    @staticmethod
    def _logits_from_counts(counts: torch.Tensor) -> torch.Tensor:
        return torch.log1p(counts)

    def build(self, queries: Sequence[Quad]) -> Dict[str, torch.Tensor]:
        batch = len(queries)
        hist_entities = torch.zeros(batch, self.history_len, dtype=torch.long)
        hist_relations = torch.zeros(batch, self.history_len, dtype=torch.long)
        hist_times = torch.zeros(batch, self.history_len, dtype=torch.long)
        hist_mask = torch.zeros(batch, self.history_len, dtype=torch.bool)
        snapshot_entities = torch.zeros(batch, self.snapshot_len, dtype=torch.long)
        snapshot_relations = torch.zeros(batch, self.snapshot_len, dtype=torch.long)
        snapshot_times = torch.zeros(batch, self.snapshot_len, dtype=torch.long)
        snapshot_roles = torch.zeros(batch, self.snapshot_len, dtype=torch.long)
        snapshot_mask = torch.zeros(batch, self.snapshot_len, dtype=torch.bool)
        copy_counts = torch.zeros(batch, self.num_entities)
        rel_copy_counts = torch.zeros(batch, self.num_entities)
        query_relations = torch.tensor([row[1] for row in queries], dtype=torch.long)
        relation_prior_counts = self.relation_prior_dense.index_select(0, query_relations).clone()
        rule_counts = torch.zeros(batch, self.num_entities)
        rule_candidates = torch.zeros(batch, self.rule_topk, dtype=torch.long)
        rule_r1 = torch.zeros(batch, self.rule_topk, dtype=torch.long)
        rule_r2 = torch.zeros(batch, self.rule_topk, dtype=torch.long)
        rule_dt1 = torch.zeros(batch, self.rule_topk)
        rule_dt2 = torch.zeros(batch, self.rule_topk)
        rule_type = torch.zeros(batch, self.rule_topk, dtype=torch.long)
        rule_prior = torch.zeros(batch, self.rule_topk)
        rule_mask = torch.zeros(batch, self.rule_topk, dtype=torch.bool)
        rule_confidence = torch.zeros(batch)
        rule_struct_ratio = torch.zeros(batch)
        history_freshness = torch.zeros(batch)
        history_density = torch.zeros(batch)
        relation_time_scale = torch.zeros(batch)
        rel_copy_strength = torch.zeros(batch)
        in_rel_history = torch.zeros(batch)
        in_history = torch.zeros(batch)

        for i, (s, r, o, t) in enumerate(queries):
            recent = self._recent_subject_events(s, t)
            snapshot_rows = self._snapshot_neighborhood(s, r, t)
            for j, (neighbor, edge_relation, edge_time, edge_role) in enumerate(snapshot_rows):
                snapshot_entities[i, j] = neighbor
                snapshot_relations[i, j] = edge_relation
                snapshot_times[i, j] = edge_time
                snapshot_roles[i, j] = edge_role
                snapshot_mask[i, j] = True
            start = self.history_len - len(recent)
            history_density[i] = len(recent) / max(1, self.history_len)
            relation_time_scale[i] = self._normalized_relation_scale(r)
            candidate_scores: Dict[int, float] = defaultdict(float)
            candidate_feats: Dict[int, Tuple[int, int, float, float, int]] = {}
            structural_sum = 0.0
            backoff_sum = 0.0
            max_struct_score = 0.0
            for j, (ht, hr, ho) in enumerate(recent, start=start):
                age = max(0, t - ht)
                weight = self._decay_weight(hr, age)
                history_freshness[i] = max(float(history_freshness[i].item()), float(weight))
                hist_entities[i, j] = ho
                hist_relations[i, j] = hr
                hist_times[i, j] = ht
                hist_mask[i, j] = True
                copy_counts[i, ho] += weight
                if ho == o:
                    in_history[i] = 1.0

                score = weight
                candidate_scores[ho] += score
                structural_sum += score
                max_struct_score = max(max_struct_score, score)
                candidate_feats.setdefault(ho, (hr, r, float(age), 0.0, 1))

                # One-hop temporal path: s --hr--> mid, then mid --r--> tail.
                chain_rows = self.events_by_subject_relation.get((ho, r), [])
                chain_idx = bisect.bisect_left(chain_rows, (t, -1))
                for ct, tail in chain_rows[max(0, chain_idx - 6) : chain_idx]:
                    age2 = max(0, t - ct)
                    score = 0.35 * self._decay_weight(r, age2)
                    rule_counts[i, tail] += score
                    candidate_scores[tail] += score
                    structural_sum += score
                    max_struct_score = max(max_struct_score, score)
                    candidate_feats.setdefault(tail, (hr, r, float(age), float(age2), 2))

            same_rel = self.events_by_subject_relation.get((s, r), [])
            same_idx = bisect.bisect_left(same_rel, (t, -1))
            for ht, tail in same_rel[max(0, same_idx - self.history_len) : same_idx]:
                age = max(0, t - ht)
                rel_copy_score = self._decay_weight(r, age)
                rel_copy_counts[i, tail] += rel_copy_score
                rel_copy_strength[i] = max(float(rel_copy_strength[i].item()), float(rel_copy_score))
                if tail == o:
                    in_rel_history[i] = 1.0
                    in_history[i] = 1.0

                score = 1.2 * rel_copy_score
                rule_counts[i, tail] += score
                candidate_scores[tail] += score
                structural_sum += score
                max_struct_score = max(max_struct_score, score)
                candidate_feats.setdefault(tail, (r, r, float(age), 0.0, 3))

            for tail, cnt in self.relation_tail_prior.get(r, Counter()).most_common(32):
                score = 0.03 * cnt
                rule_counts[i, tail] += score
                candidate_scores[tail] += score
                backoff_sum += score
                candidate_feats.setdefault(tail, (r, r, 0.0, 0.0, 4))

            ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[: self.rule_topk]
            total_rule_score = structural_sum + backoff_sum
            if total_rule_score > 0:
                struct_ratio = structural_sum / total_rule_score
                rule_struct_ratio[i] = struct_ratio
                rule_confidence[i] = min(1.0, struct_ratio * math.log1p(max_struct_score + structural_sum))
            for k, (tail, score) in enumerate(ranked):
                fr1, fr2, fdt1, fdt2, ftype = candidate_feats[tail]
                rule_candidates[i, k] = tail
                rule_r1[i, k] = fr1
                rule_r2[i, k] = fr2
                rule_dt1[i, k] = fdt1
                rule_dt2[i, k] = fdt2
                rule_type[i, k] = ftype
                rule_prior[i, k] = math.log1p(score)
                rule_mask[i, k] = True

        return {
            "hist_entities": hist_entities,
            "hist_relations": hist_relations,
            "hist_times": hist_times,
            "hist_mask": hist_mask,
            "snapshot_entities": snapshot_entities,
            "snapshot_relations": snapshot_relations,
            "snapshot_times": snapshot_times,
            "snapshot_roles": snapshot_roles,
            "snapshot_mask": snapshot_mask,
            "copy_logits": self._logits_from_counts(copy_counts),
            "rel_copy_logits": self._logits_from_counts(rel_copy_counts),
            "relation_prior_logits": self._logits_from_counts(relation_prior_counts),
            "rule_logits": self._logits_from_counts(rule_counts),
            "rule_candidates": rule_candidates,
            "rule_r1": rule_r1,
            "rule_r2": rule_r2,
            "rule_dt1": rule_dt1,
            "rule_dt2": rule_dt2,
            "rule_type": rule_type,
            "rule_prior": rule_prior,
            "rule_mask": rule_mask,
            "rule_confidence": rule_confidence,
            "rule_struct_ratio": rule_struct_ratio,
            "history_freshness": history_freshness,
            "history_density": history_density,
            "relation_time_scale": relation_time_scale,
            "rel_copy_strength": rel_copy_strength,
            "in_rel_history": in_rel_history,
            "in_history": in_history,
        }


def build_filter_dict(quads: Sequence[Quad]) -> Dict[Tuple[int, int, int], set]:
    answers: Dict[Tuple[int, int, int], set] = defaultdict(set)
    for s, r, o, t in quads:
        answers[(s, r, t)].add(o)
    return answers


def tensorize(quads: Sequence[Quad], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(quads, dtype=torch.long, device=device)
