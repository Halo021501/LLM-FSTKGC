from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import sys
import threading
import time
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .data import (
    FewShotRelationSampler,
    HistoryIndex,
    Quad,
    add_inverse,
    build_filter_dict,
    load_temporal_kg,
    tensorize,
)
from .model import NineFuseTKG
from .llm_cache import LLMEvidenceCache, cache_coverage, dataset_files_fingerprint


class CudaResourceMonitor:
    """Sample allocator state without opening a second CUDA/NVML context."""

    def __init__(self, device: torch.device, path: str, interval: float = 30.0) -> None:
        self.device = device
        self.path = path
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.device.type != "cuda" or not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("unix_time,allocated_mb,reserved_mb,max_allocated_mb,max_reserved_mb\n")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _sample(self) -> None:
        values = (
            time.time(),
            torch.cuda.memory_allocated(self.device) / (1024.0**2),
            torch.cuda.memory_reserved(self.device) / (1024.0**2),
            torch.cuda.max_memory_allocated(self.device) / (1024.0**2),
            torch.cuda.max_memory_reserved(self.device) / (1024.0**2),
        )
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(",".join(f"{value:.3f}" for value in values) + "\n")

    def _run(self) -> None:
        self._sample()
        while not self.stop_event.wait(self.interval):
            self._sample()

    def stop(self) -> None:
        if self.thread is None:
            return
        self.stop_event.set()
        self.thread.join(timeout=5.0)
        self._sample()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_features(features: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in features.items()}


def attach_llm_features(
    cache: LLMEvidenceCache | None,
    batch: Sequence[Quad],
    features: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return features if cache is None else cache.augment_features(batch, features)


def group_by_relation(quads: Sequence[Quad]) -> Dict[int, List[Quad]]:
    grouped: Dict[int, List[Quad]] = defaultdict(list)
    for quad in quads:
        grouped[quad[1]].append(quad)
    for rows in grouped.values():
        rows.sort(key=lambda x: (x[3], x[0], x[2]))
    return grouped


class RelationBatchSampler:
    """Sample supervised full-data batches while keeping relation-specific support coherent."""

    def __init__(
        self,
        grouped: Dict[int, List[Quad]],
        batch_size: int,
        seed: int,
    ) -> None:
        self.grouped = {r: rows for r, rows in grouped.items() if len(rows) >= 2}
        if not self.grouped:
            raise ValueError("Need at least one relation with two facts for warmup")
        self.relations = list(self.grouped.keys())
        self.batch_size = batch_size
        self.rng = random.Random(seed)

    def sample(self) -> Tuple[int, List[Quad]]:
        relation = self.rng.choice(self.relations)
        rows = self.grouped[relation]
        count = min(len(rows), self.batch_size)
        batch = self.rng.sample(rows, count) if len(rows) > count else list(rows)
        return relation, batch


def choose_support(
    grouped: Dict[int, List[Quad]],
    relation: int,
    shot: int,
    fallback: Sequence[Quad],
    rng: random.Random | None = None,
    exclude: Quad | None = None,
    exclude_set: set[Quad] | None = None,
) -> List[Quad]:
    rows = [
        x
        for x in grouped.get(relation, [])
        if x != exclude and (exclude_set is None or x not in exclude_set)
    ]
    if not rows:
        rows = list(fallback)
    if not rows:
        raise ValueError("No support facts are available")
    if len(rows) <= shot:
        return rows
    if rng is None:
        return rows[:shot]
    return rng.sample(rows, shot)


def choose_causal_support(
    grouped: Dict[int, List[Quad]],
    query: Quad,
    shot: int,
) -> List[Quad]:
    """Select the most recent support strictly before the query timestamp."""
    s, relation, _, query_time = query
    rows = grouped.get(relation, [])
    lo, hi = 0, len(rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if rows[mid][3] < query_time:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        # Neutral structural placeholder; it contains no answer label.
        return [(s, relation, s, max(0, query_time - 1))]
    # Inspect a bounded recent window, preferring subject matches only within
    # comparable recency. This keeps full evaluation logarithmic in relation size.
    window = rows[max(0, lo - max(shot * 8, 32)) : lo]
    window.sort(key=lambda row: (row[3], row[0] == s), reverse=True)
    return window[:shot]


def _support_row_key(row: Quad, query: Quad, seed: int) -> int:
    """Return a stable pseudo-random key without Python's salted hash()."""

    s, r, o, t = row
    qs, qr, _, qt = query
    value = (
        (s + 1) * 73856093
        ^ (r + 1) * 19349663
        ^ (o + 1) * 83492791
        ^ (t + 1) * 2654435761
        ^ (qs + 1) * 97531
        ^ (qr + 1) * 421
        ^ (qt + 1) * 65537
        ^ (seed + 1) * 104729
    )
    return value & 0xFFFFFFFFFFFFFFFF


def choose_perturbed_causal_support(
    grouped: Dict[int, List[Quad]],
    query: Quad,
    shot: int,
    perturbation: str = "none",
    seed: int = 0,
) -> List[Quad]:
    """Choose deterministic, strictly causal support for stress evaluation.

    ``mismatched`` keeps the oriented relation and temporal boundary fixed but
    selects recent facts from other subjects outside the nominal support set.
    ``stale`` selects the oldest eligible facts.  Neither mode reads the query
    target or any fact at/after the query timestamp.
    """

    if perturbation == "none":
        return choose_causal_support(grouped, query, shot)
    if perturbation not in {"mismatched", "stale"}:
        raise ValueError(f"unknown support perturbation: {perturbation}")

    subject, relation, _, query_time = query
    eligible = [row for row in grouped.get(relation, []) if row[3] < query_time]
    if not eligible:
        return [(subject, relation, subject, max(0, query_time - 1))]

    if perturbation == "stale":
        return eligible[:shot]

    nominal = set(choose_causal_support(grouped, query, shot))
    recent = eligible[-max(shot * 16, 64) :]
    primary = [row for row in recent if row not in nominal and row[0] != subject]
    fallback = [row for row in eligible if row not in nominal and row not in primary]
    primary.sort(key=lambda row: _support_row_key(row, query, seed))
    fallback.sort(key=lambda row: _support_row_key(row, query, seed + 1))
    selected = (primary + fallback)[:shot]
    if selected:
        return selected
    # A neutral placeholder is preferable to silently reverting to the exact
    # nominal support when no causal mismatch exists for a rare relation.
    return [(subject, relation, subject, max(0, query_time - 1))]


def causal_support_rows(
    grouped: Dict[int, List[Quad]],
    queries: Sequence[Quad],
    shot: int,
    perturbation: str = "none",
    seed: int = 0,
) -> List[List[Quad]]:
    return [
        choose_perturbed_causal_support(grouped, query, shot, perturbation, seed)
        for query in queries
    ]


def support_rows_tensor(
    supports: Sequence[Sequence[Quad]], device: torch.device
) -> torch.Tensor:
    if not supports:
        raise ValueError("support rows must not be empty")
    width = max(len(rows) for rows in supports)
    padded = [list(rows) + [rows[-1]] * (width - len(rows)) for rows in supports]
    return torch.as_tensor(padded, dtype=torch.long, device=device)


def causal_support_tensor(
    grouped: Dict[int, List[Quad]],
    queries: Sequence[Quad],
    shot: int,
    device: torch.device,
) -> torch.Tensor:
    return support_rows_tensor(causal_support_rows(grouped, queries, shot), device)


def oracle_loss(aux: Dict[str, torch.Tensor], features: Dict[str, torch.Tensor]) -> torch.Tensor:
    labels = copy_target_labels(features).to(aux["oracle_logit"].device)
    return F.binary_cross_entropy_with_logits(aux["oracle_logit"], labels)


def copy_target_labels(features: Dict[str, torch.Tensor]) -> torch.Tensor:
    labels = features["in_history"]
    return labels


def rel_copy_target_labels(features: Dict[str, torch.Tensor]) -> torch.Tensor:
    return features.get("in_rel_history", torch.zeros_like(features["in_history"]))


def rule_hit_labels(features: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    candidates = features["rule_candidates"].to(target.device)
    mask = features["rule_mask"].to(target.device)
    return ((candidates == target.unsqueeze(1)) & mask).any(dim=1).float()


def rule_reliability_loss(
    aux: Dict[str, torch.Tensor],
    features: Dict[str, torch.Tensor],
    target: torch.Tensor,
) -> torch.Tensor:
    labels = rule_hit_labels(features, target).to(aux["rule_reliability_logit"].device)
    return F.binary_cross_entropy_with_logits(aux["rule_reliability_logit"], labels)


def router_consistency_loss(
    aux: Dict[str, torch.Tensor],
    features: Dict[str, torch.Tensor],
    target: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    logits = aux.get("router_logits", aux["residual_gate_logits"])
    # v1.6.1 uses a rank-aligned target.  A fixed base-preserving mixture is
    # compared with generate using a top-negative listwise surrogate, avoiding
    # direct comparison of differently normalized standalone likelihoods.
    expert_logps = aux["expert_logps"].detach()
    batch = target.shape[0]
    coverage = torch.stack(
        [
            copy_target_labels(features).to(expert_logps.device),
            rel_copy_target_labels(features).to(expert_logps.device),
            rule_hit_labels(features, target).to(expert_logps.device),
        ],
        dim=-1,
    )
    base = expert_logps[:, 0]
    negative_mask = torch.ones_like(base, dtype=torch.bool)
    negative_mask.scatter_(1, target.unsqueeze(1), False)
    topk = min(32, max(1, base.shape[1] - 1))

    def rank_risk(scores: torch.Tensor) -> torch.Tensor:
        target_score = scores.gather(1, target.unsqueeze(1)).squeeze(1)
        negatives = scores.masked_fill(~negative_mask, -1e9)
        hard = negatives.topk(topk, dim=1).values
        return F.softplus(torch.logsumexp(hard, dim=1) - target_score)

    base_risk = rank_risk(base)
    gains = []
    log_base = math.log(0.65)
    log_expert = math.log(0.35)
    for expert_index in range(1, expert_logps.shape[1]):
        candidate = torch.logaddexp(log_base + base, log_expert + expert_logps[:, expert_index])
        gains.append(base_risk - rank_risk(candidate))
    rank_gain = torch.stack(gains, dim=-1)
    gain_target = coverage * torch.sigmoid(
        rank_gain / max(args.router_target_temperature, 1e-3)
    )
    return F.binary_cross_entropy_with_logits(logits, gain_target)


def sparse_correctness_loss(
    aux: Dict[str, torch.Tensor],
    features: Dict[str, torch.Tensor],
    target: torch.Tensor,
) -> torch.Tensor:
    device = aux["sparse_correctness_logits"].device
    labels = torch.stack(
        [
            copy_target_labels(features).to(device),
            rel_copy_target_labels(features).to(device),
            rule_hit_labels(features, target).to(device),
        ],
        dim=-1,
    )
    # Per-batch balancing prevents high availability from becoming an always-on
    # prediction and directly attacks selected-but-miss errors.
    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(0.5, 8.0)
    return F.binary_cross_entropy_with_logits(
        aux["sparse_correctness_logits"], labels, pos_weight=pos_weight
    )


def alterego_tournament_loss(
    aux: Dict[str, torch.Tensor], target: torch.Tensor, margin: float
) -> torch.Tensor:
    candidate_ids = aux["alterego_candidate_ids"]
    candidate_mask = aux["alterego_candidate_mask"]
    pairwise = aux["alterego_pairwise_scores"]
    if candidate_ids.shape[1] == 0:
        return pairwise.sum() * 0.0
    positives = candidate_ids.eq(target.unsqueeze(1)) & candidate_mask
    recalled = positives.any(dim=1)
    if not bool(recalled.any()):
        return pairwise.sum() * 0.0
    target_index = positives.float().argmax(dim=1)
    target_row = pairwise.gather(
        1, target_index.view(-1, 1, 1).expand(-1, 1, pairwise.shape[2])
    ).squeeze(1)
    negatives = candidate_mask & ~positives
    pair_loss = F.softplus(margin - target_row) * negatives.float()
    per_query = pair_loss.sum(dim=1) / negatives.sum(dim=1).clamp_min(1)
    return per_query[recalled].mean()


def corrupt_temporal_batch(batch: Sequence[Quad], num_times: int) -> List[Quad]:
    if num_times <= 1:
        return list(batch)
    corrupted: List[Quad] = []
    for s, r, o, t in batch:
        offset = 1 + (r % max(1, num_times - 1))
        wrong_t = (t + offset) % num_times
        if wrong_t == t:
            wrong_t = (t + 1) % num_times
        corrupted.append((s, r, o, int(wrong_t)))
    return corrupted


def temporal_hard_negative_loss(
    model: NineFuseTKG,
    batch: Sequence[Quad],
    support_tensor: torch.Tensor,
    history: HistoryIndex,
    target: torch.Tensor,
    positive_log_probs: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    negative_batch = corrupt_temporal_batch(batch, model.num_times)
    negative_features = move_features(history.build(negative_batch), device)
    negative_tensor = tensorize(negative_batch, device)
    negative_log_probs, _ = model(negative_tensor, support_tensor, negative_features)
    positive_scores = positive_log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    negative_scores = negative_log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    return F.softplus(args.hard_negative_margin + negative_scores - positive_scores).mean()


def evidence_auxiliary_loss(
    model: NineFuseTKG,
    batch: Sequence[Quad],
    support_tensor: torch.Tensor,
    history: HistoryIndex,
    features: Dict[str, torch.Tensor],
    target: torch.Tensor,
    log_probs: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    loss = torch.zeros((), device=device)
    if args.rule_reliability_weight > 0 and not args.disable_rule:
        loss = loss + args.rule_reliability_weight * rule_reliability_loss(aux, features, target)
    if args.router_weight > 0 and not args.disable_router:
        loss = loss + args.router_weight * router_consistency_loss(aux, features, target, args)
    if args.correctness_weight > 0:
        loss = loss + args.correctness_weight * sparse_correctness_loss(aux, features, target)
    if args.alterego_pair_loss_weight > 0 and not args.disable_alterego_tournament:
        loss = loss + args.alterego_pair_loss_weight * alterego_tournament_loss(
            aux, target, args.alterego_pair_margin
        )
    if args.hard_negative_weight > 0:
        loss = loss + args.hard_negative_weight * temporal_hard_negative_loss(
            model,
            batch,
            support_tensor,
            history,
            target,
            log_probs,
            args,
            device,
        )
    return loss


def run_supervised_warmup(
    model: NineFuseTKG,
    optimizer: torch.optim.Optimizer,
    history: HistoryIndex,
    support_by_rel: Dict[int, List[Quad]],
    train_aug: Sequence[Quad],
    args: argparse.Namespace,
    device: torch.device,
    llm_cache: LLMEvidenceCache | None = None,
) -> None:
    """Stage 1: fact-balanced, shuffled, complete passes over training facts."""

    if args.warmup_epochs <= 0:
        return

    rng = random.Random(args.seed + 23)
    facts = list(train_aug)
    natural_batches = (len(facts) + args.warmup_batch_size - 1) // args.warmup_batch_size
    batches_per_epoch = natural_batches
    if args.warmup_batches_per_epoch > 0:
        batches_per_epoch = min(natural_batches, args.warmup_batches_per_epoch)

    print(
        f"warmup_start epochs={args.warmup_epochs} "
        f"batches_per_epoch={batches_per_epoch}/{natural_batches} "
        f"batch_size={args.warmup_batch_size} fact_balanced=true"
    )
    for epoch in range(1, args.warmup_epochs + 1):
        model.train()
        total_loss = 0.0
        total_nll = 0.0
        rng.shuffle(facts)
        seen = 0
        for batch_index in range(batches_per_epoch):
            start = batch_index * args.warmup_batch_size
            batch = facts[start : start + args.warmup_batch_size]
            raw_features = attach_llm_features(llm_cache, batch, history.build(batch))
            features = move_features(raw_features, device)
            query_tensor = tensorize(batch, device)
            support_tensor = causal_support_tensor(support_by_rel, batch, args.shot, device)
            target = query_tensor[:, 2]

            log_probs, aux = model(query_tensor, support_tensor, features)
            # First establish a full-support generate backbone. Sparse experts
            # and the router are deliberately not optimized in this stage.
            generate_log_probs = aux["expert_logps"][:, 0]
            nll = F.nll_loss(generate_log_probs, target)
            loss = nll + args.freq_weight * aux["freq_reg"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += float(loss.item())
            total_nll += float(nll.item())
            seen += len(batch)

        avg_loss = total_loss / max(1, batches_per_epoch)
        avg_nll = total_nll / max(1, batches_per_epoch)
        print(f"warmup_epoch={epoch:03d} loss={avg_loss:.4f} nll={avg_nll:.4f} seen_facts={seen}")


def compute_batch_loss(
    model: NineFuseTKG,
    batch: Sequence[Quad],
    support: Sequence[Quad],
    history: HistoryIndex,
    support_by_rel: Dict[int, List[Quad]],
    args: argparse.Namespace,
    device: torch.device,
    llm_cache: LLMEvidenceCache | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raw_features = attach_llm_features(llm_cache, batch, history.build(batch))
    features = move_features(raw_features, device)
    # Episodic supports from older versions could contain future labels.  v1.5
    # always rebuilds them from the query's causal past.
    support_tensor = causal_support_tensor(support_by_rel, batch, args.shot, device)
    query_tensor = tensorize(batch, device)
    target = query_tensor[:, 2]
    log_probs, aux = model(query_tensor, support_tensor, features)
    nll = F.nll_loss(log_probs, target)
    loss = (
        nll
        + args.oracle_weight * oracle_loss(aux, features)
        + args.freq_weight * aux["freq_reg"]
        + evidence_auxiliary_loss(model, batch, support_tensor, history, features, target, log_probs, aux, args, device)
    )
    return loss, nll


def rank_stats_from_scores(log_probs: torch.Tensor, target: int, filtered_answers: Sequence[int]) -> Tuple[int, float, int]:
    scores = log_probs.detach().clone()
    target_score = scores[target].clone()
    for answer in filtered_answers:
        if answer != target:
            scores[answer] = -1e9
    scores[target] = target_score
    higher = int(torch.sum(scores > target_score).item())
    ties = int(torch.sum(scores == target_score).item()) - 1
    rank = higher + 1
    avg_tie_rank = rank + max(0, ties) / 2.0
    return rank, avg_tie_rank, max(0, ties)


def rank_from_scores(log_probs: torch.Tensor, target: int, filtered_answers: Sequence[int]) -> int:
    rank, _, _ = rank_stats_from_scores(log_probs, target, filtered_answers)
    return rank


def summarize_ranks(ranks: Sequence[float]) -> Dict[str, float]:
    if not ranks:
        return {"mrr": 0.0, "hits1": 0.0, "hits3": 0.0, "hits10": 0.0}
    denom = float(len(ranks))
    return {
        "mrr": sum(1.0 / r for r in ranks) / denom,
        "hits1": sum(r <= 1 for r in ranks) / denom,
        "hits3": sum(r <= 3 for r in ranks) / denom,
        "hits10": sum(r <= 10 for r in ranks) / denom,
    }


def dataset_manifest(data_dir: str, kg) -> Dict[str, object]:
    manifest: Dict[str, object] = {
        "benchmark": "ICEWS14-extrapolation" if (kg.num_entities, kg.num_relations) == (7128, 230) else "custom",
        "num_entities": kg.num_entities,
        "num_relations": kg.num_relations,
        "num_times": kg.num_times,
        "splits": {},
    }
    for name, rows in (("train", kg.train), ("valid", kg.valid), ("test", kg.test)):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            path += ".txt"
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest["splits"][name] = {
            "rows": len(rows),
            "time_min": min(row[3] for row in rows),
            "time_max": max(row[3] for row in rows),
            "sha256": digest.hexdigest(),
        }
    return manifest


def source_manifest_metadata(manifest_path: str = "") -> Dict[str, object]:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = manifest_path or os.path.join(project_root, "SOURCE_MANIFEST.sha256")
    path = os.path.abspath(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": path,
        "sha256": digest.hexdigest(),
    }


@torch.no_grad()
def evaluate(
    model: NineFuseTKG,
    quads: Sequence[Quad],
    history: HistoryIndex,
    support_by_rel: Dict[int, List[Quad]],
    train_aug: Sequence[Quad],
    filters: Dict[Tuple[int, int, int], set],
    num_relations: int,
    shot: int,
    device: torch.device,
    limit: int = 0,
    eval_batch_size: int = 64,
    history_protocol: str = "standard_rolling_history",
    llm_cache: LLMEvidenceCache | None = None,
    support_perturbation: str = "none",
    support_perturbation_seed: int = 0,
    query_export_path: str = "",
) -> Dict[str, float]:
    model.eval()
    ranks: List[int] = []
    raw_ranks: List[float] = []
    subject_ranks: List[float] = []
    object_ranks: List[float] = []
    tie_avg_ranks: List[float] = []
    total_target_ties = 0
    expert_weight_sum = torch.zeros(getattr(model, "num_experts", 4))
    residual_gate_sum = torch.zeros(3)
    residual_gate_low = torch.zeros(3)
    residual_gate_high = torch.zeros(3)
    temporal_diag_sum = defaultdict(float)
    expert_names = ("generate", "copy", "rel_copy", "rule")
    expert_ranks: Dict[str, List[float]] = {name: [] for name in expert_names}
    oracle_ranks: List[float] = []
    coverage_sum = torch.zeros(4)
    selected_miss = 0
    llm_cache_hits = 0
    llm_candidate_count = 0
    llm_candidate_hits = 0
    llm_bonus_sum = 0.0
    support_count_sum = 0.0
    support_age_min_sum = 0.0
    support_age_mean_sum = 0.0
    support_age_max_sum = 0.0
    support_perturbed_count = 0
    diag_count = 0
    query_records: List[Dict[str, object]] = []
    rows = list(quads)
    if limit and limit > 0:
        rows = rows[:limit]

    if history_protocol not in {"strict_static_history", "standard_rolling_history"}:
        raise ValueError(f"unknown history protocol: {history_protocol}")
    if support_perturbation not in {"none", "mismatched", "stale"}:
        raise ValueError(f"unknown support perturbation: {support_perturbation}")
    eval_history = history.clone()
    eval_support_by_rel = group_by_relation(eval_history.facts)
    snapshots: Dict[int, List[Quad]] = defaultdict(list)
    for quad in rows:
        snapshots[quad[3]].append(quad)

    for timestamp in sorted(snapshots):
        snapshot = snapshots[timestamp]
        oriented_rows = list(snapshot)
        oriented_rows.extend((o, r + num_relations, s, t) for s, r, o, t in snapshot)
        by_relation: Dict[int, List[Quad]] = defaultdict(list)
        for oriented in oriented_rows:
            by_relation[oriented[1]].append(oriented)

        for relation, rel_rows in by_relation.items():
          for start in range(0, len(rel_rows), eval_batch_size):
            batch = rel_rows[start : start + eval_batch_size]
            selected_supports = causal_support_rows(
                eval_support_by_rel,
                batch,
                shot,
                perturbation=support_perturbation,
                seed=support_perturbation_seed,
            )
            nominal_supports = (
                causal_support_rows(eval_support_by_rel, batch, shot)
                if support_perturbation != "none"
                else selected_supports
            )
            support_tensor = support_rows_tensor(selected_supports, device)
            raw_features = attach_llm_features(llm_cache, batch, eval_history.build(batch))
            features = move_features(raw_features, device)
            query_tensor = tensorize(batch, device)
            log_probs, aux = model(query_tensor, support_tensor, features)
            batch_count = query_tensor.shape[0]
            llm_ids = aux.get("llm_candidate_ids")
            llm_mask = aux.get("llm_candidate_mask")
            if llm_ids is not None and llm_mask is not None and llm_ids.numel() > 0:
                llm_target = query_tensor[:, 2].unsqueeze(1)
                llm_candidate_hits += int(((llm_ids == llm_target) & llm_mask).any(dim=1).sum().item())
                llm_candidate_count += int(llm_mask.sum().item())
            llm_cache_hits += int(aux.get("llm_cache_hit", torch.zeros(batch_count, device=device)).sum().item())
            llm_bonus_sum += float(aux.get("llm_bonus", torch.zeros_like(log_probs)).amax(dim=1).sum().item())
            expert_weight_sum += aux["expert_weights"].detach().cpu().sum(dim=0)
            residual_gates = aux.get("residual_gates")
            if residual_gates is not None:
                residual_gates_cpu = residual_gates.detach().cpu()
                residual_gate_sum += residual_gates_cpu.sum(dim=0)
                residual_gate_low += (residual_gates_cpu < 0.01).float().sum(dim=0)
                residual_gate_high += (
                    residual_gates_cpu > (0.95 * model.max_residual_gate)
                ).float().sum(dim=0)
            evidence = aux.get("evidence")
            evidence_cpu = None
            if evidence is not None and evidence.shape[1] >= 14:
                evidence_cpu = evidence.detach().cpu()
                temporal_diag_sum["history_freshness"] += float(evidence_cpu[:, 9].sum().item())
                temporal_diag_sum["history_density"] += float(evidence_cpu[:, 10].sum().item())
                temporal_diag_sum["relation_time_scale"] += float(evidence_cpu[:, 11].sum().item())
                temporal_diag_sum["support_freshness"] += float(evidence_cpu[:, 12].sum().item())
                temporal_diag_sum["support_temporal_focus"] += float(evidence_cpu[:, 13].sum().item())
                if evidence.shape[1] >= 16:
                    temporal_diag_sum["rel_copy_available"] += float(evidence_cpu[:, 14].sum().item())
                    temporal_diag_sum["rel_copy_strength"] += float(evidence_cpu[:, 15].sum().item())
            diag_count += batch_count
            log_probs_cpu = log_probs.cpu()
            expert_logps_cpu = aux["expert_logps"].detach().cpu()
            coverage = torch.stack(
                [
                    torch.ones(batch_count),
                    raw_features["in_history"],
                    raw_features["in_rel_history"],
                    rule_hit_labels(raw_features, torch.tensor([row[2] for row in batch])),
                ],
                dim=-1,
            )
            coverage_sum += coverage.sum(dim=0)
            if residual_gates is not None:
                max_gate, max_gate_index = residual_gates_cpu.max(dim=1)
                selected = torch.where(max_gate >= 0.5, max_gate_index + 1, torch.zeros_like(max_gate_index))
            else:
                selected = aux["expert_weights"].detach().cpu().argmax(dim=1)
            selected_miss += int(sum(coverage[i, int(selected[i])] < 0.5 for i in range(batch_count)))
            for i, oriented in enumerate(batch):
                support_rows_i = selected_supports[i]
                support_ages = [max(0, oriented[3] - row[3]) for row in support_rows_i]
                support_count = len(set(support_rows_i))
                support_age_min = min(support_ages)
                support_age_mean = sum(support_ages) / len(support_ages)
                support_age_max = max(support_ages)
                support_changed = support_rows_i != nominal_supports[i]
                support_count_sum += support_count
                support_age_min_sum += support_age_min
                support_age_mean_sum += support_age_mean
                support_age_max_sum += support_age_max
                support_perturbed_count += int(support_changed)
                filtered = filters.get((oriented[0], oriented[1], oriented[3]), set())
                rank, tie_avg_rank, target_ties = rank_stats_from_scores(log_probs_cpu[i], oriented[2], filtered)
                _, raw_rank, _ = rank_stats_from_scores(log_probs_cpu[i], oriented[2], ())
                ranks.append(rank)
                raw_ranks.append(raw_rank)
                (object_ranks if oriented[1] < num_relations else subject_ranks).append(tie_avg_rank)
                tie_avg_ranks.append(tie_avg_rank)
                total_target_ties += target_ties
                row_expert_ranks = []
                for expert_index, name in enumerate(expert_names):
                    _, expert_rank, _ = rank_stats_from_scores(
                        expert_logps_cpu[i, expert_index], oriented[2], filtered
                    )
                    expert_ranks[name].append(expert_rank)
                    row_expert_ranks.append(expert_rank)
                oracle_ranks.append(min(row_expert_ranks))
                if query_export_path:
                    llm_count_i = (
                        int(llm_mask[i].sum().item())
                        if llm_mask is not None and llm_mask.numel() > 0
                        else 0
                    )
                    llm_hit_i = (
                        bool(((llm_ids[i] == oriented[2]) & llm_mask[i]).any().item())
                        if llm_ids is not None and llm_mask is not None and llm_ids.numel() > 0
                        else False
                    )
                    evidence_values = (
                        evidence_cpu[i].tolist() if evidence_cpu is not None else []
                    )
                    query_records.append(
                        {
                            "subject": oriented[0],
                            "relation": oriented[1],
                            "object": oriented[2],
                            "timestamp": oriented[3],
                            "direction": "object" if oriented[1] < num_relations else "subject",
                            "rank": rank,
                            "tie_avg_rank": tie_avg_rank,
                            "raw_rank": raw_rank,
                            "target_ties": target_ties,
                            "generate_rank": row_expert_ranks[0],
                            "copy_rank": row_expert_ranks[1],
                            "rel_copy_rank": row_expert_ranks[2],
                            "rule_rank": row_expert_ranks[3],
                            "oracle_rank": min(row_expert_ranks),
                            "selected_expert": expert_names[int(selected[i])],
                            "selected_expert_available": bool(
                                coverage[i, int(selected[i])].item() >= 0.5
                            ),
                            "copy_target_available": bool(coverage[i, 1].item() >= 0.5),
                            "rel_copy_target_available": bool(coverage[i, 2].item() >= 0.5),
                            "rule_target_available": bool(coverage[i, 3].item() >= 0.5),
                            "history_freshness": evidence_values[9] if len(evidence_values) > 9 else 0.0,
                            "history_density": evidence_values[10] if len(evidence_values) > 10 else 0.0,
                            "relation_time_scale": evidence_values[11] if len(evidence_values) > 11 else 0.0,
                            "support_freshness": evidence_values[12] if len(evidence_values) > 12 else 0.0,
                            "support_temporal_focus": evidence_values[13] if len(evidence_values) > 13 else 0.0,
                            "relation_history_count": len(eval_support_by_rel.get(oriented[1], [])),
                            "support_count": support_count,
                            "support_age_min": support_age_min,
                            "support_age_mean": support_age_mean,
                            "support_age_max": support_age_max,
                            "support_perturbation": support_perturbation,
                            "support_changed": support_changed,
                            "llm_cache_hit": bool(
                                aux.get("llm_cache_hit", torch.zeros(batch_count, device=device))[i].item()
                            ),
                            "llm_mapped_candidate_count": llm_count_i,
                            "llm_target_recalled": llm_hit_i,
                            "llm_max_bonus": float(
                                aux.get("llm_bonus", torch.zeros_like(log_probs))[i].max().item()
                            ),
                        }
                    )

        if history_protocol == "standard_rolling_history":
            eval_history.add_facts(oriented_rows)
            for fact in oriented_rows:
                eval_support_by_rel[fact[1]].append(fact)

    metrics = summarize_ranks(ranks)
    tie_metrics = summarize_ranks(tie_avg_ranks)
    metrics.update({f"tie_avg_{key}": value for key, value in tie_metrics.items()})
    metrics.update({f"raw_{key}": value for key, value in summarize_ranks(raw_ranks).items()})
    metrics.update({f"object_{key}": value for key, value in summarize_ranks(object_ranks).items()})
    metrics.update({f"subject_{key}": value for key, value in summarize_ranks(subject_ranks).items()})
    for name, name_ranks in expert_ranks.items():
        standalone = summarize_ranks(name_ranks)
        metrics.update({f"{name}_{key}": value for key, value in standalone.items()})
    metrics.update({f"oracle_{key}": value for key, value in summarize_ranks(oracle_ranks).items()})
    if diag_count > 0:
        metrics.update(
            {
                "evaluated_query_count": float(diag_count),
                "avg_generate_weight": float(expert_weight_sum[0].item() / diag_count),
                "avg_copy_weight": float(expert_weight_sum[1].item() / diag_count),
                "avg_rel_copy_weight": float(expert_weight_sum[2].item() / diag_count),
                "avg_rule_weight": float(expert_weight_sum[3].item() / diag_count),
                "avg_target_ties": total_target_ties / diag_count,
                "avg_history_freshness": temporal_diag_sum["history_freshness"] / diag_count,
                "avg_history_density": temporal_diag_sum["history_density"] / diag_count,
                "avg_relation_time_scale": temporal_diag_sum["relation_time_scale"] / diag_count,
                "avg_support_freshness": temporal_diag_sum["support_freshness"] / diag_count,
                "avg_support_temporal_focus": temporal_diag_sum["support_temporal_focus"] / diag_count,
                "avg_rel_copy_available": temporal_diag_sum["rel_copy_available"] / diag_count,
                "avg_rel_copy_strength": temporal_diag_sum["rel_copy_strength"] / diag_count,
                "copy_target_coverage": float(coverage_sum[1].item() / diag_count),
                "rel_copy_target_coverage": float(coverage_sum[2].item() / diag_count),
                "rule_target_coverage": float(coverage_sum[3].item() / diag_count),
                "router_selected_miss_rate": selected_miss / diag_count,
                "avg_copy_gate": float(residual_gate_sum[0].item() / diag_count),
                "avg_rel_copy_gate": float(residual_gate_sum[1].item() / diag_count),
                "avg_rule_gate": float(residual_gate_sum[2].item() / diag_count),
                "copy_gate_low_rate": float(residual_gate_low[0].item() / diag_count),
                "rel_copy_gate_low_rate": float(residual_gate_low[1].item() / diag_count),
                "rule_gate_low_rate": float(residual_gate_low[2].item() / diag_count),
                "copy_gate_high_rate": float(residual_gate_high[0].item() / diag_count),
                "rel_copy_gate_high_rate": float(residual_gate_high[1].item() / diag_count),
                "rule_gate_high_rate": float(residual_gate_high[2].item() / diag_count),
                "history_protocol_rolling": float(history_protocol == "standard_rolling_history"),
                "llm_cache_hit_rate": llm_cache_hits / diag_count,
                "llm_avg_mapped_candidates": llm_candidate_count / diag_count,
                "llm_candidate_recall_at_10": llm_candidate_hits / diag_count,
                "llm_avg_max_bonus": llm_bonus_sum / diag_count,
                "avg_support_count": support_count_sum / diag_count,
                "avg_support_age_min": support_age_min_sum / diag_count,
                "avg_support_age_mean": support_age_mean_sum / diag_count,
                "avg_support_age_max": support_age_max_sum / diag_count,
                "support_perturbation_change_rate": support_perturbed_count / diag_count,
            }
        )
    if query_export_path:
        os.makedirs(os.path.dirname(os.path.abspath(query_export_path)), exist_ok=True)
        with open(query_export_path, "w", encoding="utf-8") as handle:
            for record in query_records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return metrics


def train(args: argparse.Namespace) -> Dict[str, float]:
    started_at = time.perf_counter()
    started_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    set_seed(args.seed)
    device = torch.device(args.device)
    resource_monitor = CudaResourceMonitor(device, args.resource_log)
    resource_monitor.start()
    kg = load_temporal_kg(args.data_dir, max_train=args.max_train)
    cache_paths = (
        args.llm_train_cache,
        args.llm_valid_cache,
        args.llm_test_cache,
        args.llm_alt_test_cache,
    )
    llm_dataset_fingerprint = dataset_files_fingerprint(args.data_dir) if any(cache_paths) else None
    frozen_parent_evaluation = bool(
        args.init_from_v5 and args.epochs <= 0 and args.warmup_epochs <= 0
    )
    if args.support_perturbation != "none" and not frozen_parent_evaluation:
        raise ValueError(
            "--support-perturbation is evaluation-only and requires "
            "--init-from-v5 with zero warmup/training epochs"
        )
    if args.llm_mode != "off" and not args.llm_test_cache:
        raise ValueError("active --llm-mode requires --llm-test-cache")
    if args.llm_mode != "off" and not frozen_parent_evaluation and not args.llm_valid_cache:
        raise ValueError(
            "an active training/model-selection run requires --llm-valid-cache; "
            "only a zero-epoch --init-from-v5 evaluation may use a test cache alone"
        )

    def load_cache(path: str, split: str, protocol: str) -> LLMEvidenceCache | None:
        if not path:
            return None
        return LLMEvidenceCache(
            path,
            max_candidates=args.llm_max_candidates,
            expected_shot=args.shot,
            expected_history_protocol=protocol,
            expected_split=split,
            expected_dataset_fingerprint=llm_dataset_fingerprint,
            require_generation_metadata=args.llm_mode != "off",
        )

    train_llm_cache = load_cache(args.llm_train_cache, "train", args.history_protocol)
    valid_llm_cache = load_cache(args.llm_valid_cache, "valid", args.history_protocol)
    test_llm_cache = load_cache(args.llm_test_cache, "test", args.history_protocol)
    alternate_protocol = (
        "strict_static_history" if args.history_protocol == "standard_rolling_history" else "standard_rolling_history"
    )
    alt_test_llm_cache = load_cache(args.llm_alt_test_cache, "test", alternate_protocol)

    def oriented(rows: Sequence[Quad]) -> List[Quad]:
        return list(rows) + [(o, r + kg.num_relations, s, t) for s, r, o, t in rows]

    valid_cache_coverage = cache_coverage(valid_llm_cache, oriented(kg.valid)) if valid_llm_cache else None
    test_cache_coverage = cache_coverage(test_llm_cache, oriented(kg.test)) if test_llm_cache else None
    alt_test_cache_coverage = cache_coverage(alt_test_llm_cache, oriented(kg.test)) if alt_test_llm_cache else None
    if args.llm_mode != "off" and not args.allow_partial_llm_cache:
        incomplete = {
            name: values["cache_hit_rate"]
            for name, values in {
                "valid": valid_cache_coverage,
                "test": test_cache_coverage,
                "alternate_test": alt_test_cache_coverage,
            }.items()
            if values is not None and values["cache_hit_rate"] < 1.0
        }
        if incomplete:
            raise ValueError(
                f"active formal run requires complete LLM caches; incomplete={incomplete}. "
                "Use --allow-partial-llm-cache only for smoke/debug runs."
            )
    train_aug = kg.train_aug
    use_temporal_calibration = not args.disable_temporal_calibration
    valid_history = HistoryIndex(
        train_aug,
        kg.num_entities,
        kg.num_relations * 2,
        args.history_len,
        adaptive_decay=use_temporal_calibration,
    )
    test_history = HistoryIndex(
        train_aug + kg.valid_aug,
        kg.num_entities,
        kg.num_relations * 2,
        args.history_len,
        adaptive_decay=use_temporal_calibration,
    )
    train_history = HistoryIndex(
        train_aug,
        kg.num_entities,
        kg.num_relations * 2,
        args.history_len,
        adaptive_decay=use_temporal_calibration,
    )

    sampler = FewShotRelationSampler(train_aug, shot=args.shot, query=args.query, seed=args.seed)
    support_by_rel = group_by_relation(train_aug)
    supervised_sampler = RelationBatchSampler(
        support_by_rel,
        batch_size=args.supervised_batch_size,
        seed=args.seed + 101,
    )
    all_aug = add_inverse(kg.train + kg.valid + kg.test, kg.num_relations)
    filters = build_filter_dict(all_aug)

    model = NineFuseTKG(
        num_entities=kg.num_entities,
        num_relations_total=kg.num_relations * 2,
        num_times=kg.num_times,
        dim=args.dim,
        history_len=args.history_len,
        channels=args.channels,
        dropout=args.dropout,
        use_copy=not args.disable_copy,
        use_rel_copy=not args.disable_rel_copy,
        use_rule=not args.disable_rule,
        use_geo=args.enable_geo and not args.disable_geo,
        use_freq=not args.disable_freq,
        use_history=not args.disable_history,
        use_support=not args.disable_support,
        use_support_gate=not args.disable_support_gate,
        use_router=not args.disable_router,
        use_temporal_calibration=use_temporal_calibration,
        use_snapshot_backbone=not args.disable_snapshot_backbone,
        use_candidate_rerank=not args.disable_candidate_rerank,
        use_alterego_tournament=not args.disable_alterego_tournament,
        alterego_candidate_k=args.alterego_candidate_k,
        alterego_tournament_rank=args.alterego_tournament_rank,
        alterego_max_delta=args.alterego_max_delta,
        llm_mode=args.llm_mode,
        llm_max_candidates=args.llm_max_candidates,
        llm_max_delta=args.llm_max_delta,
        llm_score_scale=args.llm_score_scale,
        llm_disable_confidence=args.llm_disable_confidence,
        expert_dropout=args.expert_dropout,
        max_residual_gate=args.max_residual_gate,
        max_expert_mass=args.max_expert_mass,
        fusion_mode=args.fusion_mode,
    ).to(device)
    checkpoint_initialization: Dict[str, object] | None = None
    if args.init_from_v5:
        source_checkpoint = torch.load(args.init_from_v5, map_location=device)
        source_state = source_checkpoint.get("model", source_checkpoint)
        incompatible = model.load_state_dict(source_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        invalid_missing = [key for key in incompatible.missing_keys if not key.startswith("llm_sidecar.")]
        if unexpected or invalid_missing:
            raise ValueError(
                f"incompatible v5 initialization: unexpected={unexpected}, invalid_missing={invalid_missing}"
            )
        checkpoint_initialization = {
            "path": os.path.abspath(args.init_from_v5),
            "allowed_missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": unexpected,
            "strict_base_compatibility": True,
        }
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "best.pt")
    meta = {
        "experiment_audit": {
            "command_argv": [sys.executable, *sys.argv],
            "seed": args.seed,
            "started_at_utc": started_at_utc,
            "hostname": platform.node(),
            "configured_device": args.device,
            "cache_paths": {
                "train": os.path.abspath(args.llm_train_cache) if args.llm_train_cache else None,
                "valid": os.path.abspath(args.llm_valid_cache) if args.llm_valid_cache else None,
                "test": os.path.abspath(args.llm_test_cache) if args.llm_test_cache else None,
                "alternate_test": os.path.abspath(args.llm_alt_test_cache)
                if args.llm_alt_test_cache
                else None,
            },
        },
        "data_dir": args.data_dir,
        "num_entities": kg.num_entities,
        "num_relations": kg.num_relations,
        "num_times": kg.num_times,
        "model": "NineFuseTKG",
        "version": "1.7.0alterego_v5_llm",
        "architecture": "STLP LLM candidate side-evidence + Antisymmetric Pairwise Tournament Reranker + ACR-TFCM-GS",
        "fusion_mode": args.fusion_mode,
        "dataset_manifest": dataset_manifest(args.data_dir, kg),
        "source_manifest": source_manifest_metadata(args.source_manifest),
        "mrr_target": args.mrr_target,
        "parameter_count": parameter_count,
        "model_config": {
            "dim": args.dim,
            "channels": args.channels,
            "history_len": args.history_len,
            "dropout": args.dropout,
            "fusion_mode": args.fusion_mode,
            "max_residual_gate": args.max_residual_gate,
            "max_expert_mass": args.max_expert_mass,
            "snapshot_backbone_enabled": not args.disable_snapshot_backbone,
            "support_gate_enabled": not args.disable_support_gate,
            "candidate_rerank_enabled": not args.disable_candidate_rerank,
            "alterego_tournament_enabled": not args.disable_alterego_tournament,
            "alterego_candidate_k": args.alterego_candidate_k,
            "alterego_tournament_rank": args.alterego_tournament_rank,
            "alterego_max_delta": args.alterego_max_delta,
            "llm_mode": args.llm_mode,
            "llm_max_candidates": args.llm_max_candidates,
            "llm_max_delta": args.llm_max_delta,
            "llm_score_scale": args.llm_score_scale,
            "llm_disable_confidence": args.llm_disable_confidence,
            "geo_enabled": args.enable_geo and not args.disable_geo,
            "support_perturbation": args.support_perturbation,
            "support_perturbation_seed": args.support_perturbation_seed,
        },
        "checkpoint_initialization": checkpoint_initialization,
        "llm": {
            "role": "target-blind candidate-side evidence; never a fifth expert or full-entity softmax",
            "runtime_api_calls": False,
            "mode": args.llm_mode,
            "train_cache": train_llm_cache.metadata() if train_llm_cache else None,
            "valid_cache": valid_llm_cache.metadata() if valid_llm_cache else None,
            "test_cache": test_llm_cache.metadata() if test_llm_cache else None,
            "alternate_test_cache": alt_test_llm_cache.metadata() if alt_test_llm_cache else None,
            "valid_cache_coverage": valid_cache_coverage,
            "test_cache_coverage": test_cache_coverage,
            "alternate_test_cache_coverage": alt_test_cache_coverage,
            "frozen_parent_evaluation": frozen_parent_evaluation,
        },
        "checkpoint_precision": "fp32",
        "primary_history_protocol": args.history_protocol,
        "training_stages": [
            "fact-balanced shuffled complete-epoch generate/fusion pretraining",
            "joint full-data supervised and few-shot episodic optimization",
        ],
        "v1_6_2advant_changes": [
            "candidate-level second-stage reranker over generate/copy/rel-copy/rule/relation-prior banks",
            "causal multi-scale temporal convolution and FFT low/high-frequency history memory",
            "geometry-aware conservative gate for support prototypes",
            "Gaussian time-decay freshness for support reliability",
            "long-term advant branch for implementing v1.7-v2.0 semantic/LLM improvements",
        ],
        "v1_7_0alterego_v5_changes": [
            "exactly antisymmetric low-rank pairwise candidate game",
            "GPU batched K-by-K payoff matrix without materializing K-by-K-by-D tensors",
            "soft Copeland win aggregation with zero-sum candidate corrections",
            "zero-utility and zero-interaction initialization preserves the four-expert base",
            "natural-recall pairwise margin supervision without target injection",
        ],
        "v1_7_0alterego_v5_llm_changes": [
            "target-blind semantic-temporal LLM prior generated outside model training",
            "immutable JSONL evidence cache with prompt hash and answer-free query key",
            "mapped candidates enter the v5 tournament without creating a fifth expert",
            "optional bounded score and temporal-rationale calibration",
            "LLM-off path preserves the complete v5 four-expert computation",
        ],
        "model_selection": "best validation average-tie filtered MRR checkpoint; test never selects checkpoints",
        "warmup": {
            "epochs": args.warmup_epochs,
            "batches_per_epoch": args.warmup_batches_per_epoch,
            "batch_size": args.warmup_batch_size,
        },
        "joint_training": {
            "supervised_weight": args.joint_supervised_weight,
            "supervised_batch_size": args.supervised_batch_size,
        },
        "evidence_router": {
            "enabled": not args.disable_router,
            "expert_dropout": args.expert_dropout,
            "rule_reliability_weight": args.rule_reliability_weight,
            "router_weight": args.router_weight,
            "hard_negative_weight": args.hard_negative_weight,
            "hard_negative_margin": args.hard_negative_margin,
            "router_target": "independent detached top-negative rank gain over generate",
            "router_target_temperature": args.router_target_temperature,
            "correctness_weight": args.correctness_weight,
        },
        "temporal_pace_calibration": {
            "enabled": use_temporal_calibration,
            "history_decay": "relation-specific median time-gap scale",
            "support_attention": "query-support time-gap penalty",
            "router_evidence": [
                "history_freshness",
                "history_density",
                "relation_time_scale",
                "support_freshness",
                "support_temporal_focus",
                "rel_copy_available",
                "rel_copy_strength",
            ],
        },
        "modules": [
            "causal two-hop snapshot graph propagation and recurrent evolution",
            "RE-Net recurrent history encoder",
            "causal multi-scale time-frequency history memory",
            "RE-GCN gated evolution",
            "CyGNet copy-generate mixture",
            "relation-copy fourth expert",
            "CENET oracle/history contrast signal",
            "function-only extrapolatable Fourier time encoder",
            "causal recent query-conditioned multi-prototype support adapter",
            "confidence-gated temporal path rule scorer",
            "evidence-calibrated temporal expert router",
            "temporal pace-calibrated history/rule/support evidence",
            "rule reliability self-supervision",
            "temporal hard negative contrast",
            "optional TeLM geometric temporal decoder (disabled by default)",
            "TeRDy FFT low/high-frequency relation adapter",
            "candidate-level second-stage reranker over recalled expert banks",
            "alterego antisymmetric pairwise tournament reranker",
            "optional STLP target-blind LLM candidate side-evidence",
            "sparse calibrated residual experts with learned temperatures",
            "snapshot-safe standard rolling history",
        ],
        "disabled_modules": [
            name
            for name, disabled in {
                "copy": args.disable_copy,
                "rel_copy": args.disable_rel_copy,
                "rule": args.disable_rule,
                "geo": args.disable_geo,
                "freq": args.disable_freq,
                "history": args.disable_history,
                "support": args.disable_support,
                "support_gate": args.disable_support_gate,
                "router": args.disable_router,
                "temporal_calibration": args.disable_temporal_calibration,
                "snapshot_backbone": args.disable_snapshot_backbone,
                "candidate_rerank": args.disable_candidate_rerank,
                "alterego_tournament": args.disable_alterego_tournament,
                "llm_side_evidence": args.llm_mode == "off",
                "llm_confidence": args.llm_disable_confidence,
            }.items()
            if disabled
        ],
    }
    with open(os.path.join(args.output_dir, "run_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)

    model.set_alterego_runtime_enabled(False)
    model.set_llm_runtime_mode("off")
    run_supervised_warmup(
        model=model,
        optimizer=optimizer,
        history=train_history,
        support_by_rel=support_by_rel,
        train_aug=train_aug,
        args=args,
        device=device,
        llm_cache=None,
    )
    model.set_alterego_runtime_enabled(not args.disable_alterego_tournament)
    model.set_llm_runtime_mode(args.llm_mode)

    best_valid = -1.0
    last_metrics: Dict[str, float] = {}
    rng = random.Random(args.seed + 1)
    supervised_rng = random.Random(args.seed + 121)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_epi_nll = 0.0
        total_sup_nll = 0.0
        for _ in range(args.episodes_per_epoch):
            support, query = sampler.sample()
            relation = query[0][1]
            if args.resample_support:
                support = choose_support(support_by_rel, relation, args.shot, train_aug, rng=rng, exclude=query[0])

            epi_loss, epi_nll = compute_batch_loss(
                model, query, support, train_history, support_by_rel, args, device, llm_cache=train_llm_cache
            )
            loss = epi_loss
            sup_nll = torch.zeros((), device=device)
            if args.joint_supervised_weight > 0:
                sup_relation, sup_batch = supervised_sampler.sample()
                sup_support = choose_support(
                    support_by_rel,
                    sup_relation,
                    args.shot,
                    train_aug,
                    rng=supervised_rng if args.resample_support else None,
                    exclude_set=set(sup_batch),
                )
                sup_loss, sup_nll = compute_batch_loss(
                    model,
                    sup_batch,
                    sup_support,
                    train_history,
                    support_by_rel,
                    args,
                    device,
                    llm_cache=train_llm_cache,
                )
                loss = loss + args.joint_supervised_weight * sup_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += float(loss.item())
            total_epi_nll += float(epi_nll.item())
            total_sup_nll += float(sup_nll.item())

        avg_loss = total_loss / max(1, args.episodes_per_epoch)
        avg_epi_nll = total_epi_nll / max(1, args.episodes_per_epoch)
        avg_sup_nll = total_sup_nll / max(1, args.episodes_per_epoch)
        print(f"epoch={epoch:03d} loss={avg_loss:.4f} epi_nll={avg_epi_nll:.4f} sup_nll={avg_sup_nll:.4f}")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            valid_metrics = evaluate(
                model,
                kg.valid,
                valid_history,
                support_by_rel,
                train_aug,
                filters,
                kg.num_relations,
                args.shot,
                device,
                limit=args.eval_limit,
                eval_batch_size=args.eval_batch_size,
                history_protocol=args.history_protocol,
                llm_cache=valid_llm_cache,
            )
            last_metrics = {f"valid_{k}": v for k, v in valid_metrics.items()}
            print(
                "valid "
                + " ".join(f"{key}={value:.4f}" for key, value in valid_metrics.items())
            )
            if valid_metrics["tie_avg_mrr"] > best_valid:
                best_valid = valid_metrics["tie_avg_mrr"]
                torch.save(
                    {"model": model.state_dict(), "args": vars(args), "meta": meta},
                    best_path,
                )

    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"loaded_best_checkpoint valid_mrr={best_valid:.4f} path={best_path}")

    test_metrics = evaluate(
        model,
        kg.test,
        test_history,
        support_by_rel,
        train_aug,
        filters,
        kg.num_relations,
        args.shot,
        device,
        limit=args.eval_limit,
        eval_batch_size=args.eval_batch_size,
        history_protocol=args.history_protocol,
        llm_cache=test_llm_cache,
        support_perturbation=args.support_perturbation,
        support_perturbation_seed=args.support_perturbation_seed,
        query_export_path=(
            os.path.join(args.query_export_dir, "test_queries.jsonl")
            if args.query_export_dir
            else ""
        ),
    )
    last_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
    configured_llm_mode = args.llm_mode
    if configured_llm_mode != "off" and alt_test_llm_cache is None:
        # A cache generated with the primary history protocol must never be
        # silently reused for a different causal context.
        model.set_llm_runtime_mode("off")
    static_or_rolling_metrics = evaluate(
        model,
        kg.test,
        test_history,
        support_by_rel,
        train_aug,
        filters,
        kg.num_relations,
        args.shot,
        device,
        limit=args.eval_limit,
        eval_batch_size=args.eval_batch_size,
        history_protocol=alternate_protocol,
        llm_cache=alt_test_llm_cache,
        support_perturbation=args.support_perturbation,
        support_perturbation_seed=args.support_perturbation_seed,
        query_export_path=(
            os.path.join(args.query_export_dir, f"test_{alternate_protocol}_queries.jsonl")
            if args.query_export_dir
            else ""
        ),
    )
    model.set_llm_runtime_mode(configured_llm_mode)
    alt_prefix = "test_static" if alternate_protocol == "strict_static_history" else "test_rolling"
    last_metrics.update({f"{alt_prefix}_{k}": v for k, v in static_or_rolling_metrics.items()})
    last_metrics[f"{alt_prefix}_llm_active"] = float(
        configured_llm_mode != "off" and alt_test_llm_cache is not None
    )
    last_metrics["mrr_target"] = args.mrr_target
    last_metrics["mrr_target_reached"] = float(test_metrics["tie_avg_mrr"] >= args.mrr_target)
    elapsed_seconds = time.perf_counter() - started_at
    last_metrics["parameter_count"] = float(parameter_count)
    last_metrics["elapsed_seconds"] = elapsed_seconds
    last_metrics["peak_gpu_allocated_mb"] = (
        torch.cuda.max_memory_allocated(device) / (1024.0 ** 2) if device.type == "cuda" else 0.0
    )
    last_metrics["peak_gpu_reserved_mb"] = (
        torch.cuda.max_memory_reserved(device) / (1024.0 ** 2) if device.type == "cuda" else 0.0
    )
    meta["resources"] = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "parameter_count": parameter_count,
        "elapsed_seconds": elapsed_seconds,
        "peak_gpu_allocated_mb": last_metrics["peak_gpu_allocated_mb"],
        "peak_gpu_reserved_mb": last_metrics["peak_gpu_reserved_mb"],
    }
    meta["experiment_audit"]["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    with open(os.path.join(args.output_dir, "run_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(last_metrics, handle, indent=2)
    print("test " + " ".join(f"{key}={value:.4f}" for key, value in test_metrics.items()))
    print(
        f"target_gate tie_aware_mrr={test_metrics['tie_avg_mrr']:.4f} "
        f"target={args.mrr_target:.4f} reached={bool(last_metrics['mrr_target_reached'])}"
    )
    resource_monitor.stop()
    return last_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train NineFuseTKG for few-shot temporal KG completion")
    parser.add_argument("--data-dir", default="data/ICEWS14")
    parser.add_argument("--output-dir", default="runs/ninefuse")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resource-log", default="")
    parser.add_argument(
        "--source-manifest",
        default="",
        help="optional frozen source manifest recorded in run metadata",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--episodes-per-epoch", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--warmup-batches-per-epoch", type=int, default=0, help="0 means a complete fact-balanced epoch")
    parser.add_argument("--warmup-batch-size", type=int, default=256)
    parser.add_argument("--joint-supervised-weight", type=float, default=0.5)
    parser.add_argument("--supervised-batch-size", type=int, default=256)
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--query", type=int, default=8)
    parser.add_argument("--history-len", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--oracle-weight", type=float, default=0.1)
    parser.add_argument("--freq-weight", type=float, default=1e-3)
    parser.add_argument("--rule-reliability-weight", type=float, default=0.03)
    parser.add_argument("--router-weight", type=float, default=0.10)
    parser.add_argument("--router-target-temperature", type=float, default=1.0)
    parser.add_argument("--generate-router-anchor", type=float, default=0.35)
    parser.add_argument("--correctness-weight", type=float, default=0.05)
    parser.add_argument("--hard-negative-weight", type=float, default=0.01)
    parser.add_argument("--hard-negative-margin", type=float, default=0.5)
    parser.add_argument("--alterego-pair-loss-weight", type=float, default=0.05)
    parser.add_argument("--alterego-pair-margin", type=float, default=0.20)
    parser.add_argument("--alterego-candidate-k", type=int, default=96)
    parser.add_argument("--alterego-tournament-rank", type=int, default=32)
    parser.add_argument("--alterego-max-delta", type=float, default=0.5)
    parser.add_argument(
        "--llm-mode",
        choices=["off", "candidate", "score", "rationale"],
        default="off",
        help="off is the exact v5 path; active modes consume target-blind JSONL caches only",
    )
    parser.add_argument("--llm-train-cache", default="")
    parser.add_argument("--llm-valid-cache", default="")
    parser.add_argument("--llm-test-cache", default="")
    parser.add_argument("--llm-alt-test-cache", default="")
    parser.add_argument("--llm-max-candidates", type=int, default=10)
    parser.add_argument("--llm-max-delta", type=float, default=0.35)
    parser.add_argument("--llm-score-scale", type=float, default=1.0)
    parser.add_argument(
        "--llm-disable-confidence",
        action="store_true",
        help="ablation: retain mapped LLM candidates and all other side features but zero LLM confidence",
    )
    parser.add_argument(
        "--allow-partial-llm-cache",
        action="store_true",
        help="debug only; formal active modes require 100%% validation/test query coverage",
    )
    parser.add_argument(
        "--init-from-v5",
        default="",
        help="optional v1.7.0alterego_v5 checkpoint; only new llm_sidecar keys may be missing",
    )
    parser.add_argument("--expert-dropout", type=float, default=0.05)
    parser.add_argument("--max-residual-gate", type=float, default=0.80)
    parser.add_argument("--max-expert-mass", type=float, default=0.65)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument(
        "--support-perturbation",
        choices=["none", "mismatched", "stale"],
        default="none",
        help="evaluation-only causal support stress condition",
    )
    parser.add_argument(
        "--support-perturbation-seed",
        type=int,
        default=0,
        help="stable selector seed for mismatched support evaluation",
    )
    parser.add_argument(
        "--query-export-dir",
        default="",
        help="optional directory for per-query JSONL diagnostics from final evaluations",
    )
    parser.add_argument(
        "--history-protocol",
        choices=["standard_rolling_history", "strict_static_history"],
        default="standard_rolling_history",
    )
    parser.add_argument("--mrr-target", type=float, default=0.50)
    parser.add_argument(
        "--fusion-mode",
        choices=["residual", "probability_mixture", "base_preserving_mixture"],
        default="probability_mixture",
    )
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--resample-support", action="store_true")
    parser.add_argument("--disable-copy", action="store_true")
    parser.add_argument("--disable-rel-copy", action="store_true")
    parser.add_argument("--disable-rule", action="store_true")
    parser.add_argument("--disable-geo", action="store_true")
    parser.add_argument("--enable-geo", action="store_true", help="restore the v1.5 geometric branch for ablation only")
    parser.add_argument("--disable-freq", action="store_true")
    parser.add_argument("--disable-history", action="store_true")
    parser.add_argument("--disable-support", action="store_true")
    parser.add_argument(
        "--disable-support-gate",
        action="store_true",
        help="gate-only ablation: retain support prototypes and replace their reliability gate by identity",
    )
    parser.add_argument("--disable-router", action="store_true")
    parser.add_argument("--disable-temporal-calibration", action="store_true")
    parser.add_argument("--disable-snapshot-backbone", action="store_true")
    parser.add_argument("--disable-candidate-rerank", action="store_true")
    parser.add_argument("--disable-alterego-tournament", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.dim % 4 != 0:
        raise ValueError("--dim must be divisible by 4")
    if not 0.0 <= args.generate_router_anchor < 1.0:
        raise ValueError("--generate-router-anchor must be in [0, 1)")
    if not 0.0 < args.max_residual_gate < 1.0:
        raise ValueError("--max-residual-gate must be in (0, 1)")
    if not 0.0 < args.max_expert_mass < 1.0:
        raise ValueError("--max-expert-mass must be in (0, 1)")
    if args.alterego_candidate_k < 2 or args.alterego_tournament_rank < 1:
        raise ValueError("alterego tournament needs candidate K >= 2 and positive rank")
    if args.alterego_max_delta <= 0 or args.alterego_pair_margin < 0:
        raise ValueError("alterego max delta must be positive and pair margin non-negative")
    if args.llm_max_candidates < 1 or args.llm_max_delta <= 0 or args.llm_score_scale < 0:
        raise ValueError("LLM candidates/max delta must be positive and score scale non-negative")
    if args.disable_support and args.disable_support_gate:
        raise ValueError("--disable-support and --disable-support-gate are redundant")
    if args.max_train <= 0:
        args.max_train = None
    train(args)


if __name__ == "__main__":
    main()
