from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class ContinuousTimeEncoder(nn.Module):
    """Function-only Fourier time encoder that extrapolates to unseen times.

    v1.4 used an embedding row for every timestamp.  Under a chronological
    split every validation/test row was therefore random and untrained.  This
    encoder has no timestamp lookup table: future times use the same smooth
    basis that is optimized on training timestamps.
    """

    def __init__(self, num_times: int, dim: int) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("time encoder dimension must be even")
        self.num_times = max(1, num_times)
        half = dim // 2
        periods = torch.logspace(0.0, math.log10(float(max(2, num_times * 2))), half)
        self.register_buffer("frequencies", (2.0 * math.pi / periods).view(1, -1))
        self.amplitude = nn.Parameter(torch.ones(1, dim))
        self.phase = nn.Parameter(torch.zeros(1, half))
        self.trend = nn.Sequential(nn.Linear(2, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        x = times.float().unsqueeze(-1)
        angle = x * self.frequencies + self.phase
        periodic = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
        normalized = x / float(max(1, self.num_times - 1))
        trend = self.trend(torch.cat([normalized, torch.log1p(x.clamp_min(0.0)) / math.log1p(self.num_times)], dim=-1))
        return self.norm(self.amplitude * periodic + trend)


class HistoryEncoder(nn.Module):
    """RE-Net style recurrent history aggregation with query attention."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Linear(dim * 3, dim)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.query = nn.Linear(dim * 3, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        hist_entity_emb: torch.Tensor,
        hist_rel_emb: torch.Tensor,
        hist_time_emb: torch.Tensor,
        query_entity: torch.Tensor,
        query_relation: torch.Tensor,
        query_time: torch.Tensor,
        hist_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([hist_entity_emb, hist_rel_emb, hist_time_emb], dim=-1)
        x = self.dropout(torch.tanh(self.input(x)))
        encoded, _ = self.gru(x)
        q = self.query(torch.cat([query_entity, query_relation, query_time], dim=-1))
        scores = torch.einsum("bd,bld->bl", q, encoded) / math.sqrt(encoded.shape[-1])
        scores = scores.masked_fill(~hist_mask, -1e9)
        attn = torch.softmax(scores, dim=-1).unsqueeze(-1)
        context = torch.sum(attn * encoded, dim=1)
        has_history = hist_mask.any(dim=1).float().unsqueeze(-1)
        context = context * has_history
        return self.norm(context)


class CausalMultiScaleMemoryEncoder(nn.Module):
    """Causal multi-scale history memory inspired by NFGFE/TDFT-style signals.

    v1.6.2 proved that rolling history is the strongest positive component, but
    its GRU attention reads only one sequential view.  This module adds a small
    temporal-convolution memory over the same strictly-past history window.  It
    is stateless and query-local, so it preserves chronological causality and
    cannot leak future validation/test facts across batches.
    """

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.dim = dim
        self.event_proj = nn.Sequential(nn.Linear(dim * 3, dim), nn.Tanh())
        self.convs = nn.ModuleList(
            nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel - 1)
            for kernel in (2, 3, 5)
        )
        self.frequency_fuse = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.memory_fuse = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.geometry_gate = nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.query_gate = nn.Sequential(nn.Linear(dim * 4, dim), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        hist_entity_emb: torch.Tensor,
        hist_rel_emb: torch.Tensor,
        hist_time_emb: torch.Tensor,
        hist_mask: torch.Tensor,
        base_context: torch.Tensor,
        query_entity: torch.Tensor,
        query_relation: torch.Tensor,
        query_time: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([hist_entity_emb, hist_rel_emb, hist_time_emb], dim=-1)
        x = self.dropout(self.event_proj(x))
        mask = hist_mask.float().unsqueeze(-1)
        x = x * mask

        conv_in = x.transpose(1, 2)
        contexts = []
        for conv in self.convs:
            out = F.relu(conv(conv_in)[..., : x.shape[1]])
            # HistoryIndex right-aligns valid events, so the final position is
            # the latest causal state whenever history exists.
            contexts.append(out[..., -1])
        conv_context = torch.stack(contexts, dim=0).mean(dim=0)

        # Causal time-frequency view over the same strictly-past window.  This
        # borrows the spirit of TDFT/TETFD without importing their code: low
        # frequencies summarize slow relation drift; high frequencies capture
        # sudden local changes in the recent event sequence.
        freq = torch.fft.rfft(x.float(), dim=1)
        freq_ids = torch.fft.rfftfreq(x.shape[1], d=1.0).to(x.device).view(1, -1, 1)
        low_mask = (freq_ids <= 0.25).to(freq.dtype)
        low_time = torch.fft.irfft(freq * low_mask, n=x.shape[1], dim=1).to(x.dtype)
        high_time = torch.fft.irfft(freq * (1.0 - low_mask), n=x.shape[1], dim=1).to(x.dtype)
        freq_context = self.frequency_fuse(torch.cat([low_time[:, -1], high_time[:, -1], x[:, -1]], dim=-1))
        multiscale = self.memory_fuse(torch.cat([conv_context, freq_context], dim=-1))

        cosine = F.cosine_similarity(base_context, multiscale, dim=-1).nan_to_num(0.0)
        cosine = ((cosine + 1.0) / 2.0).clamp(0.0, 1.0)
        distance = torch.exp(
            -torch.norm(base_context - multiscale, p=2, dim=-1) / math.sqrt(float(self.dim))
        ).clamp(0.0, 1.0)
        geom_gate = self.geometry_gate(torch.stack([cosine, distance], dim=-1))
        query_gate = self.query_gate(torch.cat([query_entity, query_relation, query_time, base_context], dim=-1))
        has_history = hist_mask.any(dim=1).float().unsqueeze(-1)
        return self.norm(multiscale * geom_gate * query_gate) * has_history


class CausalSnapshotGraphEncoder(nn.Module):
    """Relation-conditioned propagation over a sampled causal snapshot graph.

    Direct query-entity edges and their second-hop neighborhood are encoded
    separately, while a chronological GRU models how the local graph changes
    across snapshots.  The data index guarantees every edge time is strictly
    earlier than the query time.
    """

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.role = nn.Embedding(2, dim)
        self.message = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.relation_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.query = nn.Linear(dim * 3, dim)
        self.evolution = nn.GRU(dim, dim, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def _attend(query: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("bd,bld->bl", query, values) / math.sqrt(values.shape[-1])
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("bl,bld->bd", weights, values)
        return context * mask.any(dim=1).float().unsqueeze(-1)

    def forward(
        self,
        neighbor: torch.Tensor,
        edge_relation: torch.Tensor,
        edge_time: torch.Tensor,
        edge_role: torch.Tensor,
        edge_mask: torch.Tensor,
        query_entity: torch.Tensor,
        query_relation: torch.Tensor,
        query_time: torch.Tensor,
    ) -> torch.Tensor:
        roles = self.role(edge_role.clamp(min=0, max=1))
        messages = self.message(torch.cat([neighbor, edge_relation, edge_time, roles], dim=-1))
        rel_query = query_relation.unsqueeze(1).expand_as(edge_relation)
        messages = messages * self.relation_gate(torch.cat([edge_relation, rel_query], dim=-1))
        messages = messages * edge_mask.unsqueeze(-1)
        evolved, _ = self.evolution(messages)
        q = self.query(torch.cat([query_entity, query_relation, query_time], dim=-1))
        direct = self._attend(q, messages, edge_mask & edge_role.eq(0))
        second = self._attend(q, messages, edge_mask & edge_role.eq(1))
        temporal = self._attend(q, evolved, edge_mask)
        fused = self.fuse(torch.cat([query_entity, direct, second, temporal], dim=-1))
        return self.norm(fused) * edge_mask.any(dim=1).float().unsqueeze(-1)


class MultiPrototypeSupportEncoder(nn.Module):
    """Query-adaptive structural, temporal, and historical support prototypes."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.struct_event = nn.Sequential(nn.Linear(dim * 4, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.temporal_event = nn.Sequential(nn.Linear(dim * 3, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.history_event = nn.Sequential(nn.Linear(dim * 3, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.struct_query = nn.Linear(dim * 3, dim)
        self.temporal_query = nn.Linear(dim * 3, dim)
        self.history_query = nn.Linear(dim * 3, dim)
        self.proto_gate = nn.Sequential(nn.Linear(dim * 4, dim), nn.ReLU(), nn.Linear(dim, 3))
        self.time_gap_penalty = nn.Parameter(torch.tensor([0.05, 0.12, 0.08]))
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        support_s: torch.Tensor,
        support_o: torch.Tensor,
        support_t: torch.Tensor,
        query_s: torch.Tensor,
        query_r: torch.Tensor,
        query_t: torch.Tensor,
        history_ctx: torch.Tensor,
        support_time_gap: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        struct_events = self.struct_event(torch.cat([support_s, support_o, support_o - support_s, support_s * support_o], dim=-1))
        temporal_events = self.temporal_event(torch.cat([support_s, support_o, support_t], dim=-1))
        history_events = self.history_event(torch.cat([support_o, support_t, support_o - support_s], dim=-1))

        q_struct = self.struct_query(torch.cat([query_s, query_r, history_ctx], dim=-1))
        q_temporal = self.temporal_query(torch.cat([query_r, query_t, history_ctx], dim=-1))
        q_history = self.history_query(torch.cat([query_s, query_t, history_ctx], dim=-1))

        def attend(q: torch.Tensor, events: torch.Tensor, gap_index: int) -> torch.Tensor:
            # Support may be shared [S,D] or query-conditioned [B,S,D].
            if events.dim() == 2:
                scores = q @ events.t() / math.sqrt(q.shape[-1])
            else:
                scores = torch.einsum("bd,bsd->bs", q, events) / math.sqrt(q.shape[-1])
            if support_time_gap is not None:
                penalty = F.softplus(self.time_gap_penalty[gap_index]) * support_time_gap.to(scores.device)
                scores = scores - penalty
            weights = torch.softmax(scores, dim=-1)
            if events.dim() == 2:
                return weights @ events
            return torch.einsum("bs,bsd->bd", weights, events)

        struct_ctx = attend(q_struct, struct_events, 0)
        temporal_ctx = attend(q_temporal, temporal_events, 1)
        history_ctx_s = attend(q_history, history_events, 2)
        gate = torch.softmax(self.proto_gate(torch.cat([query_s, query_r, query_t, history_ctx], dim=-1)), dim=-1)
        contexts = torch.stack([struct_ctx, temporal_ctx, history_ctx_s], dim=1)
        fused = torch.sum(gate.unsqueeze(-1) * contexts, dim=1)
        support_axis = 0 if struct_events.dim() == 2 else 1
        global_proto = torch.stack(
            [struct_events.mean(dim=support_axis), temporal_events.mean(dim=support_axis), history_events.mean(dim=support_axis)],
            dim=0,
        ).mean(dim=0)
        if global_proto.dim() == 1:
            global_proto = global_proto.unsqueeze(0).expand_as(fused)
        return self.norm(fused), self.norm(global_proto)


class GeometryAwareSupportGate(nn.Module):
    """Conservative support reliability gate.

    v1.6.2 ablation showed that raw support prototypes slightly hurt the final
    MRR.  This gate lets support help only when it is geometrically consistent
    with the query/history state and temporally fresh.
    """

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim * 5 + 4, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -1.25)

    def forward(
        self,
        query_entity: torch.Tensor,
        query_relation: torch.Tensor,
        query_time: torch.Tensor,
        history_ctx: torch.Tensor,
        support_ctx: torch.Tensor,
        support_freshness: torch.Tensor,
        support_temporal_focus: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        anchor = query_entity + history_ctx
        alignment = F.cosine_similarity(anchor, support_ctx, dim=-1).nan_to_num(0.0)
        alignment = ((alignment + 1.0) / 2.0).clamp(0.0, 1.0)
        distance = torch.exp(
            -torch.norm(anchor - support_ctx, p=2, dim=-1) / math.sqrt(float(self.dim))
        ).clamp(0.0, 1.0)
        freshness = support_freshness.clamp(0.0, 1.0)
        focus = support_temporal_focus.clamp(0.0, 1.0)
        reliability = (0.35 * alignment + 0.25 * distance + 0.25 * freshness + 0.15 * focus).clamp(0.0, 1.0)
        scalar_features = torch.stack([alignment, distance, freshness, focus], dim=-1)
        gate_logit = self.net(
            torch.cat(
                [query_entity, query_relation, query_time, history_ctx, support_ctx, scalar_features],
                dim=-1,
            )
        )
        gate = torch.sigmoid(gate_logit) * (0.25 + 0.75 * reliability).unsqueeze(-1)
        return gate, reliability


class CandidateReranker(nn.Module):
    """Entity-level second-stage reranker over recalled candidate banks.

    Earlier versions mixed complete expert distributions.  The v1.6.2 oracle
    gap shows that the right answer is often present in one sparse expert but
    not promoted by the global router.  This module scores each recalled entity
    directly using query/history/support representations and per-candidate
    expert evidence, then writes a bounded bonus back into the full ranking.
    """

    def __init__(self, dim: int, scalar_dim: int = 13, dropout: float = 0.2) -> None:
        super().__init__()
        self.scalar_dim = scalar_dim
        self.scorer = nn.Sequential(
            nn.Linear(dim * 9 + scalar_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
        )
        self.scalar_prior = nn.Linear(scalar_dim, 1)
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.constant_(self.scorer[-1].bias, -2.0)
        nn.init.zeros_(self.scalar_prior.weight)
        nn.init.zeros_(self.scalar_prior.bias)
        # A light handcrafted prior gives the reranker a useful starting point
        # before the learned scorer specializes.  The dimensions are defined in
        # _apply_candidate_rerank.
        with torch.no_grad():
            self.scalar_prior.weight[0, 1] = 0.40  # copy score
            self.scalar_prior.weight[0, 2] = 0.35  # copy available
            self.scalar_prior.weight[0, 3] = 0.55  # rel-copy score
            self.scalar_prior.weight[0, 4] = 0.45  # rel-copy available
            self.scalar_prior.weight[0, 5] = 0.25  # rule score
            self.scalar_prior.weight[0, 6] = 0.20  # rule available
            self.scalar_prior.weight[0, 7] = 0.15  # relation prior

    def forward(
        self,
        dyn_s: torch.Tensor,
        rel: torch.Tensor,
        hist_ctx: torch.Tensor,
        support_ctx: torch.Tensor,
        time_emb: torch.Tensor,
        candidate_emb: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> torch.Tensor:
        batch, count, dim = candidate_emb.shape

        def expand(x: torch.Tensor) -> torch.Tensor:
            return x.unsqueeze(1).expand(batch, count, dim)

        dyn = expand(dyn_s)
        rel_e = expand(rel)
        hist = expand(hist_ctx)
        support = expand(support_ctx)
        time = expand(time_emb)
        x = torch.cat(
            [
                dyn,
                rel_e,
                hist,
                support,
                time,
                candidate_emb,
                dyn * candidate_emb,
                rel_e * candidate_emb,
                candidate_emb - dyn,
                scalar_features,
            ],
            dim=-1,
        )
        learned = self.scorer(x).squeeze(-1)
        prior = self.scalar_prior(scalar_features).squeeze(-1)
        return F.softplus(learned + prior)


class AntisymmetricTournamentReranker(nn.Module):
    """Pairwise candidate tournament with a soft Copeland aggregation.

    A low-rank bilinear game computes all candidate matchups in one BMM.  The
    comparator is exactly antisymmetric, so swapping candidate order negates a
    matchup and the final ranking is permutation equivariant.  This is neither
    attention nor optimal transport: candidates accumulate pairwise wins.
    """

    def __init__(
        self,
        dim: int,
        scalar_dim: int = 13,
        tournament_rank: int = 32,
        dropout: float = 0.1,
        max_delta: float = 0.5,
    ) -> None:
        super().__init__()
        self.tournament_rank = tournament_rank
        self.max_delta = max_delta
        self.query_projection = nn.Sequential(
            nn.Linear(dim * 5, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self.candidate_projection = nn.Linear(dim, dim)
        self.scalar_projection = nn.Sequential(nn.Linear(scalar_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.token_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        self.utility = nn.Linear(dim, 1)
        self.player_projection = nn.Linear(dim, tournament_rank, bias=False)
        self.opponent_projection = nn.Linear(dim, tournament_rank, bias=False)
        self.interaction_gate = nn.Parameter(torch.zeros(()))
        self.log_temperature = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.utility.weight)
        nn.init.zeros_(self.utility.bias)

    def forward(
        self,
        dyn_s: torch.Tensor,
        relation: torch.Tensor,
        history: torch.Tensor,
        support: torch.Tensor,
        time: torch.Tensor,
        candidate_emb: torch.Tensor,
        scalar_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dyn_s, relation, history, support, time = (
            value.detach() for value in (dyn_s, relation, history, support, time)
        )
        candidate_emb = candidate_emb.detach()
        scalar_features = scalar_features.detach()
        query_ctx = self.query_projection(torch.cat([dyn_s, relation, history, support, time], dim=-1))
        tokens = self.candidate_projection(candidate_emb) + self.scalar_projection(scalar_features)
        tokens = self.token_mixer(tokens + query_ctx.unsqueeze(1))

        utility = self.utility(tokens).squeeze(-1)
        player = self.player_projection(tokens)
        opponent = self.opponent_projection(tokens)
        game = torch.bmm(player, opponent.transpose(1, 2)) / math.sqrt(float(self.tournament_rank))
        interaction = game - game.transpose(1, 2)
        pairwise = utility.unsqueeze(2) - utility.unsqueeze(1)
        pairwise = pairwise + torch.tanh(self.interaction_gate) * interaction
        pair_mask = candidate_mask.unsqueeze(2) & candidate_mask.unsqueeze(1)
        pairwise = pairwise * pair_mask.float()

        temperature = F.softplus(self.log_temperature) + 0.10
        soft_wins = torch.tanh(pairwise / (2.0 * temperature)) * pair_mask.float()
        opponents = (candidate_mask.sum(dim=1, keepdim=True) - 1).clamp_min(1).to(tokens.dtype)
        copeland = soft_wins.sum(dim=2) / opponents
        delta = self.max_delta * copeland * candidate_mask.float()
        return delta, pairwise, utility


class TemporalPathRuleScorer(nn.Module):
    """Learned temporal path scorer over retrieved candidates."""

    def __init__(self, dim: int, num_path_types: int = 5, dropout: float = 0.2) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(num_path_types, dim)
        self.type_bias = nn.Parameter(torch.zeros(num_path_types))
        self.delta_mlp = nn.Sequential(nn.Linear(3, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.score = nn.Sequential(
            nn.Linear(dim * 7, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(
        self,
        query_entity: torch.Tensor,
        query_relation: torch.Tensor,
        query_time: torch.Tensor,
        candidate_emb: torch.Tensor,
        path_r1: torch.Tensor,
        path_r2: torch.Tensor,
        path_dt1: torch.Tensor,
        path_dt2: torch.Tensor,
        path_type: torch.Tensor,
        path_prior: torch.Tensor,
        path_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, topk, dim = candidate_emb.shape
        q_entity = query_entity.unsqueeze(1).expand(batch, topk, dim)
        q_relation = query_relation.unsqueeze(1).expand(batch, topk, dim)
        q_time = query_time.unsqueeze(1).expand(batch, topk, dim)
        deltas = torch.stack([torch.log1p(path_dt1), torch.log1p(path_dt2), path_prior], dim=-1)
        delta_emb = self.delta_mlp(deltas)
        clamped_type = path_type.clamp(min=0, max=self.type_embedding.num_embeddings - 1)
        type_emb = self.type_embedding(clamped_type)
        type_bias = self.type_bias[clamped_type]
        x = torch.cat([q_entity, q_relation, q_time, candidate_emb, path_r1, path_r2, delta_emb + type_emb], dim=-1)
        scores = self.score(x).squeeze(-1) + path_prior + type_bias
        return scores.masked_fill(~path_mask, -1e9)


class RuleReliabilityHead(nn.Module):
    """Predict whether retrieved rule candidates are trustworthy for a query."""

    def __init__(self, dim: int, evidence_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4 + evidence_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(
        self,
        dyn_s: torch.Tensor,
        rel: torch.Tensor,
        hist_ctx: torch.Tensor,
        support_ctx: torch.Tensor,
        evidence: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([dyn_s, rel, hist_ctx, support_ctx, evidence], dim=-1)
        return self.net(x).squeeze(-1)


class IndependentResidualGate(nn.Module):
    """Predict independent copy/rel-copy/rule residual confidence.

    Generate is deliberately absent: it is always the full-support base and no
    longer competes with sparse experts in a softmax.  This removes the
    generate-only collapse observed in five of twelve v1.5 formal runs.
    """

    def __init__(self, dim: int, evidence_dim: int, num_residuals: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.query_router = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_residuals),
        )
        self.evidence_router = nn.Sequential(
            nn.Linear(evidence_dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, num_residuals),
        )
        self.evidence_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, query_repr: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        return self.query_router(query_repr) + self.evidence_scale * self.evidence_router(evidence)


class ConvTemporalDecoder(nn.Module):
    """ConvTransE-like decoder used by RE-GCN/TiRGN families."""

    def __init__(self, num_entities: int, dim: int, channels: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(2, channels, kernel_size=3, padding=1)
        self.fc = nn.Linear(channels * dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(num_entities))

    def forward(self, lhs: torch.Tensor, rel: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack([self.input_norm(lhs), self.input_norm(rel)], dim=1)
        x = self.dropout(stacked)
        x = F.relu(self.conv(x))
        x = self.dropout(x.flatten(1))
        x = F.relu(self.output_norm(self.fc(x)))
        return x @ candidates.t() + self.bias


class FrequencyRelationAdapter(nn.Module):
    """TeRDy-style low/high-frequency relation decomposition."""

    def __init__(self, dim: int, alpha: float = 10.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.fuse = nn.Sequential(nn.Linear(dim * 3, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, rel: torch.Tensor, time: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        freq = torch.fft.fft(rel, dim=-1)
        freqs = torch.fft.fftfreq(rel.shape[-1], d=1.0).to(rel.device)
        low_mask = torch.exp(-torch.abs(freqs) * self.alpha).view(1, -1)
        high_mask = 1.0 - low_mask
        low_freq = freq * low_mask
        high_freq = freq * high_mask
        low = torch.fft.ifft(low_freq, dim=-1).real
        high = torch.fft.ifft(high_freq, dim=-1).real
        time_gradient = time - time.mean(dim=-1, keepdim=True)
        adapted = self.fuse(torch.cat([low + time.mean(dim=-1, keepdim=True), high + time_gradient, rel], dim=-1))
        separation = torch.mean(torch.abs(low * high))
        high_energy = torch.mean(torch.abs(high_freq))
        return adapted, separation + 1e-4 * high_energy


class GeometricTemporalDecoder(nn.Module):
    """TeLM-inspired 2-grade multivector score."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("`dim` must be divisible by 4 for the TeLM-style decoder")
        self.rank = dim // 4

    def _split(self, x: torch.Tensor):
        return (
            x[..., : self.rank],
            x[..., self.rank : 2 * self.rank],
            x[..., 2 * self.rank : 3 * self.rank],
            x[..., 3 * self.rank :],
        )

    def forward(self, lhs: torch.Tensor, rel: torch.Tensor, time: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        lhs = self._split(lhs)
        rel = self._split(rel)
        time = self._split(time)
        rhs = self._split(candidates)

        a = rel[0] * time[0] + rel[1] * time[1] + rel[2] * time[2] - rel[3] * time[3]
        b = rel[0] * time[1] + rel[1] * time[0] - rel[2] * time[3] + rel[3] * time[2]
        c = rel[0] * time[2] + rel[2] * time[0] + rel[1] * time[3] - rel[3] * time[1]
        d = rel[1] * time[2] - rel[2] * time[1] + rel[0] * time[3] + rel[3] * time[0]

        w = lhs[0] * a + lhs[1] * b + lhs[2] * c - lhs[3] * d
        x = lhs[0] * b + lhs[1] * a - lhs[2] * d + lhs[3] * c
        y = lhs[0] * c + lhs[2] * a + lhs[1] * d - lhs[3] * b
        z = lhs[1] * c - lhs[2] * b + lhs[0] * d + lhs[3] * a

        return w @ rhs[0].t() - x @ rhs[1].t() - y @ rhs[2].t() + z @ rhs[3].t()


class LLMCandidateSidecar(nn.Module):
    """Bounded sparse calibration for target-blind LLM candidate evidence.

    This module never produces a full-entity distribution and is deliberately
    outside the four-expert tensor.  Candidate mode returns an exact zero
    residual; score and rationale modes add only bounded sparse evidence.
    """

    def __init__(self, max_delta: float = 0.35, score_scale: float = 1.0) -> None:
        super().__init__()
        if max_delta <= 0:
            raise ValueError("LLM sidecar max_delta must be positive")
        if score_scale < 0:
            raise ValueError("LLM sidecar score_scale must be non-negative")
        self.max_delta = float(max_delta)
        self.score_scale = float(score_scale)
        # Fixed-value initialization uses no RNG, preserving the v5 initialization
        # and training random stream when the sidecar is disabled.
        self.feature_logits = nn.Parameter(torch.tensor([1.0, 1.4, 0.5, 0.2, 0.4]))
        self.scale_logit = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        base_log_probs: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
        confidence: torch.Tensor,
        mapping_score: torch.Tensor,
        template_agreement: torch.Tensor,
        temporal_score: torch.Tensor,
        rank_prior: torch.Tensor,
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mode not in {"candidate", "score", "rationale"}:
            raise ValueError(f"LLM sidecar cannot run in mode: {mode}")
        dense_bonus = torch.zeros_like(base_log_probs)
        if mode == "candidate" or candidate_ids.numel() == 0:
            return base_log_probs, dense_bonus

        raw_features = torch.stack(
            [confidence, mapping_score, template_agreement, temporal_score, rank_prior],
            dim=-1,
        ).clamp(0.0, 1.0)
        active = torch.ones(5, dtype=torch.bool, device=base_log_probs.device)
        if mode == "score":
            active[3] = False
        masked_logits = self.feature_logits.to(base_log_probs.device).masked_fill(~active, -1e9)
        weights = torch.softmax(masked_logits, dim=0)
        evidence = (raw_features * weights).sum(dim=-1)
        # Mapping confidence is a safety gate: weak/ambiguous names cannot get a
        # large residual even if the language model is overconfident.
        reliability = evidence * mapping_score.clamp(0.0, 1.0)
        learned_scale = 2.0 * torch.sigmoid(self.scale_logit)
        candidate_bonus = (
            self.max_delta * self.score_scale * learned_scale * reliability * candidate_mask.float()
        ).clamp(0.0, self.max_delta)
        safe_ids = candidate_ids.clamp(0, base_log_probs.shape[1] - 1)
        dense_bonus = dense_bonus.scatter_reduce(
            1,
            safe_ids,
            candidate_bonus,
            reduce="amax",
            include_self=True,
        )
        return base_log_probs + dense_bonus, dense_bonus


class NineFuseTKG(nn.Module):
    """v1.7.0alterego_v5_llm target-blind LLM candidate-side model.

    The model keeps the mechanisms modular so ablations can disable or replace
    individual experts later without rewriting the training loop.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations_total: int,
        num_times: int,
        dim: int = 128,
        history_len: int = 8,
        channels: int = 32,
        dropout: float = 0.2,
        use_copy: bool = True,
        use_rel_copy: bool = True,
        use_rule: bool = True,
        use_geo: bool = False,
        use_freq: bool = True,
        use_history: bool = True,
        use_support: bool = True,
        use_support_gate: bool = True,
        use_router: bool = True,
        use_temporal_calibration: bool = True,
        use_snapshot_backbone: bool = True,
        use_candidate_rerank: bool = True,
        use_alterego_tournament: bool = True,
        alterego_candidate_k: int = 96,
        alterego_tournament_rank: int = 32,
        alterego_max_delta: float = 0.5,
        llm_mode: str = "off",
        llm_max_candidates: int = 10,
        llm_max_delta: float = 0.35,
        llm_score_scale: float = 1.0,
        llm_disable_confidence: bool = False,
        expert_dropout: float = 0.15,
        max_residual_gate: float = 0.80,
        max_expert_mass: float = 0.65,
        fusion_mode: str = "probability_mixture",
    ) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_relations_total = num_relations_total
        self.num_times = num_times
        self.dim = dim
        self.history_len = history_len
        self.use_copy = use_copy
        self.use_rel_copy = use_rel_copy
        self.use_rule = use_rule
        self.use_geo = use_geo
        self.use_freq = use_freq
        self.use_history = use_history
        self.use_support = use_support
        self.use_support_gate = use_support_gate
        self.use_router = use_router
        self.use_temporal_calibration = use_temporal_calibration
        self.use_snapshot_backbone = use_snapshot_backbone
        self.use_candidate_rerank = use_candidate_rerank
        self.use_alterego_tournament = use_alterego_tournament
        self.alterego_runtime_enabled = use_alterego_tournament
        self.alterego_candidate_k = max(1, min(alterego_candidate_k, num_entities))
        if llm_mode not in {"off", "candidate", "score", "rationale"}:
            raise ValueError(f"unknown LLM mode: {llm_mode}")
        if llm_max_candidates < 1:
            raise ValueError("llm_max_candidates must be positive")
        self.llm_mode = llm_mode
        self.llm_runtime_mode = llm_mode
        self.llm_max_candidates = int(llm_max_candidates)
        self.llm_disable_confidence = bool(llm_disable_confidence)
        self.expert_dropout = expert_dropout
        self.max_residual_gate = max_residual_gate
        self.max_expert_mass = max_expert_mass
        self.fusion_mode = fusion_mode

        self.entity = nn.Embedding(num_entities, dim)
        self.relation = nn.Embedding(num_relations_total, dim)
        self.time = ContinuousTimeEncoder(num_times, dim)

        self.history_encoder = HistoryEncoder(dim, dropout)
        self.multiscale_history = CausalMultiScaleMemoryEncoder(dim, dropout)
        self.history_memory_scale = nn.Parameter(torch.tensor(-1.0))
        self.history_memory_norm = nn.LayerNorm(dim)
        self.snapshot_encoder = CausalSnapshotGraphEncoder(dim, dropout)
        self.snapshot_scale = nn.Parameter(torch.tensor(-0.5))
        self.snapshot_norm = nn.LayerNorm(dim)
        self.support_encoder = MultiPrototypeSupportEncoder(dim, dropout)
        self.support_gate = GeometryAwareSupportGate(dim, dropout)
        self.support_proto_scale = nn.Parameter(torch.tensor(-1.0))
        self.support_decay_log_sigma = nn.Parameter(torch.tensor(-2.0))
        self.freq_adapter = FrequencyRelationAdapter(dim)
        self.rule_scorer = TemporalPathRuleScorer(dim, dropout=dropout)
        self.rel_gru = nn.GRUCell(dim, dim)
        self.entity_gate = nn.Sequential(nn.Linear(dim * 3, dim), nn.Sigmoid())

        self.conv_decoder = ConvTemporalDecoder(num_entities, dim, channels, dropout)
        self.geo_decoder = GeometricTemporalDecoder(dim)
        self.trans_proj = nn.Sequential(nn.Linear(dim * 4, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.semantic_adapter = nn.Sequential(nn.Linear(dim * 2, dim), nn.Tanh())

        self.decoder_weights = nn.Parameter(torch.tensor([0.45, 0.35, 0.20]))
        self.relation_prior_scale = nn.Parameter(torch.tensor(-2.0))
        self.relation_copy_logit_boost = nn.Parameter(torch.tensor(2.0))
        self.oracle = nn.Sequential(nn.Linear(dim * 3, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, 1))
        self.evidence_dim = 16
        self.num_experts = 4
        self.candidate_reranker = CandidateReranker(dim, scalar_dim=13, dropout=dropout)
        self.candidate_rerank_scale = nn.Parameter(torch.tensor(0.75))
        self.residual_gate = IndependentResidualGate(dim, self.evidence_dim, num_residuals=3, dropout=dropout)
        self.fallback_gate = nn.Sequential(nn.Linear(dim * 4, dim), nn.ReLU(), nn.Linear(dim, 3))
        self.rule_reliability = RuleReliabilityHead(dim, self.evidence_dim, dropout=dropout)
        self.sparse_correctness = nn.Sequential(
            nn.Linear(dim * 4 + self.evidence_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 3),
        )
        # Per-expert temperatures decouple confidence scale from expert choice.
        self.expert_log_temperature = nn.Parameter(torch.zeros(self.num_experts))
        self.residual_scale = nn.Parameter(torch.tensor([3.0, 4.0, 2.0]))

        self.reset_parameters()
        self.alterego_tournament = AntisymmetricTournamentReranker(
            dim=dim,
            scalar_dim=13,
            tournament_rank=alterego_tournament_rank,
            dropout=min(dropout, 0.1),
            max_delta=alterego_max_delta,
        )
        self.llm_sidecar = LLMCandidateSidecar(
            max_delta=llm_max_delta,
            score_scale=llm_score_scale,
        )

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)
        # Start sparse residuals conservatively.  v1.5 showed that a saturated
        # router can dominate optimization before correctness is learned.
        nn.init.zeros_(self.residual_gate.query_router[-1].weight)
        nn.init.constant_(self.residual_gate.query_router[-1].bias, -1.5)
        nn.init.zeros_(self.residual_gate.evidence_router[-1].weight)
        nn.init.zeros_(self.residual_gate.evidence_router[-1].bias)
        nn.init.zeros_(self.fallback_gate[-1].weight)
        nn.init.constant_(self.fallback_gate[-1].bias, -1.5)
        nn.init.zeros_(self.sparse_correctness[-1].weight)
        nn.init.zeros_(self.sparse_correctness[-1].bias)

    def set_alterego_runtime_enabled(self, enabled: bool) -> None:
        self.alterego_runtime_enabled = bool(enabled and self.use_alterego_tournament)

    def set_llm_runtime_mode(self, mode: str) -> None:
        if mode not in {"off", "candidate", "score", "rationale"}:
            raise ValueError(f"unknown LLM mode: {mode}")
        self.llm_runtime_mode = mode if self.llm_mode != "off" else "off"

    def encode_query(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        features: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s, r, _, t = query[:, 0], query[:, 1], query[:, 2], query[:, 3]
        s_emb = self.entity(s)
        r_emb = self.relation(r)
        t_emb = self.time(t)

        hist_e = self.entity(features["hist_entities"])
        hist_r = self.relation(features["hist_relations"])
        hist_t = self.time(features["hist_times"])
        if self.use_history:
            hist_ctx = self.history_encoder(hist_e, hist_r, hist_t, s_emb, r_emb, t_emb, features["hist_mask"])
            memory_ctx = self.multiscale_history(
                hist_e,
                hist_r,
                hist_t,
                features["hist_mask"],
                hist_ctx,
                s_emb,
                r_emb,
                t_emb,
            )
            hist_ctx = self.history_memory_norm(
                hist_ctx + torch.sigmoid(self.history_memory_scale) * memory_ctx
            )
        else:
            hist_ctx = torch.zeros_like(s_emb)

        snapshot_ctx = torch.zeros_like(s_emb)
        if self.use_snapshot_backbone and "snapshot_entities" in features:
            snapshot_ctx = self.snapshot_encoder(
                self.entity(features["snapshot_entities"]),
                self.relation(features["snapshot_relations"]),
                self.time(features["snapshot_times"]),
                features["snapshot_roles"],
                features["snapshot_mask"],
                s_emb,
                r_emb,
                t_emb,
            )
            hist_ctx = self.snapshot_norm(
                hist_ctx + torch.sigmoid(self.snapshot_scale) * snapshot_ctx
            )

        if self.use_support:
            support_s = self.entity(support[..., 0])
            support_o = self.entity(support[..., 2])
            support_t = self.time(support[..., 3])
            support_time_gap = None
            support_freshness = torch.zeros_like(s_emb[:, 0])
            support_temporal_focus = torch.zeros_like(s_emb[:, 0])
            if self.use_temporal_calibration:
                support_times = support[..., 3].float()
                if support.dim() == 2:
                    support_times = support_times.unsqueeze(0)
                raw_gap = torch.abs(t.float().unsqueeze(1) - support_times)
                support_time_gap = raw_gap / max(1, self.num_times - 1)
                sigma = self.support_decay_log_sigma.exp().clamp(0.03, 0.60)
                support_freshness = torch.exp(-0.5 * (support_time_gap / sigma).pow(2)).amax(dim=1).clamp(0.0, 1.0)
                support_temporal_focus = (1.0 - support_time_gap.mean(dim=1)).clamp(0.0, 1.0)
            support_ctx, support_proto = self.support_encoder(
                support_s,
                support_o,
                support_t,
                s_emb,
                r_emb,
                t_emb,
                hist_ctx,
                support_time_gap=support_time_gap,
            )
            if self.use_support_gate:
                support_gate, support_reliability = self.support_gate(
                    s_emb,
                    r_emb,
                    t_emb,
                    hist_ctx,
                    support_ctx,
                    support_freshness,
                    support_temporal_focus,
                )
            else:
                # Gate-only ablation: retain the complete support encoder and
                # prototype path, but replace the learned reliability gate by
                # the identity.  Keeping the gate parameters in the state dict
                # makes existing checkpoints load without architectural drift.
                support_gate = torch.ones_like(s_emb[:, :1])
                support_reliability = torch.ones_like(support_freshness)
            support_ctx = support_ctx * support_gate
            support_proto = support_proto * support_gate * torch.sigmoid(self.support_proto_scale)
            support_freshness = support_freshness * support_reliability
            support_temporal_focus = support_temporal_focus * support_reliability
        else:
            support_ctx = torch.zeros_like(s_emb)
            support_proto = torch.zeros_like(s_emb)
            support_freshness = torch.zeros_like(s_emb[:, 0])
            support_temporal_focus = torch.zeros_like(s_emb[:, 0])

        if self.use_freq:
            freq_rel, freq_reg = self.freq_adapter(r_emb, t_emb)
        else:
            freq_rel = torch.zeros_like(r_emb)
            freq_reg = torch.zeros((), device=r_emb.device)
        meta_input = support_ctx + support_proto + freq_rel + t_emb
        meta_rel = self.rel_gru(meta_input, r_emb)

        gate = self.entity_gate(torch.cat([s_emb, hist_ctx, t_emb], dim=-1))
        dyn_s = gate * hist_ctx + (1.0 - gate) * s_emb
        semantic = self.semantic_adapter(torch.cat([support_ctx, support_proto], dim=-1))
        return dyn_s, meta_rel + semantic, hist_ctx, support_ctx, t_emb, freq_reg, support_freshness, support_temporal_focus

    @staticmethod
    def _top2_margin(logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] < 2:
            return torch.zeros(logits.shape[0], device=logits.device)
        top2 = torch.topk(logits, k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        return torch.tanh(margin / 5.0).clamp(min=0.0, max=1.0)

    @staticmethod
    def _masked_top2_margin(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = scores.masked_fill(~mask, -1e9)
        active = mask.sum(dim=1)
        safe = masked.clone()
        safe[active < 2] = 0.0
        top2 = torch.topk(safe, k=2, dim=-1).values
        margin = torch.where(active >= 2, top2[:, 0] - top2[:, 1], torch.zeros_like(top2[:, 0]))
        return torch.tanh(margin / 5.0).clamp(min=0.0, max=1.0)

    def _build_evidence(
        self,
        generate_logits: torch.Tensor,
        copy_logits: torch.Tensor,
        path_scores: torch.Tensor,
        rule_mask: torch.Tensor,
        dyn_s: torch.Tensor,
        support_ctx: torch.Tensor,
        features: Dict[str, torch.Tensor],
        support_freshness: torch.Tensor,
        support_temporal_focus: torch.Tensor,
    ) -> torch.Tensor:
        device = generate_logits.device
        copy_positive = copy_logits > 0
        copy_strength = torch.tanh(copy_logits.clamp_min(0).amax(dim=1) / 3.0)
        copy_density = (copy_positive.float().sum(dim=1) / max(1, self.history_len)).clamp(max=1.0)
        rule_confidence = features.get("rule_confidence", torch.zeros(generate_logits.shape[0])).to(device).clamp(0.0, 1.0)
        rule_struct_ratio = features.get("rule_struct_ratio", torch.zeros(generate_logits.shape[0])).to(device).clamp(0.0, 1.0)
        rule_available = rule_mask.any(dim=1).float()
        rule_margin = self._masked_top2_margin(path_scores, rule_mask) if rule_mask.any() else torch.zeros_like(rule_confidence)
        generate_margin = self._top2_margin(generate_logits)
        support_alignment = F.cosine_similarity(dyn_s, support_ctx, dim=-1).nan_to_num(0.0)
        support_alignment = ((support_alignment + 1.0) / 2.0).clamp(0.0, 1.0)
        history_available = features["hist_mask"].to(device).any(dim=1).float()
        history_freshness = features.get("history_freshness", torch.zeros(generate_logits.shape[0])).to(device).clamp(0.0, 1.0)
        history_density = features.get("history_density", torch.zeros(generate_logits.shape[0])).to(device).clamp(0.0, 1.0)
        relation_time_scale = features.get("relation_time_scale", torch.zeros(generate_logits.shape[0])).to(device).clamp(0.0, 1.0)
        rel_copy_strength = features.get("rel_copy_strength", torch.zeros(generate_logits.shape[0])).to(device).clamp(0.0, 1.0)
        rel_copy_available = (rel_copy_strength > 0).float()
        support_freshness = support_freshness.to(device).clamp(0.0, 1.0)
        support_temporal_focus = support_temporal_focus.to(device).clamp(0.0, 1.0)
        if not self.use_temporal_calibration:
            history_freshness = torch.zeros_like(history_freshness)
            history_density = torch.zeros_like(history_density)
            relation_time_scale = torch.zeros_like(relation_time_scale)
            support_freshness = torch.zeros_like(support_freshness)
            support_temporal_focus = torch.zeros_like(support_temporal_focus)
        return torch.stack(
            [
                generate_margin,
                copy_strength,
                copy_density,
                rule_confidence,
                rule_struct_ratio,
                rule_available,
                rule_margin,
                support_alignment,
                history_available,
                history_freshness,
                history_density,
                relation_time_scale,
                support_freshness,
                support_temporal_focus,
                rel_copy_available,
                rel_copy_strength,
            ],
            dim=-1,
        )

    @staticmethod
    def _bounded_score(values: torch.Tensor, scale: float) -> torch.Tensor:
        return torch.tanh(values / scale).clamp(-1.0, 1.0)

    @staticmethod
    def _topk_ids(scores: torch.Tensor, k: int) -> torch.Tensor:
        safe_scores = torch.where(torch.isfinite(scores), scores, torch.full_like(scores, -1e9))
        return torch.topk(safe_scores, k=min(k, safe_scores.shape[1]), dim=1).indices

    def _apply_candidate_rerank(
        self,
        generate_logits: torch.Tensor,
        raw_copy_logits: torch.Tensor,
        copy_mask: torch.Tensor,
        raw_rel_copy_logits: torch.Tensor,
        rel_copy_mask: torch.Tensor,
        rule_logits: torch.Tensor,
        rule_mask: torch.Tensor,
        relation_prior_logits: torch.Tensor | None,
        dyn_s: torch.Tensor,
        rel: torch.Tensor,
        hist_ctx: torch.Tensor,
        support_ctx: torch.Tensor,
        time_emb: torch.Tensor,
        support_freshness: torch.Tensor,
        support_temporal_focus: torch.Tensor,
        features: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k = min(32, self.num_entities)
        device = generate_logits.device
        batch = generate_logits.shape[0]
        zeros = torch.zeros_like(generate_logits)

        safe_copy_sparse = torch.where(copy_mask, raw_copy_logits, torch.full_like(raw_copy_logits, -1e9))
        safe_rel_sparse = torch.where(rel_copy_mask, raw_rel_copy_logits, torch.full_like(raw_rel_copy_logits, -1e9))
        rule_available_dense = rule_logits > -1e8
        safe_rule_sparse = torch.where(rule_available_dense, rule_logits, torch.full_like(rule_logits, -1e9))
        if relation_prior_logits is None:
            relation_prior_logits = zeros
        relation_prior_logits = relation_prior_logits.to(device)
        safe_prior = torch.where(relation_prior_logits > 0, relation_prior_logits, torch.full_like(relation_prior_logits, -1e9))

        gen_ids = self._topk_ids(generate_logits, k)
        copy_ids = self._topk_ids(safe_copy_sparse, k)
        rel_ids = self._topk_ids(safe_rel_sparse, k)
        rule_ids = self._topk_ids(safe_rule_sparse, k)
        prior_ids = self._topk_ids(safe_prior, k)
        explicit_rule_ids = features["rule_candidates"].to(device)
        candidate_ids = torch.cat([gen_ids, copy_ids, rel_ids, rule_ids, prior_ids, explicit_rule_ids], dim=1)

        gen_valid = torch.ones_like(gen_ids, dtype=torch.bool)
        copy_valid = copy_mask.gather(1, copy_ids)
        rel_valid = rel_copy_mask.gather(1, rel_ids)
        rule_valid = rule_available_dense.gather(1, rule_ids)
        prior_valid = (relation_prior_logits > 0).gather(1, prior_ids)
        explicit_rule_valid = rule_mask.to(device)
        candidate_valid = torch.cat([gen_valid, copy_valid, rel_valid, rule_valid, prior_valid, explicit_rule_valid], dim=1)

        def gather(values: torch.Tensor) -> torch.Tensor:
            return values.gather(1, candidate_ids)

        copy_avail = copy_mask.gather(1, candidate_ids).float()
        rel_avail = rel_copy_mask.gather(1, candidate_ids).float()
        rule_avail = rule_available_dense.gather(1, candidate_ids).float()
        query_scalars = torch.stack(
            [
                features.get("history_freshness", torch.zeros(batch, device=device)).to(device).clamp(0.0, 1.0),
                features.get("history_density", torch.zeros(batch, device=device)).to(device).clamp(0.0, 1.0),
                features.get("relation_time_scale", torch.zeros(batch, device=device)).to(device).clamp(0.0, 1.0),
                support_freshness.to(device).clamp(0.0, 1.0),
                support_temporal_focus.to(device).clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        query_scalars = query_scalars.unsqueeze(1).expand(batch, candidate_ids.shape[1], 5)
        scalar_features = torch.cat(
            [
                self._bounded_score(gather(generate_logits), 8.0).unsqueeze(-1),
                self._bounded_score(torch.where(copy_avail.bool(), gather(raw_copy_logits), torch.zeros_like(gather(raw_copy_logits))), 3.0).unsqueeze(-1),
                copy_avail.unsqueeze(-1),
                self._bounded_score(torch.where(rel_avail.bool(), gather(raw_rel_copy_logits), torch.zeros_like(gather(raw_rel_copy_logits))), 3.0).unsqueeze(-1),
                rel_avail.unsqueeze(-1),
                self._bounded_score(torch.where(rule_avail.bool(), gather(rule_logits), torch.zeros_like(gather(rule_logits))), 3.0).unsqueeze(-1),
                rule_avail.unsqueeze(-1),
                self._bounded_score(gather(relation_prior_logits), 3.0).unsqueeze(-1),
                query_scalars,
            ],
            dim=-1,
        )
        candidate_emb = self.entity(candidate_ids)
        candidate_bonus = self.candidate_reranker(
            dyn_s,
            rel,
            hist_ctx,
            support_ctx,
            time_emb,
            candidate_emb,
            scalar_features,
        )
        candidate_bonus = candidate_bonus * candidate_valid.float()
        candidate_bonus = (F.softplus(self.candidate_rerank_scale) * candidate_bonus).clamp(max=4.0)
        dense_bonus = torch.zeros_like(generate_logits)
        dense_bonus = dense_bonus.scatter_reduce(1, candidate_ids, candidate_bonus, reduce="amax", include_self=True)
        return generate_logits + dense_bonus, dense_bonus

    def _prepare_llm_features(
        self,
        features: Dict[str, torch.Tensor],
        batch: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_ids = features.get("llm_candidate_ids")
        if candidate_ids is None:
            empty_ids = torch.empty(batch, 0, dtype=torch.long, device=device)
            empty_values = torch.empty(batch, 0, device=device)
            empty_mask = torch.empty(batch, 0, dtype=torch.bool, device=device)
            return empty_ids, empty_mask, empty_values, empty_values, empty_values, empty_values, empty_values
        candidate_ids = candidate_ids.to(device=device, dtype=torch.long)
        if candidate_ids.ndim != 2 or candidate_ids.shape[0] != batch:
            raise ValueError("llm_candidate_ids must have shape [batch, candidates]")
        candidate_ids = candidate_ids[:, : self.llm_max_candidates]
        candidate_mask = features.get("llm_candidate_mask")
        if candidate_mask is None:
            candidate_mask = torch.zeros_like(candidate_ids, dtype=torch.bool)
        else:
            candidate_mask = candidate_mask.to(device=device, dtype=torch.bool)[:, : candidate_ids.shape[1]]
        candidate_mask = candidate_mask & candidate_ids.ge(0) & candidate_ids.lt(self.num_entities)
        candidate_ids = candidate_ids.clamp(0, self.num_entities - 1)

        def values(name: str) -> torch.Tensor:
            value = features.get(name)
            if value is None:
                return torch.zeros_like(candidate_ids, dtype=torch.float)
            value = value.to(device=device, dtype=torch.float)[:, : candidate_ids.shape[1]]
            if value.shape != candidate_ids.shape:
                raise ValueError(f"{name} must match llm_candidate_ids shape")
            return value.clamp(0.0, 1.0)

        confidence = values("llm_confidence")
        if self.llm_disable_confidence:
            confidence = torch.zeros_like(confidence)
        return (
            candidate_ids,
            candidate_mask,
            confidence,
            values("llm_mapping_score"),
            values("llm_template_agreement"),
            values("llm_temporal_score"),
            values("llm_rank_prior"),
        )

    def _apply_llm_side_evidence(
        self,
        base_log_probs: torch.Tensor,
        features: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = base_log_probs.shape[0]
        prepared = self._prepare_llm_features(features, batch, base_log_probs.device)
        candidate_ids, candidate_mask, confidence, mapping, template, temporal, rank_prior = prepared
        if self.llm_runtime_mode == "off" or candidate_ids.numel() == 0:
            return base_log_probs, torch.zeros_like(base_log_probs), candidate_ids, candidate_mask, rank_prior
        adjusted, dense_bonus = self.llm_sidecar(
            base_log_probs,
            candidate_ids,
            candidate_mask,
            confidence,
            mapping,
            template,
            temporal,
            rank_prior,
            self.llm_runtime_mode,
        )
        return adjusted, dense_bonus, candidate_ids, candidate_mask, rank_prior

    def _apply_alterego_tournament(
        self,
        base_log_probs: torch.Tensor,
        raw_copy_logits: torch.Tensor,
        copy_mask: torch.Tensor,
        raw_rel_copy_logits: torch.Tensor,
        rel_copy_mask: torch.Tensor,
        rule_logits: torch.Tensor,
        relation_prior_logits: torch.Tensor | None,
        dyn_s: torch.Tensor,
        rel: torch.Tensor,
        hist_ctx: torch.Tensor,
        support_ctx: torch.Tensor,
        time_emb: torch.Tensor,
        support_freshness: torch.Tensor,
        support_temporal_focus: torch.Tensor,
        features: Dict[str, torch.Tensor],
        llm_candidate_ids: torch.Tensor | None = None,
        llm_candidate_mask: torch.Tensor | None = None,
        llm_rank_prior: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = base_log_probs.device
        batch = base_log_probs.shape[0]
        k = min(self.alterego_candidate_k, self.num_entities)
        selection_scores = base_log_probs.detach()
        if (
            self.llm_runtime_mode != "off"
            and llm_candidate_ids is not None
            and llm_candidate_mask is not None
            and llm_candidate_ids.numel() > 0
        ):
            selection_scores = selection_scores.clone()
            safe_ids = llm_candidate_ids.clamp(0, self.num_entities - 1)
            row_peak = selection_scores.amax(dim=1, keepdim=True)
            if llm_rank_prior is None:
                llm_rank_prior = torch.zeros_like(safe_ids, dtype=selection_scores.dtype)
            forced_scores = row_peak + 1.0 + 0.01 * llm_rank_prior.to(selection_scores.device)
            forced_scores = torch.where(
                llm_candidate_mask,
                forced_scores,
                torch.full_like(forced_scores, torch.finfo(forced_scores.dtype).min),
            )
            selection_scores = selection_scores.scatter_reduce(
                1,
                safe_ids,
                forced_scores,
                reduce="amax",
                include_self=True,
            )
        candidate_ids = self._topk_ids(selection_scores, k)
        candidate_mask = torch.isfinite(base_log_probs).gather(1, candidate_ids)
        if relation_prior_logits is None:
            relation_prior_logits = torch.zeros_like(base_log_probs)
        else:
            relation_prior_logits = relation_prior_logits.to(device)
        rule_available = rule_logits > -1e8

        def gather(values: torch.Tensor) -> torch.Tensor:
            return values.gather(1, candidate_ids)

        copy_available = copy_mask.gather(1, candidate_ids).float()
        rel_copy_available = rel_copy_mask.gather(1, candidate_ids).float()
        rule_available_k = rule_available.gather(1, candidate_ids).float()
        query_scalars = torch.stack(
            [
                features.get("history_freshness", torch.zeros(batch, device=device)).to(device).clamp(0.0, 1.0),
                features.get("history_density", torch.zeros(batch, device=device)).to(device).clamp(0.0, 1.0),
                features.get("relation_time_scale", torch.zeros(batch, device=device)).to(device).clamp(0.0, 1.0),
                support_freshness.to(device).clamp(0.0, 1.0),
                support_temporal_focus.to(device).clamp(0.0, 1.0),
            ],
            dim=-1,
        ).unsqueeze(1).expand(batch, k, 5)
        scalar_features = torch.cat(
            [
                self._bounded_score(gather(base_log_probs), 8.0).unsqueeze(-1),
                self._bounded_score(
                    torch.where(copy_available.bool(), gather(raw_copy_logits), torch.zeros_like(gather(raw_copy_logits))),
                    3.0,
                ).unsqueeze(-1),
                copy_available.unsqueeze(-1),
                self._bounded_score(
                    torch.where(
                        rel_copy_available.bool(), gather(raw_rel_copy_logits), torch.zeros_like(gather(raw_rel_copy_logits)),
                    ),
                    3.0,
                ).unsqueeze(-1),
                rel_copy_available.unsqueeze(-1),
                self._bounded_score(
                    torch.where(rule_available_k.bool(), gather(rule_logits), torch.zeros_like(gather(rule_logits))),
                    3.0,
                ).unsqueeze(-1),
                rule_available_k.unsqueeze(-1),
                self._bounded_score(gather(relation_prior_logits), 3.0).unsqueeze(-1),
                query_scalars,
            ],
            dim=-1,
        )
        candidate_delta, pairwise_scores, _ = self.alterego_tournament(
            dyn_s,
            rel,
            hist_ctx,
            support_ctx,
            time_emb,
            self.entity(candidate_ids),
            scalar_features,
            candidate_mask,
        )
        dense_delta = torch.zeros_like(base_log_probs).scatter(1, candidate_ids, candidate_delta)
        candidate_scores = gather(base_log_probs).detach() + candidate_delta
        return (
            base_log_probs + dense_delta,
            dense_delta,
            candidate_ids,
            candidate_scores,
            candidate_mask,
            pairwise_scores,
        )

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        features: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        dyn_s, rel, hist_ctx, support_ctx, time_emb, freq_reg, support_freshness, support_temporal_focus = self.encode_query(query, support, features)
        candidates = self.entity.weight

        conv_logits = self.conv_decoder(dyn_s, rel, candidates)
        geo_logits = self.geo_decoder(dyn_s, rel, time_emb, candidates)
        trans_query = self.trans_proj(torch.cat([dyn_s, rel, time_emb, support_ctx], dim=-1))
        trans_logits = trans_query @ candidates.t()

        decoder_scores = self.decoder_weights
        if not self.use_geo:
            decoder_scores = decoder_scores.clone()
            decoder_scores[1] = -1e9
        decoder_w = torch.softmax(decoder_scores, dim=0)
        generate_logits = decoder_w[0] * conv_logits + decoder_w[1] * geo_logits + decoder_w[2] * trans_logits
        relation_prior_logits = features.get("relation_prior_logits")
        if relation_prior_logits is not None:
            relation_prior_logits = relation_prior_logits.to(generate_logits.device)
            relation_prior_logits = relation_prior_logits / relation_prior_logits.amax(dim=1, keepdim=True).clamp_min(1.0)
            generate_logits = generate_logits + F.softplus(self.relation_prior_scale) * relation_prior_logits
        raw_copy_logits = features["copy_logits"].to(generate_logits.device)
        copy_mask = raw_copy_logits > 0
        copy_available = copy_mask.any(dim=1)
        copy_logits = torch.where(copy_mask, raw_copy_logits, torch.full_like(raw_copy_logits, -1e9))
        raw_rel_copy_logits = features.get("rel_copy_logits")
        if raw_rel_copy_logits is None:
            raw_rel_copy_logits = torch.zeros_like(generate_logits)
        else:
            raw_rel_copy_logits = raw_rel_copy_logits.to(generate_logits.device)
        rel_copy_mask = raw_rel_copy_logits > 0
        rel_copy_available = rel_copy_mask.any(dim=1)
        rel_copy_logits = torch.full_like(generate_logits, -1e9)
        rel_copy_logits = torch.where(
            rel_copy_mask,
            raw_rel_copy_logits + F.softplus(self.relation_copy_logit_boost),
            rel_copy_logits,
        )
        rule_logits = torch.full_like(generate_logits, -1e9)
        rule_mask = features["rule_mask"].to(generate_logits.device)
        path_scores = torch.full(
            (generate_logits.shape[0], features["rule_candidates"].shape[1]),
            -1e9,
            device=generate_logits.device,
        )
        if rule_mask.any():
            rule_candidates = features["rule_candidates"].to(generate_logits.device)
            candidate_emb = self.entity(rule_candidates)
            path_scores = self.rule_scorer(
                dyn_s,
                rel,
                time_emb,
                candidate_emb,
                self.relation(features["rule_r1"].to(generate_logits.device)),
                self.relation(features["rule_r2"].to(generate_logits.device)),
                features["rule_dt1"].to(generate_logits.device),
                features["rule_dt2"].to(generate_logits.device),
                features["rule_type"].to(generate_logits.device),
                features["rule_prior"].to(generate_logits.device),
                rule_mask,
            )
            rule_logits = rule_logits.scatter_reduce(1, rule_candidates, path_scores, reduce="amax", include_self=True)

        candidate_rerank_bonus = torch.zeros_like(generate_logits)
        if self.use_candidate_rerank:
            generate_logits, candidate_rerank_bonus = self._apply_candidate_rerank(
                generate_logits,
                raw_copy_logits,
                copy_mask,
                raw_rel_copy_logits,
                rel_copy_mask,
                rule_logits,
                rule_mask,
                relation_prior_logits,
                dyn_s,
                rel,
                hist_ctx,
                support_ctx,
                time_emb,
                support_freshness,
                support_temporal_focus,
                features,
            )

        oracle_logit = self.oracle(torch.cat([dyn_s, rel, hist_ctx], dim=-1)).squeeze(-1)
        evidence = self._build_evidence(
            generate_logits,
            copy_logits,
            path_scores,
            rule_mask,
            dyn_s,
            support_ctx,
            features,
            support_freshness,
            support_temporal_focus,
        )
        rule_reliability_logit = self.rule_reliability(dyn_s, rel, hist_ctx, support_ctx, evidence)
        mix_input = torch.cat([dyn_s, rel, hist_ctx, support_ctx], dim=-1)
        gate_logits = self.residual_gate(mix_input, evidence) if self.use_router else self.fallback_gate(mix_input)
        sparse_correctness_logits = self.sparse_correctness(torch.cat([mix_input, evidence], dim=-1))
        gate_logits = gate_logits + sparse_correctness_logits
        # Keep unmasked logits for gain supervision. Availability only zeroes an
        # impossible residual and never forces another residual to activate.
        router_logits = gate_logits.clone()
        residual_available = torch.stack([copy_available, rel_copy_available, rule_mask.any(dim=1)], dim=-1)
        gate_logits = gate_logits.masked_fill(~residual_available, -20.0)
        if not self.use_copy:
            gate_logits[:, 0] = -20.0
        if not self.use_rel_copy:
            gate_logits[:, 1] = -20.0
        if not self.use_rule:
            gate_logits[:, 2] = -20.0
        else:
            rule_confidence = features.get("rule_confidence")
            if rule_confidence is not None:
                rule_confidence = rule_confidence.to(generate_logits.device).clamp(min=1e-3, max=1.0)
                learned_rule_confidence = torch.sigmoid(rule_reliability_logit).clamp(min=1e-3, max=1.0)
                calibrated_rule_confidence = 0.25 + 0.75 * rule_confidence * learned_rule_confidence
                gate_logits[:, 2] = gate_logits[:, 2] + torch.logit(calibrated_rule_confidence.clamp(1e-3, 1 - 1e-3))
            gate_logits[~rule_mask.any(dim=1), 2] = -20.0
        # Every sparse branch is a bounded correction, never a replacement for
        # the full-support generate backbone. Scaling after sigmoid preserves a
        # smooth confidence signal while preventing an always-on expert.
        residual_gates = self.max_residual_gate * torch.sigmoid(gate_logits)
        if self.training and self.expert_dropout > 0:
            drop_mask = torch.zeros_like(residual_gates, dtype=torch.bool)
            if self.use_copy:
                drop_mask[:, 0] = torch.rand(residual_gates.shape[0], device=residual_gates.device) < self.expert_dropout
            if self.use_rel_copy:
                drop_mask[:, 1] = torch.rand(residual_gates.shape[0], device=residual_gates.device) < (0.5 * self.expert_dropout)
            if self.use_rule:
                drop_mask[:, 2] = torch.rand(residual_gates.shape[0], device=residual_gates.device) < self.expert_dropout
            residual_gates = residual_gates.masked_fill(drop_mask, 0.0)
        diagnostic_weights = torch.cat([torch.ones_like(residual_gates[:, :1]), residual_gates], dim=-1)
        legacy_weights = diagnostic_weights / diagnostic_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        residual_strength = residual_gates.sum(dim=-1, keepdim=True)
        residual_mix = residual_gates / residual_strength.clamp_min(1e-8)
        mixture_alpha = self.max_expert_mass * (1.0 - torch.exp(-residual_strength))
        mixture_alpha = torch.where(residual_strength > 1e-8, mixture_alpha, torch.zeros_like(mixture_alpha))
        base_preserving_weights = torch.cat(
            [1.0 - mixture_alpha, mixture_alpha * residual_mix], dim=-1
        )
        expert_weights = base_preserving_weights if self.fusion_mode == "base_preserving_mixture" else legacy_weights
        raw_expert_logits = torch.stack(
            [
                generate_logits,
                copy_logits,
                rel_copy_logits,
                rule_logits,
            ],
            dim=1,
        )
        temperature = self.expert_log_temperature.exp().clamp(0.25, 4.0).view(1, self.num_experts, 1)
        expert_logps = F.log_softmax(raw_expert_logits / temperature, dim=-1)
        # CCRF: generate is the full-support base distribution.  Sparse experts
        # add bounded candidate bonuses instead of stealing probability mass
        # from every entity when their candidate set misses the target.
        if self.fusion_mode in {"probability_mixture", "base_preserving_mixture"}:
            log_weights = torch.log(expert_weights.clamp_min(1e-8)).unsqueeze(-1)
            log_probs = torch.logsumexp(log_weights + expert_logps, dim=1)
        else:
            sparse_masks = torch.stack([copy_mask, rel_copy_mask, rule_logits > -1e8], dim=1)
            sparse_probs = expert_logps[:, 1:].exp() * sparse_masks.float()
            peak = sparse_probs.amax(dim=-1, keepdim=True).clamp_min(1e-8)
            normalized_residual = sparse_probs / peak
            residual_weights = residual_gates.unsqueeze(-1)
            residual_scales = F.softplus(self.residual_scale).view(1, 3, 1)
            residual_bonus = (residual_weights * residual_scales * normalized_residual).sum(dim=1)
            log_probs = F.log_softmax(raw_expert_logits[:, 0] / temperature[:, 0] + residual_bonus, dim=-1)

        llm_bonus = torch.zeros_like(log_probs)
        llm_candidate_ids = torch.empty(log_probs.shape[0], 0, dtype=torch.long, device=log_probs.device)
        llm_candidate_mask = torch.empty(log_probs.shape[0], 0, dtype=torch.bool, device=log_probs.device)
        llm_rank_prior = log_probs.new_empty(log_probs.shape[0], 0)
        if self.llm_runtime_mode != "off":
            llm_logits, llm_bonus, llm_candidate_ids, llm_candidate_mask, llm_rank_prior = (
                self._apply_llm_side_evidence(log_probs, features)
            )
            normalized_llm = F.log_softmax(llm_logits, dim=-1)
            unchanged_llm = llm_bonus.abs().amax(dim=1, keepdim=True).eq(0)
            log_probs = torch.where(unchanged_llm, log_probs, normalized_llm)

        alterego_bonus = torch.zeros_like(log_probs)
        alterego_candidate_ids = torch.empty(log_probs.shape[0], 0, dtype=torch.long, device=log_probs.device)
        alterego_candidate_scores = log_probs.new_empty(log_probs.shape[0], 0)
        alterego_candidate_mask = torch.empty(log_probs.shape[0], 0, dtype=torch.bool, device=log_probs.device)
        alterego_pairwise_scores = log_probs.new_empty(log_probs.shape[0], 0, 0)
        if self.alterego_runtime_enabled:
            (
                alterego_logits,
                alterego_bonus,
                alterego_candidate_ids,
                alterego_candidate_scores,
                alterego_candidate_mask,
                alterego_pairwise_scores,
            ) = self._apply_alterego_tournament(
                log_probs,
                raw_copy_logits,
                copy_mask,
                raw_rel_copy_logits,
                rel_copy_mask,
                rule_logits,
                relation_prior_logits,
                dyn_s,
                rel,
                hist_ctx,
                support_ctx,
                time_emb,
                support_freshness,
                support_temporal_focus,
                features,
                llm_candidate_ids=llm_candidate_ids,
                llm_candidate_mask=llm_candidate_mask,
                llm_rank_prior=llm_rank_prior,
            )
            normalized_alterego = F.log_softmax(alterego_logits, dim=-1)
            unchanged = alterego_bonus.abs().amax(dim=1, keepdim=True).eq(0)
            log_probs = torch.where(unchanged, log_probs, normalized_alterego)

        aux = {
            "oracle_logit": oracle_logit,
            "expert_weights": expert_weights,
            "expert_logps": expert_logps,
            "expert_temperature": temperature.squeeze(0).squeeze(-1),
            "expert_available": torch.stack(
                [torch.ones_like(copy_available), copy_available, rel_copy_available, rule_mask.any(dim=1)], dim=-1
            ),
            "expert_logits": torch.cat([torch.zeros_like(gate_logits[:, :1]), gate_logits], dim=-1),
            "router_logits": router_logits,
            "residual_gate_logits": gate_logits,
            "residual_gates": residual_gates,
            "mixture_alpha": mixture_alpha.squeeze(-1),
            "evidence": evidence,
            "rule_reliability_logit": rule_reliability_logit,
            "sparse_correctness_logits": sparse_correctness_logits,
            "path_scores": path_scores,
            "freq_reg": freq_reg,
            "query_repr": trans_query,
            "support_freshness": support_freshness,
            "support_temporal_focus": support_temporal_focus,
            "candidate_rerank_bonus": candidate_rerank_bonus,
            "llm_bonus": llm_bonus,
            "llm_candidate_ids": llm_candidate_ids,
            "llm_candidate_mask": llm_candidate_mask,
            "llm_cache_hit": features.get(
                "llm_cache_hit", torch.zeros(query.shape[0], dtype=torch.bool, device=query.device)
            ).to(generate_logits.device),
            "alterego_bonus": alterego_bonus,
            "alterego_candidate_ids": alterego_candidate_ids,
            "alterego_candidate_scores": alterego_candidate_scores,
            "alterego_candidate_mask": alterego_candidate_mask,
            "alterego_pairwise_scores": alterego_pairwise_scores,
            "rel_copy_strength": features.get("rel_copy_strength", torch.zeros(query.shape[0], device=query.device)).to(generate_logits.device),
        }
        return log_probs, aux
