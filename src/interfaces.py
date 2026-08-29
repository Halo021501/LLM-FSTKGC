from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

import torch


@dataclass
class BasePrediction:
    """Backbone-agnostic full-support prediction contract.

    The semantic fields are optional in v1.6 and reserved for later language
    model experts.  Keeping the contract stable prevents future backbones from
    being hard-wired into the fusion layer.
    """

    logits: torch.Tensor
    confidence: Optional[torch.Tensor] = None
    uncertainty: Optional[torch.Tensor] = None
    provenance: Optional[Dict[str, str]] = None


@dataclass
class ResidualEvidence:
    """Sparse causal evidence contributed on top of a full-support base."""

    residual: torch.Tensor
    candidate_mask: torch.Tensor
    availability: torch.Tensor
    confidence: Optional[torch.Tensor] = None
    evidence_timestamp: Optional[torch.Tensor] = None
    provenance: Optional[Dict[str, str]] = None


class BaseBackbone(Protocol):
    def forward(self, query: torch.Tensor, causal_context: Dict[str, torch.Tensor]) -> BasePrediction:
        ...


class ResidualExpert(Protocol):
    def forward(self, query: torch.Tensor, causal_context: Dict[str, torch.Tensor]) -> ResidualEvidence:
        ...
