# Design: v1.7.0alterego_v5

## Hypothesis

Top-96 candidates play a query-conditioned low-rank pairwise game. The payoff matrix is exactly antisymmetric and aggregated as soft Copeland wins, producing zero-sum candidate corrections without attention or evidence transport.

The algorithm is intentionally isolated from the other four alterego branches;
combining mechanisms before validation would make gains and regressions impossible
to attribute.

## Protected integration

1. Build the unchanged v1.6.2advant generate/copy/relation-copy/rule distribution.
2. Compute the alterego sidecar without reading the query object label.
3. Apply a signed residual capped at 0.5 after four-expert fusion.
4. Initialize the final residual control to exactly zero.
5. Disable the sidecar during the proven base warmup; train it in the joint stage.

Complexity: O(B K² R), default K=96 and rank=32; never materializes B×K×K×D.

Auxiliary objective: 0.05 natural-recall pairwise margin loss. Targets are never inserted into a candidate set.
The ablation switch is `--disable-alterego-tournament`.

## Required acceptance checks

- enabled-at-initialization equals disabled output in evaluation mode;
- probability distribution remains finite and normalized;
- changing `query[:,2]` cannot change prediction;
- the expert axis remains exactly four;
- branch-specific mathematical invariants pass;
- CPU toy training completes before any GPU/full-data run;
- formal comparison uses validation-selected checkpoints and paired seeds.

## Originality statement

The primitive itself is not claimed as globally new. The independently designed
part is its target-blind, zero-initialized protected integration into the
ACR-TFCM-GS temporal-KG expert system and the branch-specific constraints above.
