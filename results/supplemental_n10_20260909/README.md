# Supplemental n=10 results

This directory is the minimal public result record for the completed
September 2026 supplemental experiments. It contains aggregate metrics,
seed-level metrics, and the prespecified MRR paired statistics. Metrics were
recomputed from complete query exports before this compact record was made.

## Design

- Datasets: ICEWS14 and YAGO.
- Support budgets: K=5 and K=10.
- Seeds: 42--51; the statistical unit is a seed-matched model-run pair.
- Each selected checkpoint was evaluated in semantic-off and rationale modes
  on the same complete test queries.
- Checkpoints were selected by semantic-off validation tie-average MRR.
- Final runs used `probability_mixture` fusion,
  `standard_rolling_history`, and `--llm-residual-control hard`.
- ICEWS14 has 14,742 directional test queries; YAGO has 40,052.

The four primary gate-off comparisons all had 10/10 positive seed-level MRR
differences. Mean rationale-minus-off MRR changes were +0.551 pp (ICEWS14
K=5), +0.580 pp (ICEWS14 K=10), +0.293 pp (YAGO K=5), and +0.248 pp (YAGO
K=10). Their exact two-sided paired signed-rank p-values were 0.001953 and
Holm-adjusted p-values were 0.007812 within the four-comparison family.

The optional ICEWS14 K=10 support gate reduced mean MRR relative to gate-off
by 0.484 pp in semantic-off mode and 0.395 pp in rationale mode. This result is
reported rather than hidden; it does not establish a mechanism or a universal
effect of the gate.

## Files

- `aggregate_metrics.csv`: mean and sample standard deviation for MRR and
  Hits@1/3/10 (both 0--1 and x100 columns).
- `metrics_by_seed.csv`: the 100 model/mode/seed metric rows used for the
  aggregate table. Local server paths were removed for release.
- `mrr_paired_statistics.csv`: paired effects, bootstrap intervals, exact
  tests, Holm adjustments, and leave-one-seed-out ranges.

The exact experiment matrix and fixed training parameters are in
`../../configs/supplemental_n10.json`. The directory intentionally excludes
datasets, checkpoints, semantic caches, per-query exports, credentials,
provider responses, logs, and internal scheduler paths.

These records support the tested configurations only. They are not a claim of
state-of-the-art performance or a guarantee that every query improves.
