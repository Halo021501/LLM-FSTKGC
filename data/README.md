# Data preparation

The ICEWS14 and ICEWS18 datasets are not redistributed in this repository.
Obtain each dataset from a source whose terms permit your intended use, then
place it under `data/ICEWS14/` or `data/ICEWS18/`.

Each dataset directory must contain:

```text
train.txt
valid.txt
test.txt
stat.txt
entity2id.txt
relation2id.txt
```

The three split files contain whitespace- or tab-separated quadruples in the
order `subject relation object timestamp`. Extra columns are ignored. Numeric
entity and relation identifiers are supported. `stat.txt` begins with the
entity and direct-relation counts; an optional third value may record the
number of timestamps.

`entity2id.txt` and `relation2id.txt` provide the public names used to build and
ground STLP prompts. Their identifiers must agree with the split files. All
support and history selection is performed by the code with the strict
`timestamp < query_timestamp` rule; do not merge validation or test events into
the training split.

The small `data/toy/` dataset is included only for the CPU smoke test:

```bash
./scripts/run_toy_smoke.sh
```
