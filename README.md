# LLM-FSTKGC

LLM-FSTKGC is a PyTorch implementation for few-shot temporal knowledge graph
completion. Given a timestamped query with one missing entity, the program
ranks the complete entity vocabulary and reports filtered MRR and Hits@K for
both tail and head prediction.

The repository contains the model, training and evaluation code, an optional
Alibaba Cloud Qwen candidate-cache pipeline, a small toy dataset, and offline
tests. It does not contain ICEWS datasets, generated Qwen responses or caches,
trained checkpoints, experiment logs, or manuscript result tables.

## Implemented model

The graph ranking path uses four score sources:

- `generate`: a learned full-vocabulary temporal decoder;
- `copy`: candidates from the known entity's causal history;
- `rel_copy`: candidates previously observed with the oriented relation;
- `rule`: candidates retrieved by temporal relation paths.

The implementation also contains continuous-time encoding, causal multi-scale
history encoding, causal snapshot-graph encoding, few-shot support encoding,
candidate reranking, and an antisymmetric pairwise candidate tournament. The
final output remains a score over every entity. Sparse sources only adjust
eligible candidates.

For a query at timestamp `t`, history and support use facts with timestamps
strictly smaller than `t`. For support budget `K`, the support selector takes
the last `max(8*K, 32)` eligible same-relation rows and then orders that bounded
pool by decreasing timestamp and subject match. Head prediction is evaluated
through inverse relations using the known object.

### Optional Qwen evidence

The released provider path is:

| Setting | Value |
| --- | --- |
| Provider identifier | `aliyun_qwen_realtime` |
| Service | Alibaba Cloud Model Studio / DashScope |
| Model alias | `qwen-flash` |
| Region | `cn-beijing` |
| API base URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| Temperature | `0` |
| Thinking mode | disabled |
| Response format | JSON object |

Qwen is used only before model execution to generate sparse candidate records.
Training, validation, testing, and model `forward` read local JSONL caches and
do not call the provider.

The available `--llm-mode` values are:

| Mode | Runtime behavior |
| --- | --- |
| `off` | Ignores all LLM tensors and uses the graph ranking path only. |
| `candidate` | Adds mapped Qwen entity IDs to the tournament candidate bank; it adds no direct Qwen score bonus. |
| `score` | Adds a bounded sparse bonus using confidence, mapping quality, name-pattern agreement, and candidate rank prior. |
| `rationale` | Uses the `score` features and the returned temporal-consistency value. |

The Qwen evidence is not a fifth full-vocabulary expert. Its bonus is
non-negative, sparse, and capped by `--llm-max-delta` (default `0.35`).
Unresolved or ambiguous entity names have no effect on the model ranking.

Cache keys are built without the hidden answer. They contain the dataset
fingerprint, split, prediction direction, known entity, oriented relation,
timestamp, shot count, seed, history protocol, causal-context digests, and
prompt version. Label-like fields are rejected. An entity equal to the hidden
answer can still occur naturally in a strictly earlier public fact; the answer
is not consulted to remove such occurrences.

## Repository layout

```text
.
├── data/toy/                 # Small CPU smoke-test dataset
├── scripts/                  # Cache, evaluation, collection, and verification tools
├── src/
│   ├── model.py              # Model modules and ranking path
│   ├── train.py              # Training, validation, testing, and metrics
│   ├── data.py               # Dataset loading and causal feature construction
│   ├── stlp.py               # Qwen prompt construction and entity-name mapping
│   ├── llm_cache.py          # Target-blind JSONL cache validation and loading
│   └── aliyun_qwen_*.py      # Realtime transport and artifact I/O
├── tests/                    # Offline behavior and invariant tests
├── train.py                  # Main command-line entry point
├── requirements.txt
├── SOURCE_MANIFEST.sha256
└── ALIYUN_QWEN_REALTIME_PROVENANCE.json
```

`ALIYUN_QWEN_REALTIME_PROVENANCE.json` is machine-readable input used when the
cache collector records provider metadata. It is not a separate project guide.

## Requirements

- Python 3.10 or newer;
- PyTorch 2.0 or newer;
- Linux shell utilities for the supplied `.sh` scripts;
- an NVIDIA GPU only for CUDA runs; the toy smoke test can run on CPU.

Create an environment and install the declared dependency:

```bash
git clone https://github.com/Halo021501/LLM-FSTKGC.git
cd LLM-FSTKGC

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick verification

The release verification is offline and does not contact Alibaba Cloud:

```bash
PYTHON_BIN="$(pwd)/.venv/bin/python" bash scripts/verify_release.sh
```

Run a small end-to-end CPU training and evaluation job:

```bash
PYTHON_BIN="$(pwd)/.venv/bin/python" bash scripts/run_toy_smoke.sh
```

Outputs are written to `runs/toy_smoke/`. The toy data only checks that the
code path runs; its metrics are not research results.

## Dataset preparation

ICEWS data are not redistributed. Obtain ICEWS14 or ICEWS18 from a source whose
terms permit your use, then create `data/ICEWS14/` or `data/ICEWS18/` with:

```text
train.txt
valid.txt
test.txt
stat.txt
entity2id.txt
relation2id.txt
```

Each split contains whitespace- or tab-separated rows in this order:

```text
subject  relation  object  timestamp
```

Extra columns are ignored. `stat.txt` starts with the number of entities and
the number of direct relations; a third value may record the number of
timestamps. Numeric IDs and textual split values are supported by the dataset
loader.

The two mapping files provide the public entity and relation names used by the
Qwen prompt and entity mapper. A row may place the integer ID first or last.
The IDs must agree with the split files and cover the complete corresponding
vocabulary. Do not merge validation or test events into the training split.

## Graph-only training

The following command trains without LLM evidence:

```bash
python train.py \
  --data-dir data/ICEWS14 \
  --output-dir runs/icews14_graph_seed42 \
  --device cuda:0 \
  --seed 42 \
  --shot 5 \
  --epochs 40 \
  --episodes-per-epoch 300 \
  --warmup-epochs 5 \
  --history-len 10 \
  --dim 256 \
  --channels 64 \
  --dropout 0.2 \
  --llm-mode off
```

Each run writes `best.pt`, `metrics.json`, and `run_meta.json` under its output
directory. Use `python train.py --help` for all training and ablation options.

## Building Qwen caches

### 1. Configure local credentials

```bash
cp .env.aliyun_qwen_realtime.example .env.aliyun_qwen_realtime
chmod 600 .env.aliyun_qwen_realtime
```

Set `DASHSCOPE_API_KEY` only in that gitignored file. Initially leave both
confirmation values as `NO`:

```text
CONFIRM_ALIYUN_QWEN_DATA_UPLOAD=NO
CONFIRM_ALIYUN_QWEN_PAID_REALTIME=NO
```

### 2. Build request plans offline

Active training needs a validation cache and a test cache. The example below
prepares shot 5 for both splits without a credential or network request:

```bash
REALTIME_SCOPE=full REALTIME_SPLIT=valid REALTIME_SHOTS="5" \
  bash scripts/generate_aliyun_qwen_realtime_caches.sh prepare

REALTIME_SCOPE=full REALTIME_SPLIT=test REALTIME_SHOTS="5" \
  bash scripts/generate_aliyun_qwen_realtime_caches.sh prepare
```

Review the request plans under
`runs/aliyun_qwen_request_plans/qwen-flash/standard/full/`. The public payload
contains entity names, relation names, the query timestamp and direction, and
strictly earlier support/history facts. The hidden answer and API key are not
serialized.

Estimate token cost from the reviewed plans:

```bash
REALTIME_SCOPE=full REALTIME_SPLIT=valid REALTIME_SHOTS="5" \
  bash scripts/generate_aliyun_qwen_realtime_caches.sh estimate

REALTIME_SCOPE=full REALTIME_SPLIT=test REALTIME_SHOTS="5" \
  bash scripts/generate_aliyun_qwen_realtime_caches.sh estimate
```

### 3. Run the provider requests

Check the current Alibaba Cloud data policy and pricing yourself. When you
accept the upload and cost, change both confirmation values in the private env
file to `YES`, then start each split:

```bash
REALTIME_SCOPE=full REALTIME_SPLIT=valid REALTIME_SHOTS="5" \
  bash scripts/generate_aliyun_qwen_realtime_caches.sh start

REALTIME_SCOPE=full REALTIME_SPLIT=test REALTIME_SHOTS="5" \
  bash scripts/generate_aliyun_qwen_realtime_caches.sh start
```

Read local progress without making a provider request:

```bash
REALTIME_SCOPE=full REALTIME_SPLIT=valid REALTIME_SHOTS="5" \
  bash scripts/generate_aliyun_qwen_realtime_caches.sh status

REALTIME_SCOPE=full REALTIME_SPLIT=test REALTIME_SHOTS="5" \
  bash scripts/generate_aliyun_qwen_realtime_caches.sh status
```

After successful completion, formal cache files and schema-v2 metadata
sidecars are stored as:

```text
cache/standard_rolling_history/aliyun_qwen_realtime/qwen-flash/standard/
├── valid_s5.jsonl
├── valid_s5.jsonl.meta.json
├── test_s5.jsonl
└── test_s5.jsonl.meta.json
```

The runner bounds concurrency and RPM/TPM usage, records retries, supports
resume by request ID, and stores explicit provider abstentions. A validated
provider abstention becomes an empty candidate set. It is not replaced by a
synthetic response.

To generate other supported shot counts, set `REALTIME_SHOTS` to an ordered,
unique subset of `1 3 5 10`.

## Training with cached Qwen evidence

```bash
python train.py \
  --data-dir data/ICEWS14 \
  --output-dir runs/icews14_qwen_score_s5_seed42 \
  --device cuda:0 \
  --seed 42 \
  --shot 5 \
  --epochs 40 \
  --episodes-per-epoch 300 \
  --warmup-epochs 5 \
  --history-len 10 \
  --dim 256 \
  --channels 64 \
  --dropout 0.2 \
  --llm-mode score \
  --llm-valid-cache cache/standard_rolling_history/aliyun_qwen_realtime/qwen-flash/standard/valid_s5.jsonl \
  --llm-test-cache cache/standard_rolling_history/aliyun_qwen_realtime/qwen-flash/standard/test_s5.jsonl
```

An active training run requires complete validation and test caches whose
metadata match the dataset fingerprint, split, shot count, and history
protocol. `--allow-partial-llm-cache` is intended only for smoke tests and
debugging.

## Evaluation utilities

Evaluate an existing compatible checkpoint with zero additional training:

```bash
python train.py \
  --data-dir data/ICEWS14 \
  --output-dir runs/checkpoint_score_s5_seed42 \
  --device cuda:0 \
  --seed 42 \
  --shot 5 \
  --epochs 0 \
  --warmup-epochs 0 \
  --episodes-per-epoch 0 \
  --llm-mode score \
  --init-from-v5 checkpoints/alterego_v5/main_s5_seed42/best.pt \
  --llm-test-cache cache/standard_rolling_history/aliyun_qwen_realtime/qwen-flash/standard/test_s5.jsonl
```

The repository also includes multi-run wrappers:

```bash
PYTHON_BIN="$(pwd)/.venv/bin/python" \
DATA_DIR="$(pwd)/data/ICEWS14" \
CHECKPOINT_ROOT="$(pwd)/checkpoints/alterego_v5" \
CACHE_DIR="$(pwd)/cache/standard_rolling_history/aliyun_qwen_realtime/qwen-flash/standard" \
GPUS="0" \
  bash scripts/launch_aliyun_qwen_realtime_frozen_v5_matrix.sh
```

The default matrix covers shots `1, 3, 5, 10`, seeds `42, 43, 44`, and the
three active LLM modes. It requires user-supplied data, caches, and compatible
checkpoints. The wrapper is an execution utility; the repository does not
include or claim results from that matrix.

The LLM-only evaluator reports sparse-candidate diagnostics and is not a
full-vocabulary model replacement:

```bash
python scripts/stlp_evaluate_llm_only.py \
  --data-dir data/ICEWS14 \
  --cache cache/standard_rolling_history/aliyun_qwen_realtime/qwen-flash/standard/test_s5.jsonl \
  --split test \
  --shot 5 \
  --ranking-mode confidence
```

## Prompt controls

The request-plan generator exposes four independent controls through
environment variables:

- `OMIT_SUPPORT=YES`;
- `OMIT_HISTORY=YES`;
- `PERMUTE_SUPPORT_ORDER=YES`;
- `REPLACE_ENTITY_NAMES=YES`.

Use separate request-plan, raw-response, and cache directories for every
condition. The current prompt/key version is
`stlp-aliyun-qwen-realtime-v1`; caches from another prompt version have
different query keys.

## Reproducibility and release limits

- `SOURCE_MANIFEST.sha256` verifies the files in this source snapshot.
- A run should preserve its command, source-manifest hash, dataset
  fingerprint, cache and metadata hashes, checkpoint hash, seed, and generated
  `run_meta.json`.
- `qwen-flash` is a provider-managed alias. The repository cannot pin its exact
  model-weight revision, so regenerating semantic responses is not guaranteed
  to be byte-identical. Preserving validated caches allows the local graph
  ranking path to be rerun against the same candidate evidence.
- No formal ICEWS dataset, credential, generated provider artifact, trained
  checkpoint, experiment log, or paper result is bundled.
- No software license file is included in this snapshot.

## Authors and contact

- Chunhao Chen — `20243006949@hainanu.edu.cn` —
  [ORCID 0009-0004-2023-3976](https://orcid.org/0009-0004-2023-3976)
- Siling Feng (corresponding author) — `fengsiling2008@163.com` —
  [ORCID 0000-0002-8627-2028](https://orcid.org/0000-0002-8627-2028)

College of Information and Communication Engineering, Hainan University,
Haikou, China.

## Funding

This work was supported by the National Natural Science Foundation of China
under Grant Nos. 62466016 and 62241202, and the Hainan Provincial Natural
Science Foundation of China under Grant No. 626MS0094.
