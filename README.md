# LLM-FSTKGC

This repository contains the public LLM-FSTKGC code snapshot. LLM-FSTKGC
extends the v1.7.0alterego_v5 temporal graph ranker with a target-blind
semantic-temporal language prior (STLP). The STLP contributes sparse
candidate-side evidence; it is not a fifth expert and never replaces the
full-vocabulary graph distribution.

The released LLM route is Alibaba Cloud Model Studio
`aliyun_qwen_realtime` with the `qwen-flash` model alias. Candidate
generation is an explicit preprocessing stage that writes auditable JSONL
caches. Training and evaluation read only frozen local caches and make no
provider calls.

This release describes the behavior of the included source code. It does not
bundle ICEWS data, provider responses, STLP caches, model checkpoints or paper
result tables, and it does not claim that the optional example matrix below
reproduces a particular manuscript table.

## Project information

- Repository: [Halo021501/LLM-FSTKGC](https://github.com/Halo021501/LLM-FSTKGC)
- Authors, in manuscript order: Chunhao Chen; Siling Feng
- First author: Chunhao Chen
- Corresponding author: Siling Feng
- Affiliation: College of Information and Communication Engineering, Hainan University, Haikou, China
- E-mail:
  - Chunhao Chen: `20243006949@hainanu.edu.cn`
  - Siling Feng: `fengsiling2008@163.com`
- ORCID iDs:
  - Chunhao Chen: [0009-0004-2023-3976](https://orcid.org/0009-0004-2023-3976)
  - Siling Feng: [0000-0002-8627-2028](https://orcid.org/0000-0002-8627-2028)

## Funding

This research was supported by the National Natural Science Foundation of China
under Grant Nos. 62466016 and 62241202, and the Hainan Provincial Natural
Science Foundation of China under Grant No. 626MS0094.

## Requirements

Use Python 3.10 or newer with PyTorch 2.0 or newer:

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

No provider SDK is required. The cache generator uses Python's standard-library
HTTPS client.

## Data preparation

ICEWS data are not redistributed. Obtain ICEWS14 or ICEWS18 from an authorized
source and follow `data/README.md` for the required split and name-mapping
files. The included `data/toy/` directory is only for the CPU smoke test.

## Method boundary

- generate, copy, relation-copy, and rule remain the graph model's four experts;
- `llm-mode off` ignores all LLM tensors and preserves the parent-v5 path;
- candidate mode admits mapped STLP candidates to the existing tournament;
- score and rationale modes add a sparse non-negative residual capped by
  `--llm-max-delta` (default 0.35);
- cache lookup is target-blind and uses only public query/context metadata;
- support and history contain only strictly earlier facts. The implemented
  support selector first takes the last `max(8*K, 32)` eligible same-relation
  rows, then orders that bounded window by decreasing timestamp and, within a
  timestamp, by whether the known entity matches the fact subject;
- when the graph path has no eligible same-relation fact, it uses a label-free
  structural sentinel. The STLP request planner instead serializes an empty
  support list in that case;
- unresolved or ambiguous candidate strings are discarded rather than assigned
  an unverified entity id;
- provider calls occur only while constructing caches, never inside training,
  validation, testing, or model forward.

See `DESIGN_v1.7.0alterego_v5_llm.md` for the detailed data flow and
experiment boundary.

## Alibaba Cloud Qwen cache workflow

Copy the configuration template and keep the real file private:

    cp .env.aliyun_qwen_realtime.example .env.aliyun_qwen_realtime
    chmod 600 .env.aliyun_qwen_realtime

Fill `DASHSCOPE_API_KEY` locally. Keep both confirmation values at `NO`
while reviewing the prompt and request plans.

Build deterministic request plans without a credential or network call:

    REALTIME_SCOPE=full REALTIME_SHOTS="1 3 5 10"       ./scripts/generate_aliyun_qwen_realtime_caches.sh prepare

Review the generated files under:

    runs/aliyun_qwen_request_plans/qwen-flash/standard/full/

Then estimate token cost offline:

    REALTIME_SCOPE=full REALTIME_SHOTS="1 3 5 10"       ./scripts/generate_aliyun_qwen_realtime_caches.sh estimate

Only after reviewing the public prompt content, current provider policy and
price, set these values in the gitignored `.env.aliyun_qwen_realtime` file:

    CONFIRM_ALIYUN_QWEN_DATA_UPLOAD=YES
    CONFIRM_ALIYUN_QWEN_PAID_REALTIME=YES

Start or resume the provider calls:

    REALTIME_SCOPE=full REALTIME_SHOTS="1 3 5 10"       ./scripts/generate_aliyun_qwen_realtime_caches.sh start

Read local progress without a credential or network call:

    REALTIME_SCOPE=full REALTIME_SHOTS="1 3 5 10"       ./scripts/generate_aliyun_qwen_realtime_caches.sh status

After raw responses are complete, collection validates response schema,
candidate ranges, custom IDs, prompt hashes and dataset identity before writing
formal caches:

    cache/standard_rolling_history/aliyun_qwen_realtime/
      qwen-flash/standard/{test_s1,test_s3,test_s5,test_s10}.jsonl

The runner uses bounded concurrency, RPM/TPM limits, retry auditing, fsynced
append-only response records, resume-safe custom IDs and explicit provider
abstentions. It never fabricates a successful response. A validated provider
abstention becomes an empty candidate set, so the graph model receives no
LLM-side bonus for that query.

The exact endpoint, decoding controls and reproducibility limitation of the
provider-managed moving alias are recorded in
`ALIYUN_QWEN_REALTIME_PROVENANCE.json`.

## Training and evaluation boundaries

`train.py` supports both training from scratch and evaluation from an existing
v5 checkpoint. An active LLM mode requires a complete test cache. Training with
validation-based checkpoint selection also requires a complete validation
cache; the cache can affect validation and checkpoint selection, but no provider
call is made by the training process.

Example active-mode training command:

    python train.py \
      --data-dir data/ICEWS14 \
      --output-dir runs/example_score_s5_seed42 \
      --device cuda:0 --shot 5 --seed 42 \
      --epochs 40 --episodes-per-epoch 300 --warmup-epochs 5 \
      --history-len 10 --dim 256 --channels 64 --dropout 0.2 \
      --llm-mode score \
      --llm-valid-cache cache/standard_rolling_history/aliyun_qwen_realtime/qwen-flash/standard/valid_s5.jsonl \
      --llm-test-cache cache/standard_rolling_history/aliyun_qwen_realtime/qwen-flash/standard/test_s5.jsonl

The command is a source-code usage example, not a packaged result claim. Dataset
files, name maps and cache metadata sidecars must be supplied as documented.

## Optional frozen-parent code-path check

The repository also retains an optional zero-training-epoch evaluation recipe
for user-supplied validation-selected parent-v5 checkpoints. The provider is not
contacted by these scripts.

The bundled multi-GPU example covers shots 1, 3, 5 and 10, seeds 42, 43 and 44,
and candidate/score/rationale modes:

    PYTHON_BIN=/path/to/python       ./scripts/launch_aliyun_qwen_realtime_frozen_v5_matrix.sh

Check progress and collect the completed matrix:

    ./scripts/check_aliyun_qwen_realtime_frozen_v5_matrix.sh
    python scripts/collect_aliyun_qwen_realtime_frozen_v5.py

A simpler sequential shot-5/shot-10 runner is also provided:

    ./scripts/evaluate_v5_llm_checkpoint_matrix.sh

Set `CHECKPOINT_ROOT`, `CACHE_DIR`, `DATA_DIR` and `DEVICE` when the
artifacts are outside their default locations. Active modes require both the
cache and its schema-v2 metadata sidecar; missing artifacts fail closed.

This optional three-seed frozen-parent recipe is an implemented utility, not a
declaration of the experiment matrix, seed count or training protocol used by
any paper. Do not present its output as a manuscript result without a separate,
artifact-backed provenance record.

The LLM-only evaluator is a sparse-candidate diagnostic, not a full-vocabulary
replacement:

    python scripts/stlp_evaluate_llm_only.py       --data-dir data/ICEWS14       --cache cache/standard_rolling_history/aliyun_qwen_realtime/qwen-flash/standard/test_s5.jsonl       --split test --shot 5 --ranking-mode confidence

## Prompt audits

The request planner supports the prompt perturbations reported in the
supplementary material. Each condition must use a separate request-plan, raw
response and cache directory:

- `OMIT_HISTORY=YES`;
- `PERMUTE_SUPPORT_ORDER=YES`;
- `REPLACE_ENTITY_NAMES=YES`.

The hidden answer is absent in every condition. These controls are descriptive:
`qwen-flash` is a provider-managed moving alias and the perturbation caches
were generated on different dates.

## Verification

Release verification is offline and never calls Alibaba Cloud:

    PYTHON_BIN=/path/to/python ./scripts/verify_release.sh

It checks the source manifest, shell syntax, JSON provenance, Python imports,
the realtime transport/collection tests and the parent-model integration
invariants.

## Reproducibility note

Persisted caches reproduce graph ranking when their hashes and metadata are
preserved. They do not guarantee byte-identical semantic regeneration because
`qwen-flash` is not weight-pinned and the exact provider model revision is
unavailable. Preserve request hashes, response IDs, returned model strings,
decoding controls, token usage, retries, timestamps and cache sidecars.

The source manifest verifies this public snapshot only. A run should record the
manifest hash, dataset fingerprints, complete command line, cache hashes and
checkpoint hash. Results produced from a different source snapshot or prompt
version must be identified as a different provenance set.
