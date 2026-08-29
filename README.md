# LLM-FSTKGC

This repository contains the LLM-FSTKGC implementation, developed as an
independent extension of the v1.7.0alterego_v5 graph ranker. The language model supplies a target-blind
semantic-temporal prior (STLP) as sparse candidate-side evidence. It is not a
fifth expert and never replaces the original full-entity model distribution.

The task-book primary provider is a locally deployed
`Qwen2.5-7B-Instruct-AWQ`. No language-model request is made during training or
evaluation. Candidate generation is a separate, explicit preprocessing step
that writes an auditable JSONL cache. The default training mode is llm-mode off.
DeepSeek remains an optional comparison/fallback provider. Alibaba Cloud Qwen
Batch is an additional, separately cached low-cost external comparison route.
Neither external route replaces the formal local-provider default.

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

The graph model requires Python and PyTorch 2.0 or newer:

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

The optional local-Qwen provider uses the separately pinned environment under
`deploy/`; it is not imported during graph-model training or evaluation.

## Data preparation

ICEWS data are not redistributed. Obtain ICEWS14 or ICEWS18 from an authorized
source and follow `data/README.md` for the required split and name-mapping
files. The included `data/toy/` directory is only for the CPU smoke test.

## Integrity guarantees

- generate, copy, relation-copy, and rule remain the only four experts;
- the complete v5 fusion and tournament code path remains active;
- llm-mode off ignores every LLM tensor and is bit-exact with the parent v5;
- candidate mode only admits mapped candidates to the v5 tournament;
- score and rationale modes add a sparse non-negative residual capped by
  llm-max-delta, default 0.35;
- cache lookup uses only known entity id, oriented relation id, and timestamp;
- prompt/cache metadata rejects target/gold/hidden-object fields;
- current-snapshot and future facts never enter support or history;
- cache dataset fingerprint, prompt hash, mapping diagnostics, latency, and token
  counts are recorded.

The bit-exact parent check and leakage tests are in
tests/test_v170alterego_v5_llm_invariants.py.

Release verification is one command and never calls an external API:

    ./scripts/verify_release.sh

## Local Qwen deployment (formal task-book route)

The LLM runtime is intentionally isolated from `regcn`:

    conda create -n qwen_local python=3.10 -y
    conda run -n qwen_local python -m pip install \
      -r deploy/qwen_local_requirements.txt
    ./scripts/download_local_qwen_model.sh

The pinned environment uses vLLM 0.6.3.post1 with CUDA 12.1 binaries. Model
weights are stored in the gitignored
`models/Qwen2.5-7B-Instruct-AWQ` directory by default. The download script writes a
SHA-256 model manifest after validating that the checkpoint is AWQ. For
reproducibility, its default download is pinned to official repository revision
`b25037543e9394b818fdfca67ab2a00ecc7dd641` rather than the moving `main`
branch; the resolved repository and revision are written to `MODEL_SOURCE.txt`.
The requirements also pin the original `pyairports` 2.1.1 artifact by URL and
SHA-256 because Outlines 0.0.46's unconstrained dependency can otherwise resolve
to the incompatible 0.0.1 namespace currently published under that name.
Setuptools is held at 75.6.0 because pyairports 2.1.1 imports the legacy
`pkg_resources` module removed from newer setuptools releases.
`deploy/qwen_local_requirements.txt` is the small installation specification;
`deploy/qwen_local_environment.lock.txt` records all 119 packages from the
successfully tested environment for exact diagnosis and reproduction. Prefer
the small specification for a clean install, then compare its resolved package
set with the lock file before serving.

Copy the local configuration only if an override is needed:

    cp .env.qwen_local.example .env.qwen_local

Set `LOCAL_QWEN_GPU_ID` to a completely idle physical GPU, then start and check
the loopback-only service:

    ./scripts/start_local_qwen_server.sh
    ./scripts/check_local_qwen_server.sh

The startup script refuses a GPU with more than 512 MiB already allocated by
default. Shared operation must be explicitly enabled with `ALLOW_SHARED_GPU=YES`
and still enforces `LOCAL_QWEN_MIN_FREE_MIB`; use a reduced vLLM memory fraction,
shorter context, lower concurrency, and eager mode in that case. It binds only
to `127.0.0.1`, runs with Hugging Face offline mode after download, and survives
terminal disconnection through a detached process group. Stop it with:

    ./scripts/stop_local_qwen_server.sh

The project-owned serving entrypoint also works around vLLM 0.6.3's upstream
pre-bound-socket behavior, which otherwise widens the requested loopback bind to
`0.0.0.0`. Always confirm the effective listener with `ss -ltnp 'sport = :8000'`
after rebuilding the serving environment.
For the pinned single-GPU executor it also replaces PyTorch's wildcard-bound
TCPStore rendezvous with a permission-restricted local FileStore. This removes
the otherwise unnecessary high-port listener; the override deliberately fails
closed on a different vLLM version or multi-GPU executor. Remaining one-rank
Gloo/NCCL communication is restricted to the loopback interface.
The local configuration disables vLLM frontend multiprocessing so guided JSON
decoding errors remain in the serving process and cannot silently kill a
separate RPC engine.
Candidate responses use an explicit JSON Schema with LM Format Enforcer 0.10.6;
the client re-parses and canonicalizes every response before it can reach a
cache file.
Each new cache has schema-v2 metadata containing the exact model revision,
model-manifest hash, runtime-lock hash, serving kernel, initial command, UTC
start time, host, and physical GPU id. `--resume` refuses any cache whose
scientific metadata differs, including shot, provider, dataset, history
protocol, prompt ablation, or model provenance.
Active formal LLM modes refuse caches without this schema-v2 sidecar; its path,
SHA-256, provider provenance, and generation audit are copied into each
experiment's `run_meta.json`.
On the RTX 4060 Ti, the same official AWQ weights are executed with vLLM's
`awq_marlin` kernel; set `LOCAL_QWEN_QUANTIZATION=awq` only as a compatibility
fallback.

Before the formal generation, run a tiny target-blind smoke cache:

    python scripts/stlp_generate_candidates.py \
      --data-dir data/ICEWS14 --split test --shot 5 \
      --provider local_qwen --max-tokens 512 --limit 4 \
      --output runs/local_qwen_smoke/test_s5.jsonl

For the full test cache, set `CONFIRM_LOCAL_QWEN_GENERATION=YES` in the local
configuration and run:

    ./scripts/generate_local_qwen_caches.sh

Its default output is
`cache/standard_rolling_history/qwen2.5-7b-awq/{test_s5,test_s10}.jsonl`.
Point an evaluation matrix to it with
`CACHE_DIR=cache/standard_rolling_history/qwen2.5-7b-awq`.

For the persistent multi-GPU route, keep the existing GPU-2 endpoint and run:

    CONFIRM_LOCAL_QWEN_GENERATION=YES ./scripts/start_dynamic_qwen_full_generation.sh

This creates 256 deterministic shards per shot and four independent request
workers per Qwen server. GPU 4 is the explicitly approved initial shared card.
Cards 0, 1, 3, 5, and 6 are preconfigured on loopback ports 8100--8106, but a
service is started only after the card has zero compute processes, at least
12,000 MiB free, no more than 5% utilization, and passes two consecutive
30-second checks. A failed shard is recorded and never automatically retried.
The detached controller survives terminal disconnection; inspect it without an
interactive monitoring session using:

    ./scripts/check_dynamic_qwen_pool.sh

To opportunistically share a card that the authoritative controller cannot
auto-admit because another compute process is already present, first inspect a
single explicit state directory without changing it:

    python \
      scripts/opportunistic_shared_qwen_supervisor.py \
      --dry-run --state-dir logs/FORMAL_STATE --gpu-id 5

After reviewing that output, the long-lived form additionally requires
`--confirm-opportunistic-supervisor`. It defaults to 11,000 MiB GPU free and
10,000 MiB host `MemAvailable` at admission. The existing guard retires only
this project's server and sidecar after three pressure checks below 1,024 MiB
GPU free or 2,048 MiB host available; the supervisor then waits and may rejoin.
The admission thresholds can be lowered explicitly with CLI flags or
`QWEN_OPPORTUNISTIC_START_GPU_FREE_MIB` and
`QWEN_OPPORTUNISTIC_START_HOST_AVAILABLE_MIB`. A fully idle card is always left
to the authoritative controller, avoiding duplicate auto-start ownership.

The controller atomically merges and validates exactly 13,179 target-blind
locators per shot, writes a part-hash manifest, runs the LLM-only diagnostics,
and stops only the extra Qwen services that it started. The pre-existing GPU-2
service is left running.

## DeepSeek configuration (optional fallback)

No key is stored in this project. Prepare a local file only when candidate
generation is authorized:

    cp .env.deepseek.example .env.deepseek

Fill DEEPSEEK_API_KEY in .env.deepseek. The template currently uses
https://api.deepseek.com and deepseek-v4-flash; both remain overridable because
provider model names can change. API generation additionally requires
CONFIRM_DEEPSEEK_API_CALLS=YES and the explicit execute-api flag.

Before a paid run, recheck the official DeepSeek JSON-output guide at
https://api-docs.deepseek.com/guides/json_mode/ and current model/pricing page at
https://api-docs.deepseek.com/quick_start/pricing.

Review cost and the data-upload policy before enabling this step. The supplied
ICEWS names, relations, and past facts will be sent to the configured provider;
the hidden target is never sent.

## Alibaba Cloud Qwen Batch (optional low-cost external route)

The Batch route is an asynchronous preprocessing provider. It does not call an
API during v5 training, validation, testing, or model forward, and it does not
change the local-Qwen provider or its existing cache. Its fixed paper default is
`qwen3.7-flash-2026-07-15`. The moving `qwen-flash` alias can be selected only
as an explicitly named rolling-cost comparison; its potentially lower current
input price does not compensate for the loss of a fixed model version.

Prepare a local configuration, but leave both network gates disabled while
building and checking the request plan:

    cp .env.aliyun_qwen_batch.example .env.aliyun_qwen_batch
    chmod 600 .env.aliyun_qwen_batch
    ./scripts/generate_aliyun_qwen_batch_caches.sh prepare-pilot
    ./scripts/generate_aliyun_qwen_batch_caches.sh estimate

The pilot defaults to 250 target-blind queries per shot, or 500 test requests
across the default shot 5 and shot 10 pair. Set `BATCH_SHOTS="1 3"` for the
additional low-shot robustness pair. Set `PILOT_LIMIT=100` for a smaller 200-request
quality gate, or use `prepare-smoke` for 25 queries per shot (50 test requests
total). Pilot collection stays below `runs/` and cannot overwrite a formal
cache. Keep the same `BATCH_SCOPE`, model, `INCLUDE_VALID`, and prompt-ablation
settings for every later action; the default scope is `pilot`.

Only after inspecting request counts, estimated token volume, current official
pricing, and the data policy should the key and both confirmations be filled in
the gitignored file:

    DASHSCOPE_API_KEY=...
    CONFIRM_ALIYUN_QWEN_DATA_UPLOAD=YES
    CONFIRM_ALIYUN_QWEN_PAID_BATCH=YES

The shell action and the Python `--execute-api` flag are additional deliberate
actions, but neither bypasses those two gates. Submit, inspect, download, and
collect the pilot with:

    ./scripts/generate_aliyun_qwen_batch_caches.sh submit
    ./scripts/generate_aliyun_qwen_batch_caches.sh status
    ./scripts/generate_aliyun_qwen_batch_caches.sh download
    ./scripts/generate_aliyun_qwen_batch_caches.sh collect

An in-progress Batch can be stopped without deleting its local audit trail:

    BATCH_SCOPE=smoke ALIYUN_QWEN_BATCH_MODEL=qwen-flash \
      ./scripts/generate_aliyun_qwen_batch_caches.sh cancel

`prepare`, `estimate`, `collect`, and `prepare-retry` are offline operations and
their Python children do not receive `DASHSCOPE_API_KEY`. Every external
operation, including status and download, requires both confirmations. Failed,
missing, or invalid outputs must be isolated with `prepare-retry` and submitted
as a separate retry job; formal collection must reject partial output.

For the full test cache, use a distinct full scope for every step:

    BATCH_SCOPE=full ./scripts/generate_aliyun_qwen_batch_caches.sh prepare-full
    BATCH_SCOPE=full ./scripts/generate_aliyun_qwen_batch_caches.sh estimate
    BATCH_SCOPE=full ./scripts/generate_aliyun_qwen_batch_caches.sh submit
    BATCH_SCOPE=full ./scripts/generate_aliyun_qwen_batch_caches.sh status
    BATCH_SCOPE=full ./scripts/generate_aliyun_qwen_batch_caches.sh download
    BATCH_SCOPE=full ./scripts/generate_aliyun_qwen_batch_caches.sh collect

The formal outputs are provider- and model-specific:

    cache/standard_rolling_history/aliyun_qwen_batch/
      qwen3.7-flash-2026-07-15/standard/{test_s5,test_s10}.jsonl

Set `INCLUDE_VALID=YES` before preparation only when the from-scratch matrix has
been authorized; keep it enabled through collection. Prompt ablations set
`OMIT_SUPPORT=YES` or `OMIT_HISTORY=YES` and automatically use separate
artifact directories. To try the rolling-price option, explicitly set
`ALIYUN_QWEN_BATCH_MODEL=qwen-flash`; never merge its records with the dated
model's cache.

Batch request files necessarily contain the target-blind prompt that is sent to
the provider. They are stored only under the gitignored
`runs/aliyun_qwen_batch/` staging tree, while the canonical cache retains only
query metadata and a prompt hash. The uploaded public context contains ICEWS
entity/relation names, the public timestamp/direction, and strictly earlier
facts; it never contains the hidden answer. Request/result matching uses the
target-blind `custom_id`, not output order. The request uses JSON-object output,
disables thinking, and deliberately omits a fixed output-token cap so a valid
JSON object is not truncated; all returned fields are still validated locally.

Alibaba Cloud uploaded files persist remotely until explicitly deleted. After
the local output, error file, cache, metadata, counts, and hashes have been
verified, remote cleanup requires a third destructive-operation confirmation:

    CONFIRM_ALIYUN_QWEN_REMOTE_DELETE=YES \
      BATCH_SCOPE=full ./scripts/generate_aliyun_qwen_batch_caches.sh cleanup-remote

Batch completion is asynchronous and is not a guaranteed per-query latency
measurement. Report job wall time, requests per second, token usage, retry/error
rates, and cost separately from local-Qwen request latency. Provider identity,
fixed versus rolling model policy, data boundary, and confirmation gates are
recorded in `ALIYUN_QWEN_BATCH_PROVENANCE.json`.

## Alibaba Cloud Qwen realtime concurrency (time-critical external route)

The realtime route consumes the same immutable, target-blind request and index
plans reviewed for Batch, but sends each request to the normal OpenAI-compatible
chat-completions endpoint. It writes to a separate
`runs/aliyun_qwen_realtime/` tree and identifies every cache record as
`aliyun_qwen_realtime`; Batch, local Qwen, and realtime outputs must never be
merged as one provider condition.

The wrapper can reuse the protected `.env.aliyun_qwen_batch` key file. Realtime
billing has its own explicit gate and therefore also requires:

    CONFIRM_ALIYUN_QWEN_PAID_REALTIME=YES

Start or resume the reviewed 50-query smoke plan under a detached controller:

    CONFIRM_ALIYUN_QWEN_PAID_REALTIME=YES REALTIME_SCOPE=smoke \
      ./scripts/generate_aliyun_qwen_realtime_caches.sh start

Read progress locally without a key or network call:

    REALTIME_SCOPE=smoke ./scripts/generate_aliyun_qwen_realtime_caches.sh status

The aggregate profiles are smoke=8 workers/240 RPM/300,000 TPM,
pilot=16/480/600,000, and full=32/960/1,200,000. Each is divided equally
between the shots selected by `REALTIME_SHOTS` and remains far below the documented Beijing
floating-`qwen-flash` limit. A sliding-window limiter reserves 1,200 tokens per
attempt. HTTP 429 honors `Retry-After` and sets a shared cooldown; 408/429/5xx
and transport failures use bounded exponential backoff with deterministic
jitter; authentication and invalid-request errors stop the run. The provider's
query-local `data_inspection_failed` response is handled separately: that one
query is tried at most three times, then recorded in the append-only
`realtime_abstentions.jsonl` audit without stopping its peers. Collection does
not fabricate a provider response; it emits an explicitly diagnosed empty LLM
candidate set, so the v5 score receives no LLM-side bonus for that query.
Valid results and abstentions are fsynced one at a time, so `--resume` sends
only unresolved custom IDs. Five attempts is the general default ceiling and
every retry is recorded. The runner keeps at most one future per worker in
flight instead of pre-submitting the full split. A sanitized unexpected worker
exception stops further submission while the bounded live window is drained
and persisted. The state file receives a main-thread heartbeat every 30 seconds
even when no request completes, and separately preserves the last-progress
timestamp. This prevents an active worker pool from being mistaken for a
healthy result consumer. The urllib timeout remains a per-blocking-operation
timeout rather than a guaranteed whole-request wall-clock deadline.

After all selected shots complete, the wrapper automatically performs strict offline
collection. Smoke/pilot caches remain under the run tree; full outputs use:

    cache/standard_rolling_history/aliyun_qwen_realtime/
      qwen-flash/standard/{test_s1,test_s3,test_s5,test_s10}.jsonl

The task-book 5/10-shot controller keeps the legacy `controller.pid/log` names.
The user-requested 1/3-shot extension runs independently, so both pairs can be
active without sharing a PID or log:

    ALIYUN_QWEN_BATCH_MODEL=qwen-flash BATCH_SCOPE=full BATCH_SHOTS="1 3" \
      ./scripts/generate_aliyun_qwen_batch_caches.sh prepare-full
    REALTIME_SCOPE=full REALTIME_SHOTS="1 3" \
      ./scripts/generate_aliyun_qwen_realtime_caches.sh estimate
    CONFIRM_ALIYUN_QWEN_PAID_REALTIME=YES REALTIME_SCOPE=full \
      REALTIME_SHOTS="1 3" ./scripts/generate_aliyun_qwen_realtime_caches.sh start

This added pair uses `controller_1_3.pid/log`; status is read offline with the
same `REALTIME_SCOPE=full REALTIME_SHOTS="1 3"` selection.

For a full run, first prepare and review the matching `qwen-flash` full plans,
then estimate them before setting `REALTIME_SCOPE=full`:

    REALTIME_SCOPE=full ./scripts/generate_aliyun_qwen_realtime_caches.sh estimate

Recheck price and predicted token volume before starting: realtime inference is
billed at the normal rate, and the moving model
alias does not provide an immutable weight revision. Sidecars preserve request
and result hashes, returned model string, timestamps, per-request latency,
token usage, retry count, abstention count/reason/hash, concurrency/rate
controls, and estimated list-price cost. The boundary is declared in
`ALIYUN_QWEN_REALTIME_PROVENANCE.json`.

## Build target-blind caches

The original task-book protocol requires separate shot-5 and shot-10 files; the
extended robustness protocol additionally permits shot 1 and shot 3, always as
independent caches. A no-network mock smoke test is safe to run first:

    python scripts/stlp_generate_candidates.py \
      --data-dir data/ICEWS14 \
      --split test --shot 5 --provider mock --limit 4 \
      --output runs/cache_smoke/test_s5.jsonl

After reviewing the prompt contract and enabling credentials, generate the
optional DeepSeek validation/test caches with:

    ./scripts/generate_deepseek_caches.sh

The default creates the task-required test_s5.jsonl and test_s10.jsonl. ICEWS14
contains 13,179 unique oriented public test queries, so this is 26,358 API
requests with the one-query-per-request implementation; review expected cost and
runtime first. Set INCLUDE_VALID=YES only for the from-scratch matrix. That adds
valid_s5.jsonl and valid_s10.jsonl for 14,771 unique validation queries per shot.

Validation caches are mandatory whenever LLM-aware validation selects an epoch.
They are not required by the frozen, zero-epoch v5 checkpoint matrix because
those checkpoints were already selected without looking at LLM test outputs.

Prompt ablations use --omit-support or --omit-history and must write to distinct
cache files. A cache built under one history protocol is rejected under another.

## Train or evaluate the integrated model

Example score-mode run:

    python train.py \
      --data-dir data/ICEWS14 \
      --output-dir runs/s5_seed42_score \
      --device cuda:0 --shot 5 --seed 42 \
      --llm-mode score \
      --llm-valid-cache cache/standard_rolling_history/valid_s5.jsonl \
      --llm-test-cache cache/standard_rolling_history/test_s5.jsonl

The task-book `w/o LLM confidence` ablation uses the same cache and candidates:

    --llm-mode rationale --llm-disable-confidence

This zeros only the LLM confidence tensor; mapping, template, temporal, rank,
and candidate admission remain active. Score mode is the `w/o temporal
rationale` comparison, while off mode removes LLM candidate admission.

If an exact matching v5 checkpoint is available, add --init-from-v5 PATH and use
the checkpoint's dim/history/channels configuration. The loader permits only
llm_sidecar parameters to be missing and rejects every other architecture
mismatch.

The prepared from-scratch formal matrix matches the existing v5 configuration
(40 epochs, dim 256, history 10) and covers shots 5/10, seeds 42/43/44, and modes
off, candidate, score, and rationale:

    ./scripts/run_llm_main_matrix.sh

The script is prepared but is not launched automatically.

To isolate the additive effect without changing any learned v5 parameter, use
the six existing validation-selected v5 checkpoints:

    ./scripts/evaluate_v5_llm_checkpoint_matrix.sh

This second matrix runs with zero training epochs. Set `CHECKPOINT_ROOT` to the
directory containing `main_s{5,10}_seed{42,43,44}/best.pt`; the script fails
instead of silently falling back if any checkpoint is absent.

## LLM-only diagnostic baseline

    python scripts/stlp_evaluate_llm_only.py \
      --data-dir data/ICEWS14 \
      --cache cache/standard_rolling_history/test_s5.jsonl \
      --split test --shot 5 --ranking-mode confidence

This reports mapped candidate Recall@10, mapping/hallucination rates, latency,
tokens, and average-tie MRR for unlisted entities. It is a diagnostic baseline,
not a replacement for the graph model.

## Acceptance rule from the task specification

Performance improvement may be claimed only if the three-seed mean MRR improves
at shot 5 or shot 10 and the corresponding Hits@10 decrease is no greater than
0.003. Report all seeds, both directions, cache coverage, mapping rate,
hallucination rate, candidate Recall@10, latency, and token usage. A missing
cache entry must fall back to v5 rather than be silently fabricated.

The directly comparable completed v5 means are MRR 0.4074618 / Hits@10 0.5863067
for shot 5 and MRR 0.4086973 / Hits@10 0.5865554 for shot 10. Therefore the
minimum Hits@10 guardrails are 0.5833067 and 0.5835554 respectively. These
numbers are comparison references, not LLM-version results.

See DESIGN_v1.7.0alterego_v5_llm.md for the complete integration and ablation
matrix. Existing smoke outputs are explicitly non-formal and must not be cited
as experimental results.
