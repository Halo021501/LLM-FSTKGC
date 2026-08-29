# Design: v1.7.0alterego_v5_llm / LLM-FSTKGC

## Scope and protected parent

The direct parent is NineFuseTKG v1.7.0alterego_v5. Its source manifest passed
before the independent directory was created. The new implementation preserves
all parent experts, losses, full-entity generation, candidate reranking, and the
antisymmetric tournament. The LLM integration is removable side evidence.

The task specification used for this design has SHA-256
8db4a1ef31529335a1dd2f221064b56f92e7e500e27fece113a5eda32fb6477a.

## STLP data flow

1. For each public oriented query, select relation support strictly at t less
   than query time and recent subject history strictly at t less than query time.
2. Render entity/relation names and request at most ten semantic-temporal
   candidates with confidence and a short temporal rationale.
3. Map names back to dataset ids using exact, normalized-exact, then conservative
   fuzzy matching with a threshold and ambiguity margin.
4. Save only target-blind query metadata, prompt hash, candidates, diagnostics,
   latency, and token usage to JSONL. Do not save the prompt or hidden target.
5. At train/evaluation time, validate schema, dataset fingerprint, split, shot,
   history protocol, and query hash before creating sparse tensors.
6. Fuse mapped candidates after the original four-expert probability fusion and
   before the existing v5 candidate tournament.

There is no local-LLM or external API call in model forward, loss computation,
validation, or test. The formal provider is a loopback vLLM service loading the
official Qwen2.5-7B-Instruct-AWQ checkpoint on one isolated GPU.

## Four modes

- off: exact parent path. LLM tensors are ignored.
- candidate: mapped ids are guaranteed admission to the tournament candidate
  bank when space permits. No LLM score residual is added.
- score: bounded residual from LLM confidence, mapping quality, naming-template
  agreement, and source rank prior.
- rationale: score mode plus temporal consistency from the short rationale.

The feature weights and global scale are trainable when a target-blind training
cache is explicitly supplied; otherwise they remain a fixed deterministic
calibration. The mapping score is a safety gate and the dense residual is
hard-capped. Duplicate mapped ids use the maximum candidate residual.
Invalid/unmapped ids have zero effect.

The expert log-probability tensor remains B by 4 by E. LLM candidate evidence is
never normalized as a fifth expert and never performs direct E-way prediction.

## Target-blindness and causal history

The runtime locator is the tuple known_entity_id, oriented_relation_id,
timestamp. The cache query key additionally hashes split, shot, seed, protocol,
dataset fingerprint, support digest, and prompt version. No key accepts a target
argument. Explicit target/gold/hidden-object fields are rejected.

Rolling history adds a whole completed timestamp only after every query at that
timestamp has been generated or evaluated. Strict-static history never adds
validation/test facts. Head prediction uses the inverse relation and the known
object; it does not expose the hidden subject.

## Required experiment matrix

Main comparison for each shot in 5 and 10 and seed in 42, 43, 44:

1. v5 / llm-mode off;
2. LLM-only mapped candidate diagnostic;
3. candidate;
4. score;
5. rationale.

Run both a frozen-checkpoint matrix, which changes no learned v5 parameter, and
a matched from-scratch matrix using the parent's 40-epoch configuration. The
former isolates inference-time LLM evidence; the latter permits validation to
select the best epoch under the integrated prediction rule.

Shot 1 and shot 3 are a user-requested low-resource robustness extension rather
than a silent change to the original task-book matrix. They use the identical
target-blind query, causal-history, provider, mapping, and filtered-ranking
contracts and are stored as separate `test_s1`/`test_s3` conditions. Their
results must be labelled as an additional analysis when reported.

The frozen matrix needs only the task-required test_s5/test_s10 caches. The
from-scratch matrix also requires matching validation caches; it is forbidden to
select an epoch or calibration using test-cache metrics.

Ablations:

- omit support from the prompt;
- omit recent history from the prompt;
- `--llm-disable-confidence`: retain candidates and every other feature while
  zeroing only LLM confidence;
- score mode versus rationale mode removes only the temporal-rationale feature;
- off mode removes LLM candidate admission;
- primary rolling versus separately generated strict-static cache.

Report filtered MRR, Hits@1/3/10, subject/object metrics, raw metrics, all
four-expert diagnostics, cache hit rate, mapped candidates/query, LLM candidate
Recall@10, mapping rate, hallucination rate, latency, and prompt/completion
tokens. Cache artifacts and run_meta.json provide provenance.

## Acceptance and non-claims

Do not claim an improvement from a single seed or a smoke run. A positive result
requires better three-seed mean MRR in at least one of shot 5 or shot 10, with
Hits@10 degradation no greater than 0.003. If this gate fails, the valid outcome
is a diagnostic/negative result. No fabricated API output or metric is allowed.

The completed direct-parent means are 0.4074618 MRR / 0.5863067 Hits@10 at shot
5 and 0.4086973 MRR / 0.5865554 Hits@10 at shot 10 under the same rolling,
tie-aware filtered protocol.

## Provider boundary

The formal task-book provider is Qwen2.5-7B-Instruct-AWQ served locally through
vLLM's OpenAI-compatible interface. The client rejects every non-loopback URL,
the server binds to 127.0.0.1, and Hugging Face network access is disabled while
serving. `qwen_local` owns only the inference runtime; `regcn` continues to own
all candidate construction, mapping, v5 training, and evaluation.

For pinned vLLM 0.6.3, a project-owned entrypoint avoids the upstream wildcard
pre-bind and verifies the effective loopback listener. Generation uses an
explicit candidate JSON Schema with LM Format Enforcer, followed by independent
client-side field/type/range validation before any cache append. The default GPU
kernel is AWQ Marlin; ordinary AWQ remains a compatibility fallback.
For its single-GPU executor, the same entrypoint substitutes a
permission-restricted FileStore for vLLM's default wildcard-bound PyTorch
TCPStore rendezvous. This removes an unauthenticated auxiliary network listener
without changing model weights, decoding, or the loopback API; remaining
one-rank Gloo/NCCL communication is restricted to `lo`.
Cache sidecars use metadata schema v2. Local caches are tied to the exact model
revision, model-manifest hash, runtime-lock hash, and serving kernel, and record
the initial command, UTC time, host, and physical GPU. Resume validates every
scientific metadata field before skipping any existing query locator.

DeepSeek remains an optional provider through its external OpenAI-compatible
chat-completion endpoint. Credentials exist only in a gitignored local file,
and external generation retains its explicit confirmation gate. Switching
providers requires a separately named cache and new cache SHA-256; it never
changes the protected parent checkpoint.

## Optional Alibaba Cloud Qwen Batch branch

Alibaba Cloud Qwen Batch is an independent asynchronous cache-generation
branch, not a replacement for the task-book's local Qwen provider. Its default
paper model is the dated `qwen3.7-flash-2026-07-15` alias. The moving
`qwen-flash` alias is permitted only as a separately named rolling-price
condition; no cache, metric row, or claim may conflate the two identities.
Both are provider-managed models whose exact weight revision is unavailable.

The Batch lifecycle is deliberately separated into offline and external
phases:

1. `prepare` deterministically builds target-blind request JSONL, a public
   custom-id index, hashes, counts, and a cost-estimation plan without API
   access or a credential in the child process.
2. `estimate` reviews the prepared bytes and request/token estimates. No
   submission is implicit.
3. `submit`, `status`, and `download` each require both
   `CONFIRM_ALIYUN_QWEN_DATA_UPLOAD=YES` and
   `CONFIRM_ALIYUN_QWEN_PAID_BATCH=YES`, a non-empty credential, and the
   explicit API-execution flag.
4. `collect` works offline, associates unordered output by target-blind
   `custom_id`, validates every response, maps names, and atomically writes the
   normal record-schema-v1 cache plus metadata-schema-v2 sidecar.
5. `prepare-retry` creates a separate request set containing only failed,
   missing, or invalid ids. It must not resubmit successful ids.
6. Remote deletion is a separate destructive operation requiring the two
   network confirmations plus `CONFIRM_ALIYUN_QWEN_REMOTE_DELETE=YES`.

Each request line uses `/v1/chat/completions`, one fixed model and thinking mode
per input file, `enable_thinking=false`, and JSON-object structured output. The
prompt explicitly requests JSON. A fixed `max_tokens` is not sent because a
truncated object cannot enter the scientific cache. Candidate count, required
fields, types, ranges, JSON validity, HTTP status, provider error, and finish
state are nevertheless checked after download. Duplicate, unknown, or missing
custom ids make collection fail without replacing any prior cache.

`custom_id` is derived from the existing target-blind query key, which includes
dataset fingerprint, split, shot, seed, history protocol, support/history
digests, prompt version, known entity, oriented relation, direction, and time,
but no answer. Output order has no semantic meaning. Batch preparation keeps the
same whole-timestamp rolling-history barrier as local generation, including the
inverse-relation head query, so parallel provider execution cannot introduce a
future or current-snapshot fact.

The provider requires a temporary serialization of each prompt for upload.
Those request and downloaded-result artifacts live under the gitignored
permission-restricted `runs/aliyun_qwen_batch/` staging tree. The canonical
cache still stores no raw prompt or hidden answer, only its SHA-256. The uploaded
content is limited to public ICEWS names, query direction/time, and strictly
earlier facts. Remote files are retained until the explicitly confirmed cleanup
step, so local output and hashes must be verified before deletion.

Aliyun caches use a provider/model/prompt-variant namespace, for example:

    cache/standard_rolling_history/aliyun_qwen_batch/
      qwen3.7-flash-2026-07-15/standard/test_s5.jsonl

Metadata records the provider, requested and returned model identities,
provider-managed status, region/official host, job and remote file ids,
request/result/index hashes, request counts, UTC lifecycle timestamps, token
usage, retry/error counts, batch wall time, and throughput. It never records the
credential. The immutable extension boundary and fixed-versus-rolling model
policy are declared in `ALIYUN_QWEN_BATCH_PROVENANCE.json`; the existing local
model provenance file remains unchanged.

### Batch experiment gate

The external branch advances in three stages:

1. A 50-query smoke run verifies upload, structured output, unordered result
   matching, download, offline collection, and remote cleanup.
2. A 200--500-query pilot compares the exact same public query locators against
   local Qwen using JSON success, mapped candidates/query, mapping and
   hallucination rates, candidate Recall@10, token usage, wall-time throughput,
   retry rate, and actual billed cost. No metric from this partial cache is a
   paper result.
3. Only a passing pilot authorizes the complete test_s5/test_s10 cache and the
   frozen-v5 matrix. Validation caches and the matched from-scratch matrix are
   generated only after the full test comparison is scientifically promising.

Local Qwen and Aliyun results are separate provider conditions. They require
separate LLM-only diagnostics, frozen-v5 candidate/score/rationale results, and,
if pursued, matching validation/from-scratch results. The same three seeds and
the existing v5 acceptance thresholds apply. Batch wall time and throughput are
reported separately from local per-request latency because an asynchronous job
does not expose a comparable query latency.

## Optional Alibaba Cloud Qwen realtime branch

The realtime branch is a time-critical transport alternative, not a new model
fusion method. It consumes the already reviewed target-blind Batch request and
index hashes, calls the Beijing OpenAI-compatible chat-completions endpoint, and
stores raw responses and formal caches under an independent
`aliyun_qwen_realtime/qwen-flash` namespace. It never changes local-Qwen files,
the v5 parent, training, or evaluation code.

The wrapper runs any explicitly selected ordered subset of shots 1, 3, 5, and
10. The default remains the task-book shot 5/10 pair; the added shot 1/3 pair
uses an independent controller and log. Aggregate scope profiles use
8/16/32 workers for smoke/pilot/full, with corresponding 240/480/960 RPM and
300,000/600,000/1,200,000 TPM ceilings. Each profile uses a conservative
1,200-token reservation per attempt and is split evenly across the two
selected processes. These project ceilings are
deliberately below the provider's documented floating-`qwen-flash` Beijing
limits. A thread-safe sliding-window gate controls both requests and reserved
tokens. Rate limits honor `Retry-After` and impose a shared cooldown; transient
transport, 408, 429, and 5xx errors use bounded exponential backoff. Invalid
credentials or request contracts fail closed. Responses are schema-validated
before an fsynced append, and resume schedules only custom IDs absent from the
validated response or abstention files. A `data_inspection_failed` response is
query-local rather than a global request-contract error: it receives at most
three bounded attempts and then becomes an append-only, sanitized provider
abstention. Other HTTP 400 responses still fail closed.

The realtime scheduler maintains a bounded in-flight window equal to the worker
count; it never queues the complete split in an executor. The main thread alone
persists outcomes and a periodic heartbeat. Bare response-read timeouts and
HTTP transport exceptions are normalized into the reviewed retry path, while
an unexpected worker exception stops new submission, records a sanitized fatal
audit, and drains only the bounded live window. The socket timeout is an I/O
inactivity bound, not a strict whole-request wall-clock deadline; a true hard
deadline would require isolating blocking requests in replaceable processes.

Collection is a separate offline phase. It rejects duplicate, unknown, missing,
non-200, truncated, or invalid candidate responses before writing any cache.
For a validated provider abstention only, it records an empty candidate set and
numeric abstention diagnostics; it never synthesizes a successful provider
response or candidate. Consequently the unchanged v5 model applies zero
LLM-side bonus to that query. The metadata preserves abstention count, reason,
policy, and artifact hash so this fallback can be disclosed and audited.
Unlike Batch, realtime metadata records measured per-request latency. It also
records token usage, attempts/retries, concurrency and limiter settings, exact
request/result hashes, returned model identities, and an estimated list-price
cost. Because `qwen-flash` is a moving provider alias, exact weights are not
available and realtime results remain a separately disclosed provider
condition. The immutable project declaration is
`ALIYUN_QWEN_REALTIME_PROVENANCE.json`.
