# Design: v1.7.0alterego_v5_llm / LLM-FSTKGC

## Scope and protected parent

The direct parent is NineFuseTKG v1.7.0alterego_v5. LLM-FSTKGC preserves the
parent's four experts, losses, full-vocabulary generation, candidate reranking
and antisymmetric tournament. The language-model channel is removable
candidate-side evidence.

The public release contains one provider route: Alibaba Cloud Model Studio
realtime `qwen-flash`. Candidate generation is separate from graph training and
evaluation. This design document describes the included implementation, not a
manuscript result matrix.

## STLP data flow

1. For each oriented public query, find strictly earlier same-relation facts,
   retain the last `max(8*K, 32)` rows, then select at most `K` after ordering
   that bounded pool by decreasing timestamp and subject match. Recent subject
   history is also restricted to timestamps strictly earlier than the query.
2. Render the public query, support and history into the target-blind STLP
   prompt.
3. Call the official Alibaba Cloud OpenAI-compatible chat-completions endpoint
   with `temperature=0`, `enable_thinking=false` and JSON-object output.
4. Validate the response and map names to the closed entity vocabulary using
   exact, normalized-exact and conservative fuzzy matching.
5. Store target-blind query metadata, prompt hash, mapped candidates,
   diagnostics, latency, token usage, response identity and provenance in a
   frozen JSONL cache plus schema-v2 sidecar.
6. During evaluation, validate the dataset fingerprint, split, shot, history
   protocol and query hash before constructing sparse evidence tensors.
7. Fuse mapped evidence after the original four-expert probability fusion and
   before the existing v5 candidate tournament.

Provider calls never occur in model forward, loss computation, training,
validation or testing.

## Integration modes

- `off`: the exact parent path; all LLM tensors are ignored.
- `candidate`: mapped IDs are admitted to the tournament candidate bank when
  space permits; no semantic score residual is added.
- `score`: add a bounded residual from provider confidence, mapping quality,
  template agreement and source-rank prior.
- `rationale`: score mode plus the returned temporal-consistency field.

The expert log-probability tensor remains batch-by-4-by-entity. STLP evidence is
never normalized as a fifth expert and never performs direct dense prediction.
The residual is non-negative, sparse and hard-capped; invalid or unmapped IDs
have zero effect.

## Target-blindness and chronological history

The runtime locator is
`(known_entity_id, oriented_relation_id, timestamp)`. The immutable query key
also hashes the dataset fingerprint, split, shot, seed, history protocol,
support digest, history digest and prompt version. It accepts no target field;
target/gold/hidden-object fields are rejected.

Rolling history releases a completed timestamp only after every query at that
timestamp has been processed. Head prediction uses the inverse relation and the
known object; it does not expose the hidden subject.

The hidden target is never serialized as an answer field or used to select or
filter the public context. Its name may occur naturally in an earlier support or
history fact; the implementation does not consult the label to remove such
occurrences.

## Alibaba Cloud Qwen boundary

The only released LLM provider identity is:

- provider: `aliyun_qwen_realtime`;
- requested model alias: `qwen-flash`;
- region: `cn-beijing`;
- endpoint: `https://dashscope.aliyuncs.com/compatible-mode/v1`;
- structured output: `json_object`;
- thinking: disabled;
- temperature: 0.

The endpoint is pinned to the official HTTPS host. Network execution requires a
non-empty local credential, `--execute-api`, and separate confirmations for
public-context upload and paid realtime inference. Credentials are never
serialized to request plans, raw response artifacts, caches or run metadata.

The scheduler uses a bounded in-flight window, sliding-window RPM/TPM limits,
shared cooldown after rate limits and audited bounded retries. Responses and
provider abstentions are appended and fsynced individually. Resume schedules
only custom IDs absent from validated response or abstention artifacts.

A `data_inspection_failed` provider response receives at most three attempts
and is then represented as an explicit abstention. Collection gives that query
an empty candidate set and never synthesizes a provider answer. Other invalid
request errors fail closed.

## Included runnable recipe and release boundary

The repository includes one optional frozen-parent evaluation utility with the
following hard-coded/default scope:

- ICEWS14 support budgets `K in {1,3,5,10}`;
- seeds 42, 43 and 44;
- active modes candidate, score and rationale;
- validation-selected v5 parent checkpoints;
- zero training epochs in the LLM-aware evaluation;
- standard rolling history;
- off-mode metrics reused from the matching parent runs.

The public source also contains an LLM-only sparse-candidate diagnostic and
target-blind prompt controls: support-order permutation, recent-history removal
and deterministic opaque entity names. Each perturbation uses a separate plan,
response set and cache.

Data, caches, checkpoints and result artifacts are not included. This utility
therefore documents an executable code path only; it is not represented as the
experiment matrix or result source of a manuscript. Any scientific claim must
be linked separately to exact source, data, cache, checkpoint, command and
completed-run hashes.

## Provenance and reproducibility

Every formal cache records request/index hashes, dataset fingerprint, provider
and returned model strings, response IDs, decoding settings, usage, latency,
attempt/retry counts, abstention policy, generation timestamps and cache hash.
The provider declaration is `ALIYUN_QWEN_REALTIME_PROVENANCE.json`, and the
public-code boundary is summarized in `RELEASE_SCOPE.md`.

The `qwen-flash` alias is provider managed. Exact weights and an immutable
provider revision are unavailable. Frozen caches therefore reproduce the graph
ranking path, while byte-identical semantic regeneration is not guaranteed.
This limitation must remain visible in any result or release derived from the
repository.
