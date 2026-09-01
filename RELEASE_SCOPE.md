# Public release scope

This repository is a source-code release, not a result-artifact archive.

## Included

- the four-expert temporal graph ranker and AlterEgo tournament;
- the removable STLP candidate sidecar;
- the Alibaba Cloud Model Studio realtime `qwen-flash` request, validation,
  collection and cache-loading path;
- target-blind cache-key enforcement and strictly earlier rolling history;
- toy data and offline unit tests;
- optional training and frozen-parent evaluation utilities.

## Not included

- ICEWS14 or ICEWS18 data and entity/relation name maps;
- Alibaba Cloud credentials, provider responses or generated STLP caches;
- trained checkpoints, experiment logs or manuscript result tables;
- alternative LLM providers or unfinished local-model experiments;
- an immutable Qwen model-weight revision, because `qwen-flash` is a
  provider-managed alias.

## Interpretation boundary

The implementation uses a bounded recent support pool. For a query and support
budget `K`, `src/train.py::choose_causal_support` considers only the last
`max(8*K, 32)` strictly earlier same-relation rows and then sorts that pool by
decreasing timestamp and subject match. This is the executable selection rule;
the release does not claim a global subject-prioritized sort over every earlier
same-relation row.

The hidden answer is never serialized as a label or cache-key field and is not
used to select or filter prompt context. An entity equal to the hidden answer
may nevertheless occur naturally in a strictly earlier public fact. Such a
natural occurrence is not removed, because doing so would require consulting
the answer label.

The current prompt/key version is `stlp-aliyun-qwen-realtime-v1`. Caches made
with another prompt version have different query keys and are not represented
as byte-compatible with this generator.

The optional frozen-parent matrix scripts implement one user-runnable recipe.
They do not establish the experiment scope of a manuscript. Scientific claims
require a separate record linking the exact source manifest, dataset hashes,
cache hashes, checkpoints, commands and completed results.
