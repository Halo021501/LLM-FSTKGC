# STLP target-blind prompt contract

The executable prompt is constructed by `src/stlp.py` and serialized by the
offline Alibaba Cloud Qwen request planner.

Inputs allowed in the prompt:

- known entity name;
- oriented relation name and prediction direction;
- query timestamp;
- at most `K` same-relation support facts strictly earlier than the query;
- recent history facts strictly earlier than the query.

Inputs forbidden as explicit prompt fields, cache-key fields or context
selection signals:

- hidden target/answer ID, label or separately supplied answer name;
- facts from the current query snapshot;
- validation/test facts later than the query timestamp;
- filtered-answer labels, target-coverage labels or graph-model rank labels.

The implementation does not use the answer to remove naturally occurring names
from public history. Therefore, an entity equal to the hidden target may appear
inside a strictly earlier support/history fact without being supplied as an
answer label. Cache construction and lookup still have no target argument.

The system message requires valid JSON using only the supplied target-blind
causal context. The user message requests one object with a `candidates` list
of at most ten items. Every item contains `entity_name`, `confidence`,
`temporal_rationale` and `temporal_consistency`; numeric fields must be in
`[0,1]`.

The released provider route is Alibaba Cloud Model Studio realtime
`qwen-flash`, with thinking disabled and temperature zero. The collector
validates the JSON contract before conservatively mapping names to ICEWS entity
IDs. Unresolved or ambiguous names remain unmapped. Training and evaluation
never call the provider; they consume immutable cache records and schema-v2
metadata sidecars.

The executable prompt/key version in this source snapshot is
`stlp-aliyun-qwen-realtime-v1`. Caches built with a different version have
different query keys and are not claimed to be byte-compatible.
