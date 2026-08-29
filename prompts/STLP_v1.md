# STLP v2 local compact-wire prompt contract

The executable prompt is constructed by src/stlp.py. This document fixes the
scientific contract independently of any API provider.

Inputs allowed in the prompt:

- known entity name;
- oriented relation name and prediction direction;
- query timestamp;
- support facts strictly earlier than the query timestamp;
- recent history facts strictly earlier than the query timestamp.

Inputs forbidden in the prompt or cache key:

- hidden target id or name;
- facts from the current query snapshot;
- validation/test facts later than the query timestamp;
- filtered-answer labels, target coverage labels, or model rank labels.

The response is a JSON object containing at most ten candidates. The local-Qwen
wire response uses compact aliases e, c, r, and t to avoid repeatedly decoding
long JSON keys; r is limited to a factual phrase under eight words. The client
expands those aliases to entity_name, confidence, temporal_rationale, and
temporal_consistency before the generator sees the response. Thus the persisted
cache contract is unchanged. The postprocessor maps names to ICEWS ids
conservatively and records the mapping method and confidence. The training and
evaluation processes never call an API; they consume the resulting immutable
JSONL cache.
