# Interview contracts and scoped lexical memory

This is a **pure Python foundation**, not an interview application or a durable
interview workflow. It validates plans, source-bound evidence, assessed coverage,
memory records and scoped lexical retrieval. It does not ask questions, call a
model, persist interviews, authenticate participants or resume interrupted work.
Import these APIs from `promptwitness.interview_models` and
`promptwitness.interview_memory`. The separate
[durable interview workflow](interviews.md) now supplies scheduling, persistence
and provider orchestration; those responsibilities are not hidden in these pure classes.

## Source-bound evidence

`ParticipantAnswer` contains the interview, participant, answer and question IDs,
exact text and positive revision. Only the `participant_answer` source kind is
accepted. An empty answer is valid, but cannot supply a nonempty evidence span.

`EvidenceRef` uses half-open **Python Unicode codepoint** offsets `[start, end)`,
not UTF-8 bytes, UTF-16 units, graphemes or model tokens. No whitespace, newline or
Unicode normalization is performed. For example, the emoji in `"A😀e\u0301"`
occupies one codepoint but four UTF-8 bytes; `e` and its combining accent occupy
two codepoints. The quote must equal the exact source slice. References bind the
full answer digest and revision, not merely a text digest or answer ID.

`BoundEvidence(reference, answer)` validates that binding on construction.
`MemoryRecord` and positive `CriterionAssessment` constructors require bound
evidence, not unvalidated reference dictionaries. Deserializing either requires
an explicit `answers=` catalog. Its identity is `(interview_id, answer_id)`:
identical answer IDs in different interviews are valid, duplicate catalog keys
are rejected, and conflicting source versions under one key cannot be merged.
Distinct slices of the same consistent answer remain valid.

Exact citation checks prove source linkage, **not** that the participant was
truthful, the source was authenticated, the model summary is correct, or the
evidence semantically entails an assessment. The integrating journal/runner must
establish trusted source ownership and revisions before constructing answers.

## Plans and assessed coverage

`InterviewPlan` preserves ordered original topics and criteria. Topic IDs are
unique; criterion IDs are globally unique. It also records budgets, retrieval
policy, prompt versions and bounded nonsecret generation settings. These are
contracts, not an implemented budget-enforcing scheduler.

`CriterionAssessment` statuses are `covered`, `partial` and `unanswered`.
Covered/partial decisions require at least one exact source-bound citation from
the assessment's participant and interview. `assessed_coverage()` rejects
duplicate decisions, unknown criteria, cross-interview/participant decisions and
conflicting answers across different decisions.

The original required denominator is always `len(plan.required_criteria)`.
Accepted emergent topics must be passed separately using `accepted_emergent=`;
they cannot overwrite existing IDs or change that denominator. Their assessed
coverage is reported separately (`None` when there are no accepted emergent
criteria). Missing decisions count as unanswered, and partial is not counted as
covered. A memory citation does not implicitly cover anything. These are
**assessed coverage counts**, not factual accuracy, novelty, interview quality or
a claim that the interview is complete. Proposal acceptance and follow-up
scheduling are deliberately not implemented here.

## Memory identity, corrections and snapshots

`MemoryRecord` contains participant-bound evidence, unique topic/criterion links,
an explicitly labelled `model_proposed` summary and an optional `supersedes` ID.
Its ID is `mem-` followed by SHA-256 of the canonical content body (excluding the
ID itself). Changing the evidence, links, summary or correction link changes its
identity. Reordered evidence and links are canonicalized; duplicate references or
links are rejected. Source answers remain separate from summaries.

`supersedes` must be a memory ID and cannot reference the record itself. This
foundation does **not** verify target existence, same-participant ownership or
correction ancestry/cycles; a future durable journal must do so. Original and
corrected records remain separately visible and both can rank in retrieval. No
truth arbitration, latest-record resolution or semantic deduplication occurs.

`MemoryEntry` records the owning interview and revision separately from memory
content, avoiding a self-referential event hash. `SourceHead` declares an interview
revision and digest. `MemorySnapshot.build(participant_id, heads, entries)` filters
the catalog before computing retrieval statistics:

- Other participants, unselected owning interviews and entries newer than a
  selected head are excluded.
- A memory referencing any unselected source interview is excluded as a whole;
  evidence is never silently stripped from its content identity.
- Selected entries with source evidence newer than its selected head are invalid.
- Duplicate heads, memory IDs and conflicting composite answer identities are
  rejected rather than silently shadowed.

Snapshots retain a minimal self-contained answer catalog, canonical entries,
selected heads and a content digest. Reloading checks fields, versions, source
bindings, scope, unused answers and hashes. Head digests are **declared metadata**:
these pure classes cannot verify journal ancestry, membership or participant
authorization. A valid hash is an integrity checksum, not a signature. Callers
must authenticate heads and obtain entries from those exact revisions before
using a snapshot for real interviews.

## Executable pure example

The head below is explicitly a synthetic fixture, not evidence of a durable
journal. This example performs no I/O and no provider request.

```python
from promptwitness.interview_memory import (
    BM25MemoryIndex,
    MemoryEntry,
    MemorySnapshot,
    SourceHead,
)
from promptwitness.interview_models import (
    BoundEvidence,
    CriterionAssessment,
    CriterionLink,
    InterviewCriterion,
    InterviewPlan,
    InterviewTopic,
    MemoryRecord,
    ParticipantAnswer,
    RetrievalPolicy,
    assessed_coverage,
    contract_json,
)

plan = InterviewPlan(
    "commute-study",
    "1",
    "Understand commuting preferences.",
    (
        InterviewTopic(
            "travel",
            "Travel choices",
            (InterviewCriterion("mode", "Describe the usual travel mode."),),
        ),
    ),
)
answer = ParticipantAnswer("visit-1", "p-1", "a-1", "q-1", "I cycle to work.", 1)
evidence = BoundEvidence.from_answer(answer, 2, 7)
link = CriterionLink("travel", "mode")
memory = MemoryRecord("p-1", (evidence,), (link,), "Participant described cycling.")
decision = CriterionAssessment("p-1", "visit-1", link, "covered", (evidence,))
report = assessed_coverage(plan, (decision,)).to_dict()
assert report["required"]["assessed_coverage"] == 1.0

snapshot = MemorySnapshot.build(
    "p-1",
    (SourceHead("visit-1", 2, "0" * 64),),  # Declared fixture head only.
    (MemoryEntry(memory, "visit-1", 2),),
)
restored = MemorySnapshot.from_json(contract_json(snapshot.to_dict()))
policy = RetrievalPolicy(top_k=2, context_bytes=4096, include_model_summaries=False)
result = BM25MemoryIndex(restored, policy).search(
    "cycle", context_prefix="Untrusted interview evidence follows:\n", context_suffix="\n"
)
assert result.memory_ids == (memory.memory_id,)
assert result.context_bytes == len(result.context.encode("utf-8")) <= 4096
assert result.to_dict()["snapshot_digest"] == snapshot.digest
```

## BM25 semantics and deterministic context packing

Each scoped memory is one retrieval document. Its text is all exact quotes joined
with newlines, plus its model summary when `include_model_summaries=True` (the
default). Overlapping quotes therefore contribute repeated terms: this is a
declared representation, not unique source-token frequency. Setting that option
to `False` excludes summaries **only from ranking**. Selected context still
includes the clearly labelled summary so the emitted record retains its original
identity; the option is not a redaction or provider-data exclusion control.

The versioned tokenizer uses Unicode alphanumeric runs after `str.casefold()`.
Underscores, punctuation and combining marks are not word characters in this
tokenizer; no NFC, stemming, stopword filtering or language model is used. Empty
and punctuation-only documents remain in the corpus size and average length.
Empty/OOV queries return no matches, without inventing fallback memories. Repeated
query terms are deduplicated. Unicode categories follow the Python runtime's
Unicode database, so runtime changes can affect tokenization.

For each distinct query term present in a document:

```text
idf = log(1 + (N - df + 0.5) / (df + 0.5))
contribution = idf * tf * (k1 + 1) /
               (tf + k1 * (1 - b + b * document_length / average_length))
score = sum(contributions)
```

`N`, document frequency and average length come only from the selected immutable
snapshot, never another participant or future catalog entries. `0 < k1 <= 10`
and `0 <= b <= 1`; default values are `1.2` and `0.75`. Positive matches sort by
descending score, then ascending content-derived memory ID. Stable ordering and
canonical JSON make the same inputs reproducible on the same runtime; no
cross-platform bit-identical floating-point guarantee is made.

Packing scans that ranking. If a whole record cannot fit, it is omitted and a
smaller later match may fill the slot. `top_k` counts included records, not the
first candidates examined. There is no quote truncation. Audits distinguish
`context_bytes`, `top_k` and aggregate `omitted_no_overlap`; when top-k has already
been reached, that reason takes precedence over a candidate's byte size.

The byte limit includes **the entire supplied prefix, complete canonical JSON
envelope (IDs, citations, summaries, metadata and delimiters), and suffix** in
UTF-8. Even the empty envelope must fit or retrieval fails explicitly. Exactly
fitting results are accepted; one byte less can omit a record. This is not a
model token limit, chat-message serializer limit or HTTP wire-body limit. An
integrating runner must include its remaining prompt/request overhead and reserve
space for the answer before making calls. Independently, the entire retrieval
audit export must fit the generic 16 MiB contract cap; its extra hit metadata and
JSON escaping can cause rejection even if the context alone fits its budget.
Retrieved content is untrusted data:
this module neither executes instructions within it nor provides a model-level
prompt-injection guarantee.

## Bounds, complexity and evidence

Canonical interview JSON rejects unknown fields, unsupported versions, duplicate
keys, non-finite numbers, invalid UTF-8 text, nesting deeper than 32 and payloads
over 16 MiB. Integer bounds reject booleans. Answer text is at most 1 MiB, IDs are
at most 512 UTF-8 bytes, memory/assessment references are at most 128, snapshot
heads at most 128, and catalogs at most 10,000 entries/answers. Source references
are bounded and nonempty. Aggregate canonical-size checks include complete
export envelopes, including content IDs/digests, in addition to per-field limits.
These are payload/workload bounds, not a
maximum RSS guarantee: parsing, JSON construction, source objects, hashing and
Python container overhead allocate additional memory. Nested objects such as
topics, links and source heads inherit the enclosing contract's version.

Let `B` be serialized snapshot bytes, `T` the number of indexed token occurrences,
`U` the sum of distinct terms per document, `N` the number of scoped memories,
`Q` distinct query terms and `M` positive matches. Indexing uses `O(T + B)` work
and `O(U + N)` index storage in addition to the retained snapshot. Search scans
documents in `O(NQ)`, sorts matches in `O(M log M)`, then packs in `O(M)` using
cached per-entry byte sizes. Serialization and content-digest validation add
byte-linear work; exact citations can trigger repeated source hashing. This
implementation has no inverted postings optimization, disk-backed index, ANN
vector search or provider-based memory extraction.

The tests independently calculate BM25 weights, original/emergent denominators,
UTF-8 sizes and canonical hashes. They cover exact Unicode spans, source
conflicts, frozen inputs, source-head filtering, correction visibility,
deserialization tampering, malformed/huge numbers and oversized contracts.
These are deterministic synthetic engineering tests, not real interview data,
gold-label extraction accuracy, semantic relevance evaluation or proof of a
complete interview workflow. Snapshot exports contain full selected participant
answers; keep them private. No encryption, consent, access control, deletion
policy or durable history is implemented by these pure contracts.
