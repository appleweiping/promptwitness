# Pure interview stage requests and responses

`promptwitness.interview_stages` prepares analyst/questioner requests and validates
their responses without a database, model call, credential read or automatic
retry. It builds on [interview contracts](interview-contracts.md). This module is
not an interview runner: scheduling, participant authentication, persistence,
reservation/CAS, publication, reviewed corrections and actual provider invocation
belong to a separate coordinator.

## Public contracts

- `StageTarget(topic_id, criterion_id, mode, parent_question_id,
  source_revision, source_digest, policy_version="scheduler/v1")` is a host
  decision. Modes are `initial`, `follow_up`, `transition` and `emergent`.
  Follow-up requires a parent. Recall is grounding context, not an alternate
  scheduling mode.
- `InterviewStageContext` binds interview/participant, source head, request and
  attempt IDs, a complete plan, target, accepted emergent topics, explicit
  `source_answers`, published `question_context`, optional `analysis_answer_id`
  and `allowed_criteria`. Empty allowed criteria defaults to the target alone.
  Otherwise the target must be included in the unique, known allowlist.
- `build_analysis_request(context, provider_identity)` and
  `build_question_request(context, provider_identity, snapshot=None, query="")`
  return an immutable `InterviewStageRequest` with `.messages`, `.generation`,
  `.wire_body`, `.wire_body_bytes`, `.audit_bytes`, `.digest`, `.binding_digest`,
  `.selected_memory_ids` and `.snapshot_digest`. Source/target/attempt properties
  forward to the typed context. `from_dict()`/`from_json()` reconstruct and compare
  all derived messages, bytes, digests and selection decisions; simply recomputing
  an altered artifact's outer checksum is insufficient.
- `parse_analysis_completion(request, response)` produces `AnalysisResult`:
  validated criterion assessments, newly content-addressed memories and at most
  one optional `EmergentProposal`. Its `.proposals` convenience property is a
  tuple of zero/one items. The result exposes the bound `.analysis_answer_id`.
- `parse_question_completion(request, response)` produces `QuestionResult` with
  text, its bound target and declared memory IDs. `to_question(question_id,
  revision)` creates an `InterviewQuestion` only when the host supplies the
  publication identity and a revision after the target's source head. It does
  not increment counters or publish anything. The class also exposes
  `InterviewQuestion.from_result(result, question_id, revision)`.
- `AnalysisResult.from_dict(data, request=request)` and
  `QuestionResult.from_dict(data, request=request)` revalidate serialized results
  against that exact request without network access.
- `CompletionRecord(request, response)` retains an immutable copy of the bounded
  original provider envelope and computes `.result` by parsing it. It exposes
  `.response_digest`, `.digest`, `.to_dict()`, and
  `from_dict(data, request=request)`/`from_json(data, request=request)`. Reloading
  re-runs the completion parser and compares the normalized result, not merely
  its checksum. It never manufactures a missing finish reason or status.

Every source answer must belong to this participant/interview, be at or before
the source head, fit the plan's answer-byte limit, and have its earlier published
question included in `question_context`. Context arrays are bounded to 128 each.
The current analysis answer must match the target of its answered question.
Positive evidence is limited to that **current answer**; historical answers can
be displayed as context but cannot supply new positive evidence in this version.
Recalled memory, summaries, question text and hypothetical answers are never
accepted as new source answers. Empty answers can support an unanswered decision
without evidence, but not a covered/partial decision requiring a nonempty quote.

The stage permits only supplied known criteria; the reducer still verifies that
the allowlist, accepted topics, transcript and target match actual committed
state. It also owns cumulative assessment policy. This module does not downgrade
or merge existing decisions. Automatic memory corrections (`supersedes`) are
rejected; a reviewed correction workflow is not implemented by these stages.

## Request identity and prompt reuse

The built-in templates are original, versioned `PromptDocument` JSON. Preparation
uses `parser.parse_prompt`, `validation.validate_prompt`, and
`matrix.render_matrix(..., strict=True)`. The trusted templates are validated
**before** substitution. Schema JSON and input JSON enter as two plain-string
variables, in one rendering pass. An answer containing `{{ name }}`, quotes,
braces, emoji or CRLF remains literal source data. It is not rendered again or
treated as another template. Exactly one system and one user message are emitted;
no tools, provider-native executable blocks or model-selected destinations are
allowed.

Public `RENDERER_VERSION`, `TEMPLATE_HASHES` and `SCHEMA_HASHES` expose the actual
implementation profile. Each role's template hash covers **both** the original
template document and the complete output schema. The plan's `analysis/v1` and
`question/v1` prompt versions must be supported, and an actual template/schema
change changes the profile hash. Output schema instructions do not rely on a
provider's JSON mode: structural and domain checks run locally after completion.

The provider identity has exactly these nonsecret fields, matching the existing
`OpenAISessionProvider.identity` shape: `provider`, `transport`, explicit nonempty
`model`, `endpoint_sha256`, `headers_sha256`, `api_key_env` (name or null), and
bounded `timeout`. Credentials, raw custom headers and raw endpoints are never
inserted into stage requests. Unlike the generic provider, this workflow requires
an explicit model rather than an endpoint-selected implicit default. Scripted
fixtures use the same closed identity shape; no model installation is required.

The model echoes `binding_digest`, a digest of pre-render context, plan, selected
memory IDs, retrieval snapshot/query, provider/generation and template/schema
identity. The final host-owned request digest additionally binds the complete
rendered messages, actual body digest/length and retrieval audit. It is not placed
inside its own hashed body: that would create a self-reference. A response echo
is a correlation check, not a signature or proof that the model reasoned from
those sources. A coordinator must reserve the full request before invocation,
pin its exact attempt/source head, compare the actual provider identity before
calling, and fence stale completions. A prompt-only `ReplayProvider` key is not
sufficient because it omits generation/model/attempt identity.

## Exact byte budgets and retrieval

The shared `providers.prepare_chat_body()` is pure and uses the **same** body
builder and JSON encoder as `OpenAICompatibleProvider.complete()`. Live JSON
serialization includes spaces and preserves the builder's insertion order;
compact canonical audit JSON has a different length. Stage generation settings
are normalized into canonical key order before that live encoder is called.

`wire_body_bytes` measures all UTF-8 **HTTP JSON body** bytes: messages, system
instructions, schema, plan/agenda, target, supplied transcript, selected complete
memory records, generation settings, model, JSON escaping and delimiters. It must
fit `plan.budgets.context_bytes`. It does not mean model tokens, HTTP headers,
TCP/TLS framing or response capacity. The existing transport separately bounds
headers and framing. `max_tokens` is a distinct configured output-token limit,
not a conversion from bytes.

The fixed request without recalled memories must fit, or preparation fails before
any provider effect; answers and required context are never silently truncated.
Question requests can use an explicit same-participant `MemorySnapshot`; a head
for the current interview must exactly match the request's source head. The
coordinator must verify that this snapshot equals committed imported/current
memory, rather than trusting a caller-supplied same-participant object.

Selection reuses BM25 scores over the scoped snapshot. It scans positive matches
in descending score/content-ID order. Each candidate must fit both the retrieval
sub-budget (complete canonical memory-entry array) and the actual rendered request
body budget. Oversized candidates are skipped and smaller later candidates may
fill the slot. Audit reasons are `top_k`, `retrieval_bytes`, and `request_bytes`,
with aggregate `omitted_no_overlap`. Thus memory can fit its standalone retrieval
allocation yet be omitted because its escaped request representation does not
fit. `selected_memory_ids` always reflects what was actually sent, not the
retriever's preliminary top-k. None of the omitted records are exposed as source
material to the model. Empty/OOV queries invent no fallback memories.

The entire request **audit export** separately has a 16 MiB canonical limit. It
contains the source snapshot and duplicated provenance needed for reconstruction;
it is not the HTTP body. `audit_bytes` measures that separate object, including
its digest. Neither limit is an RSS guarantee. Ranking uses the existing BM25
scan, and packing may render/encode a full candidate body once per positive match:
in the worst case this adds `O(M B)` byte work for `M` matches and candidate-body
size `B`, in addition to ranking and repeated content hashing. This is bounded
local packing, not an optimized large-memory service or an ANN index.

## Closed provider response shapes

The analyst's response content is one JSON object:

```json
{
  "format": "promptwitness.analysis-response/v1",
  "binding_digest": "<copy from request input>",
  "evidence": [
    {"id": "e1", "answer_id": "a1", "start": 0, "end": 5, "quote": "cycle"}
  ],
  "assessments": [
    {
      "topic_id": "travel",
      "criterion_id": "mode",
      "status": "partial",
      "evidence_ids": ["e1"],
      "rationale": "Model-proposed assessment."
    }
  ],
  "memories": [
    {
      "links": [{"topic_id": "travel", "criterion_id": "mode"}],
      "evidence_ids": ["e1"],
      "summary": {"kind": "model_proposed", "text": "Participant mentioned cycling."}
    }
  ],
  "proposal": null
}
```

The `a1` fixture answer must actually contain the exact codepoint slice `cycle`.
Evidence labels such as `e1` are local response labels. The host validates each
slice and computes `EvidenceRef` and `MemoryRecord` content IDs; the model does
not choose authoritative IDs. Duplicate labels, duplicate source spans under
different labels, unknown/unused references, duplicate decisions/memories,
unknown criteria, wrong summaries and out-of-scope evidence are rejected. A
decision for the exact requested target is required. Proposal, if present, has
exactly `topic` (an `InterviewTopic`), `parent_topic_id`, and `evidence_ids`; it is
bound to the current source head and final request digest. Its content-derived
proposal ID is host-computed. Proposal creation neither accepts it nor changes the
original required-coverage denominator; total accepted topics remain plan-bounded
and at most 32.

The questioner content has only:

```json
{
  "format": "promptwitness.question-response/v1",
  "binding_digest": "<copy from request input>",
  "target": {"topic_id": "travel", "criterion_id": "mode"},
  "text": "Could you describe your usual travel mode?",
  "memory_ids": []
}
```

Its target must match the host's exact decision, text must be nonempty and fit the
question UTF-8 byte limit, and unique memory IDs must be a subset of final sent
IDs. `memory_ids` is declared grounding, not proof of semantic use. A model cannot
provide a publication ID, revision, new mode, parent, tools or termination flag.

Both parsers first require a provider **envelope** with one unambiguously completed
text result and call `task_scores.response_text()`. Accepted examples are a chat
completion with one assistant choice and `finish_reason="stop"`, or an unambiguous
`{"status":"completed","output_text":"..."}` object. Bare strings, partial
responses, tools, refusals and conflicting output branches are rejected. Then the
content is limited to 1 MiB and passed through strict duplicate-key/finite/Unicode
JSON validation, the existing local schema-subset validator, and source/identity
domain validation. Markdown fences are not stripped, and malformed output is not
repaired or retried automatically.

## Verification and remaining boundaries

No paid provider or downloaded model is needed for these tests. They use explicit
completed fixtures and an intercepted transport boundary to compare independent
`json.dumps()` bytes with both preparation and actual `complete()` output. Tests
cover Unicode source slices, malicious literal template syntax, generation-order
roundtrips, rehashed artifact tampering, exact/one-byte-short wire budgets,
retrieval/request-budget differences, ambiguous completions and host-owned
publication.

Typed results alone establish structural/source/request validity; they do not
prove a provider produced them. A coordinator uses `CompletionRecord` and retains
the raw original completion envelope alongside the normalized result in a
committed event, re-running the stage parser and comparing that result during
replay. Frozen JSON containers are normalized back to equivalent plain JSON
before the existing completion checker; no fields or status values are added.
Even a retained envelope is not cryptographic proof of remote origin: the
transport/coordinator boundary must be trusted. Likewise, supplied
head digests and participant IDs are not authentication. No test here measures
human interview quality, factual entailment, semantic novelty, non-leading
questions or learned-memory relevance. Untrusted content remains private data,
and byte-limited prompts are not a prompt-injection security proof.
