# Provider traces and replay

`TraceRecorder` wraps any callable provider accepted by `ScenarioExecutor`. It
records request, response, and conventionally declared `tool_calls` events using
digests rather than headers or raw variable assignments. `ReplayProvider` can
then replay a response by rendered prompt digest for deterministic regression
tests.

```python
from promptwitness import ReplayProvider, TraceRecorder

recorded = TraceRecorder(provider)
report = executor.run(document, scenarios, recorded)
recorded.save("provider-traces.json")
replay = ReplayProvider({trace.prompt_digest: trace.output for trace in recorded.traces})
```

`load_traces()` authenticates the saved output and event digests before making
the traces available for review or replay.

The trace file is an audit inventory, not a cryptographic signature from an
external provider. Keep outputs and trace files under the same access controls as
the original provider responses.

For OpenAI-compatible gateways, `OpenAICompatibleProvider` sends rendered messages
using direct standard-library HTTP(S), reading an optional bearer token from a
named environment variable. It does not copy request authorization headers into
the response or trace metadata. Upstream response content is retained verbatim:
if the upstream provider echoes private content or a token, that content is **not**
automatically redacted. Protect returned objects and trace artifacts accordingly.

For gateways that support server-sent events, use
`OpenAICompatibleStreamingProvider`. Its `stream(row)` iterator parses only
`data:` JSON events, ignores keep-alives, stops at `[DONE]`, and rejects
malformed UTF-8/JSON or non-object chunks. Calling the provider aggregates text
content deltas and includes the decoded `stream_chunks` so a replay fixture can
audit the exact chunk sequence without copying request authorization headers.
The aggregate now preserves an explicit terminal `finish_reason`; it is `null`
when missing, including an EOF or `[DONE]` without a terminal reason. It no longer
synthesizes `stop` for truncated output. Length-limited responses remain explicitly
length-limited. Aggregation rejects multiple choices, changed choice indices,
conflicting terminal reasons, content after a terminal choice, non-assistant roles,
tool/function calls, refusals and error chunks. `stream(row)` still exposes the raw
chunks for applications that implement other streaming policies. `[DONE]` is an
SSE boundary, not proof of a successfully completed model answer.

## Bounded direct transport (unreleased development branch)

The `feat/whole-repository-alignment` version uses a fixed
`promptwitness.direct-http/v2` contract. Session and task provider identities include
that version, so an old pending reservation cannot silently reuse materially
different transport/streaming semantics. Raw non-streaming `complete()` still
returns the entire JSON object, without requiring vendor IDs, status, choices or
finish reasons. It preserves explicit incomplete/non-stop metadata unchanged;
`task_scores.response_text` is the separate strict completed-text gate.

Bounds and behavior:

- Timeout is a finite non-boolean number in `(0, 300]` seconds. A monotonic deadline
  is shared across address attempts, TLS handshake, request sending and response
  reading. Short socket polls make slow headers/bodies and blocked writes bounded
  without timeout threads. The standard-library DNS resolver itself cannot be
  interrupted: a delayed `getaddrinfo` can exceed the deadline, but no connection
  begins after the expired lookup returns. Local JSON work and SSL trust-store
  setup are also not hard CPU/RSS deadlines.
- Successful status must be HTTP 200. Redirects are never followed, environment
  proxies are never used, and no proxy authorization is sent. Non-200 responses
  are closed without reading or retaining their error bodies. Errors expose a
  status or generic diagnosis, not upstream body/status text, URLs, credentials
  or chained network/parser exceptions. TLS uses certificate and hostname
  verification from `ssl.create_default_context`; it is not disabled for tests or
  local gateways. Plain HTTP remains supported for explicitly configured gateways
  and supplies no encryption or server authentication.
- Complete outbound request headers, including request line, bearer/default
  fields, and content length, are capped at 64 KiB. Response status/headers total
  at most 64 KiB; individual status/header/chunk/trailer lines at most 8192 bytes.
  The body limit is 16 MiB for requests and successful replies, and the response
  wire reader also caps total consumed framing plus body at 32 MiB. Chunk trailers
  have their own 64 KiB cap. These are encoded/consumed-byte limits, not a hard
  allocation bound: JSON encoding can emit one already-large string at once, and
  decoding/retaining Python objects and streaming chunks uses additional memory.
- Content-length, connection-close and valid chunked replies are supported.
  Duplicate/conflicting length or transfer headers, unknown transfer encodings,
  content encodings/compression, wrong content types, broken chunk delimiters and
  incomplete chunk/trailer termination are rejected. Missing content type remains
  accepted for existing compatible gateways; if supplied, it must be JSON or SSE
  as appropriate, optionally with UTF-8 charset. Unknown metadata fields are
  preserved. JSON must be an object with unique keys, finite numbers, valid UTF-8
  strings and nesting at most 64 levels.
- Userinfo/fragments/control characters in endpoints and duplicate, injected or
  reserved routing/framing request headers are rejected. URLs are explicit caller
  configuration, not an SSRF allowlist: do not let model/user content select them.
  Custom routing headers remain supported and are pinned by their digest in
  durable provider identity. An environment bearer replaces custom Authorization
  case-insensitively. Credential values themselves are not persisted in identity.
- No retries happen inside this transport. A timeout/disconnect cannot establish
  that an external provider did no work or incurred no cost. Session/task journals
  retain their explicit uncertain-call recovery policy. Close a partially consumed
  streaming iterator to promptly release its connection.

The streaming finish-reason change is intentionally a compatibility correction:
callers that relied on fabricated successful completion must handle `null`,
non-stop reasons, or a rejected non-text aggregate. It does not assert support for
the full SSE specification or multi-choice/tool-delta assembly. Local HTTP/TLS
fault tests exercise these engineering boundaries without making model calls;
they are not model-quality or interview-quality evidence. The checked-in localhost
certificate/private key are public, disposable test fixtures, never production
credentials.

`ToolDispatcher` handles a provider response's `tool_calls` array under explicit
call-count and result-size limits. Arguments are parsed as JSON objects, unknown
tools and handler exceptions become per-call outcomes, and successful results
can be converted to follow-up tool messages through `ToolBatch.messages()`.
Only argument/result digests are required for audit storage; callers should
apply their own authorization before registering side-effecting handlers.
