# Provider adapters

Provider request formats evolve independently from PromptWitness. Adapters therefore
support documented subsets and report information loss instead of pretending to
mirror every SDK object.

## OpenAI-shaped requests

Supported fields are `messages`, function `tools`, optional `id`/`name`, and object
`metadata`. Message content may be a string or an array containing only `text`,
`input_text`, or `output_text` blocks. Function parameters must be an object schema;
its `properties` and `required` become the native tool contract.

Multiple text blocks are joined with newline separators because native schema version
1 stores one string per message. The adapter reports this boundary flattening, along
with any block fields that were not represented.

String message IDs are preserved when supplied, so converted prompts can opt into
ID-based alignment without losing provider identity. Request controls such as model,
temperature, response format, and tool choice are
outside the prompt document and are reported as unrepresented top-level fields.
Non-text blocks and non-function tools fail conversion.

## OpenAI Responses API requests

Use `AdapterFormat.OPENAI_RESPONSES` or `--from-format openai-responses` for the
Responses API shape with `input` and optional `instructions`. A string input becomes
a user message; heterogeneous input items retain message content, function calls,
function-call outputs, reasoning, and other response events as native
`ContentBlock` values. Item IDs are preserved for message alignment, and function
tools use their top-level Responses fields (`name`, `description`, and
`parameters`). Built-in tools such as web search are reported as warnings because
the native schema models callable function contracts only. Request controls and
unknown event fields remain explicit warnings rather than being discarded silently.
The repository includes a runnable fixture at
`examples/openai-responses-request.json`, which can be converted with:

```bash
promptwitness convert examples/openai-responses-request.json \
  --from-format openai-responses
```

## Anthropic-shaped requests

String or text-block `system` content becomes the first system message. `messages`
accept string or text-block content. Each tool's object `input_schema.properties`
and `required` become a native tool.

String message IDs are preserved when supplied. Anthropic text, image, document,
tool-use, and tool-result blocks are retained as structured `ContentBlock` values;
their text portions are exposed through the message analysis string while binary
and tool fields remain available for round-trip rendering.

## Gemini / Vertex `generateContent` requests

Use `AdapterFormat.GEMINI` or `--from-format gemini` for requests containing
`contents`, optional `system_instruction`/`systemInstruction`, and Gemini
`tools.function_declarations`. A missing content role defaults to `user`, while
the provider's `model` role is normalized to native `assistant`. Text,
inline/file data, function-call, function-response, executable-code, and
code-execution-result parts are preserved as provider-native `ContentBlock`
values; text is also exposed through the message's analysis string.

String content IDs are preserved when supplied. Multiple function-declaration groups
are flattened into the native tool list
while preserving descriptions, JSON parameters, and required fields. Generation
controls and unknown fields are not silently interpreted: they are returned as
adapter warnings. External references and remote schema fetching remain
intentionally unsupported.

## LangChain-shaped templates

The adapter accepts a portable subset with `messages` whose role is in `role`,
`type`, or `_type` and whose text is in `content`, `template`, or
`prompt.template`. `human` maps to `user`; `ai` maps to `assistant`.
Optional string `name` and provider-native string `id` values are preserved. Conflicting role aliases or competing content
fields fail conversion rather than relying on an implicit precedence rule. Other
message and nested-prompt fields are named in adapter warnings.

If `input_variables` is present, it is compared with variables observed in message
templates and a warning records disagreement. Arbitrary runnable graphs, partial
variables, output parsers, and Python-serialized objects are not executed.

## Auto detection and warnings

Auto detection recognizes native `schema_version`, OpenAI Responses `input` with
request markers, Anthropic system/input-schema keys, Gemini `contents`/
`system_instruction` keys, LangChain template or message-role alias keys, then
messages-only OpenAI-shaped input. Choose an explicit `--from-format` in long-lived
CI to avoid relying on detection precedence.

Warnings go to stderr and are also returned by the Python `AdapterResult`. Conversion
does not copy ignored values into metadata, which reduces accidental credential or
request-state retention.
