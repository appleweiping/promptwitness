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

Request controls such as model, temperature, response format, and tool choice are
outside the prompt document and are reported as unrepresented top-level fields.
Non-text blocks and non-function tools fail conversion.

## Anthropic-shaped requests

String or text-block `system` content becomes the first system message. `messages`
accept string or text-block content. Each tool's object `input_schema.properties`
and `required` become a native tool.

Image, document, tool-use, and tool-result blocks cannot be represented in native
schema version 1 and fail conversion instead of being flattened.

## LangChain-shaped templates

The adapter accepts a portable subset with `messages` whose role is in `role`,
`type`, or `_type` and whose text is in `content`, `template`, or
`prompt.template`. `human` maps to `user`; `ai` maps to `assistant`.
An optional string `name` is preserved. Conflicting role aliases or competing content
fields fail conversion rather than relying on an implicit precedence rule. Other
message and nested-prompt fields are named in adapter warnings.

If `input_variables` is present, it is compared with variables observed in message
templates and a warning records disagreement. Arbitrary runnable graphs, partial
variables, output parsers, and Python-serialized objects are not executed.

## Auto detection and warnings

Auto detection recognizes native `schema_version`, Anthropic system/input-schema
keys, LangChain template or message-role alias keys, then messages-only OpenAI-shaped
input. Choose an explicit `--from-format` in long-lived CI to avoid relying on
detection precedence.

Warnings go to stderr and are also returned by the Python `AdapterResult`. Conversion
does not copy ignored values into metadata, which reduces accidental credential or
request-state retention.
