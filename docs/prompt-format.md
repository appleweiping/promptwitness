# Prompt format version 1

The top-level value is a JSON object with these fields:

| Field | Required | Type | Meaning |
|---|---:|---|---|
| `schema_version` | yes | integer | Must equal `1` |
| `id` | yes | string | Non-empty revision identifier |
| `messages` | yes | array | Ordered message templates |
| `tools` | no | array | Named function-style interfaces |
| `metadata` | no | object | Compared as opaque JSON-compatible data |

Unknown top-level and nested fields are errors. The strictness catches a typo
such as `message` instead of `messages` before it produces an empty prompt.

## Messages

A message requires string `role` and `content`; optional `name` is a string or
null. Parsing accepts any non-empty role so a custom policy can support provider
roles. Default validation recognizes `system`, `user`, `assistant`, and `tool`.

Variables use `{{ name }}`. Names start with a letter or underscore and may
then contain letters, digits, underscores, dots, or hyphens. Variable spacing is
not significant. PromptWitness does not support expressions, loops, filters,
conditionals, or attribute evaluation.

## Tools

A tool contains:

- non-empty string `name`, unique within the document;
- string `description`, which validation warns about when empty;
- object `parameters`, mapping parameter names to schema-like values;
- array `required`, whose unique strings must all exist in `parameters`.

Default validation expects each parameter value to be an object with a JSON
Schema primitive/container `type`: `array`, `boolean`, `integer`, `null`,
`number`, `object`, or `string`. An optional `enum` must be a non-empty array.
Other keys are preserved and compared but not interpreted in version 1.

## Reports

JSON reports use `schema_version: 1` and `report_type: "validation"` or
`"diff"`. Change and finding categories are lowercase stable enum values. New
fields may be added compatibly; consumers should reject unknown schema versions
but ignore unknown object fields within a known version when safe.
