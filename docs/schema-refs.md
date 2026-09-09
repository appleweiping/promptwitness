# Local tool-schema references

Tool parameter diffs are conservative by default and compare the stored
schemas exactly. Use `--resolve-tool-refs` with `promptwitness diff` when a
tool parameter contains local `$ref` pointers into its own JSON Schema object.
Only `#`/`#/...` fragments are accepted; remote URLs, missing targets, and
cycles fail explicitly. The resolver never performs network access.

The Python API exposes the same behavior through
`DiffOptions(resolve_tool_refs=True)` and `resolve_local_refs`. Sibling keys
next to `$ref` apply together with the referenced object. When necessary, the
expanded schema uses `allOf` so a sibling cannot weaken referenced constraints
or change the property scope of `additionalProperties`.
