# Tool-call validation

PromptWitness can validate a concrete JSON tool invocation before dispatch.
The check uses the declared tool parameters and required list, supports the
same bounded schema keywords used by the diff validator, and resolves local
`$ref` pointers by default.

```bash
promptwitness check-call examples/before.json lookup_order call.json
```

The command emits a stable JSON report and returns `0` for valid arguments or
`2` for a valid report containing findings. Unknown tools, malformed argument
files, and external/cyclic schema references are invocation errors. The API is
`validate_tool_arguments(tool, arguments, resolve_refs=True)`.
