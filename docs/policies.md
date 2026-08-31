# Policy files

Policy schema version 1 is a strict JSON object with optional `validation` and
`diff` objects. Unknown and duplicate keys fail.

```json
{
  "schema_version": 1,
  "validation": {
    "allowed_roles": ["system", "user", "assistant", "tool"],
    "require_system_first": true,
    "allow_empty_content": false,
    "report_repeated_variables": true,
    "scan_literal_secrets": true
  },
  "diff": {
    "message_alignment": "smart",
    "content_change_severity": "warning",
    "added_message_severity": "warning",
    "include_metadata": true,
    "context_lines": 2,
    "severity_overrides": {
      "tool_added": "warning",
      "message_content_changed": "breaking"
    }
  }
}
```

Severity values are `info`, `warning`, or `breaking`. Override keys must be stable
`ChangeKind` values listed in JSON reports. Overrides are applied after structural
classification, so they affect report counts, compatibility, SARIF levels, and CLI
gates consistently.

Keep policy files in source control and review their diffs with prompt changes. A
weaker policy can make CI pass without making a prompt safer.
