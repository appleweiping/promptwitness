# Stable message identifiers

Native prompt documents may give each message a stable `id`:

```json
{
  "schema_version": 1,
  "id": "support-v2",
  "messages": [
    {"id": "system", "role": "system", "content": "You are helpful."},
    {"id": "question", "role": "user", "content": "{{question}}"}
  ]
}
```

IDs are optional for backwards compatibility, but must be non-empty, at most
128 characters, and unique within a document. Use
`promptwitness diff --message-alignment id` when revisions reorder messages:
the comparison follows IDs instead of treating a reorder as a cascade of
positional edits. The mode intentionally rejects documents with missing IDs
so a release gate cannot silently fall back to a heuristic.
