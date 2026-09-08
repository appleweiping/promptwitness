# Multimodal prompt content

PromptWitness keeps provider content blocks as structured `ContentBlock` values
instead of silently converting an image, audio clip, or document into text.
`Message.content` remains the extracted text view used by variable inspection
and validation; `Message.content_parts` is the lossless provider-facing view.

```python
from promptwitness import AdapterFormat, adapt_prompt, prompt_to_dict

result = adapt_prompt(
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
                ],
            }
        ]
    },
    AdapterFormat.OPENAI,
)
message = result.document.messages[0]
assert message.content == "Describe this image."
assert [part.type for part in message.content_parts] == ["input_text", "image_url"]
assert prompt_to_dict(result.document)["messages"][0]["content"][1]["type"] == "image_url"
```

The native schema accepts either a string or an array of objects with a
non-empty `type`. Text-like blocks (`text`, `input_text`, and `output_text`)
must contain a string `text` field. Other fields are retained as JSON data and
are validated for basic shape; unsupported provider semantics are not guessed.
OpenAI, Anthropic, and LangChain adapters share this representation. Matrix and
registry rendering substitutes variables inside text blocks while leaving
binary/media references unchanged. Provider wrappers emit the preserved block
array on the wire.

Text extraction is intentionally a view, not a replacement: when multiple text
blocks exist, their text is joined with newlines for variable checks and
human-readable diffs, while the original block boundaries and non-text fields
remain available for serialization and provider calls.
