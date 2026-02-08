# Raw requests / re-tokenizable text

- `llm_calls.jsonl`: one line per model call (JSONL format).
- `request.retoken_text`: the concatenated plain-text content of all messages; the official evaluator may re-tokenize it with a specified tokenizer.
- Images are not included in the re-tokenization statistics: `request.messages` entries have `image_url.url` replaced by placeholders; consistency of the corresponding images is checked using `context.image_sha256`.
- `tokenizer.prompt_tokens`: if tiktoken is installed locally, it uses `cl100k_base` to compute prompt tokens; otherwise this field is null, and the final count is subject to the official recomputation.
