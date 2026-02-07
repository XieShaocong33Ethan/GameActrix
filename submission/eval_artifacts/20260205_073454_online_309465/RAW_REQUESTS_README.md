# Raw Requests and Retokenizable Text

This directory includes `llm_calls.jsonl`, where each line records one model call in JSONL format.

- `request.retoken_text` is the concatenation of the plain text parts of all messages. It is designed to be re-tokenized
  by the evaluator using a specified tokenizer.
- Images are excluded from retokenization. In `request.messages`, any `image_url.url` is replaced by a placeholder. Image
  consistency can be verified with `context.image_sha256`.
- `tokenizer.prompt_tokens` is computed with `tiktoken` using `cl100k_base` when available. If not available, it is
  recorded as null and should be recomputed by the evaluator.
