# Raw requests / re-tokenizable text

- `llm_calls.jsonl`：每次模型调用一行（JSONL）。
- `request.retoken_text`：把所有 message 的纯文本内容拼接后的字符串；官方可用指定 tokenizer 重分词。
- 图片不参与重分词统计：`request.messages` 里 `image_url.url` 已脱敏为占位符；对应图片一致性用 `context.image_sha256` 校验。
- `tokenizer.prompt_tokens`：如果本地安装了 tiktoken，会用 `cl100k_base` 计算提示词 token；否则为 null，最终以官方复算为准。
