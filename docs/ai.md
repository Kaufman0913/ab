# Making requests to Chutes

Inference:

```bash
curl -X POST "https://llm.chutes.ai/v1/chat/completions" \
    -H "Authorization: Bearer $$$" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "moonshotai/Kimi-K2-Instruct",
      "messages": [
        {"role": "user", "content": "Testing"}
      ],
      "stream": true,
      "max_tokens": 2048,
      "temperature": 0.7,
      "seed": 1234567890
    }'
```

This follows OpenAI v1, just use the library to do it.

Embedding:

```bash
curl -X POST "https://chutes-baai-bge-large-en-v1-5.chutes.ai/embed" \
    -H "Authorization: Bearer $$$" \
    -H "Content-Type: application/json" \
    -d '{
      "inputs": "Text you want to embed",
      "seed": 123456789
    }'
```

This doesn't follow OpenAI v1, do it directly.

# Making requests to Targon

TODO