# HTTP API

All endpoints bind to `127.0.0.1` by default. The current foundation handles one
HTTP/1.1 request per connection and closes it after the response. Native
inference is serialized through one engine mutex. Streaming, authentication,
and concurrent scheduling remain later milestones.

| Method | Route | Current behavior |
|---|---|---|
| `GET` | `/healthz` | Process health; returns 200 while the server can respond. |
| `GET` | `/readyz` | Returns 200 only when a model is ready, otherwise 503. |
| `GET` | `/v1/status` | Runtime state, uptime, request counters, token counters, and last error. |
| `GET` | `/v1/models` | OpenAI-style model list; empty until a model is ready. |
| `GET` | `/metrics` | Prometheus text metrics. |
| `POST` | `/v1/chat/completions` | Native chat-template, tokenization, greedy generation, usage and timing. |
| `POST` | `/v1/completions` | Native tokenization and greedy generation, usage and timing. |
| `POST` | `/admin/cache/clear` | Clears unused MLX allocator/cache blocks; active request state is serialized and never invalidated. |

Completion requests accept a string `prompt` and `max_tokens` from 1 through
256. Chat requests accept a non-empty OpenAI-style `messages` array and either
`max_completion_tokens` or `max_tokens`. The current engine is greedy-only;
`stream: true` returns 400. Responses include a non-standard `performance`
object with prompt time, generation time, measured generation tok/s, and an
`mtp` object containing round, proposal, acceptance, and fallback counts.

Thinking responses expose the text before `</think>` as `reasoning_content` and
the final answer as `content`. Send both fields back on assistant messages in a
later request so the native Qwen chat template can reproduce the prior turn.
Both configured Qwen EOS IDs terminate generation and are omitted from response
content. The `speed` profile extends its one-entry prefix cache through generated
assistant tokens; `QWEN38_EXTEND_PREFIX_CACHE=0` restores prompt-only caching.

Errors use an OpenAI-style envelope:

```json
{
  "error": {
    "code": "model_not_ready",
    "message": "The model is not ready",
    "type": "server_error"
  }
}
```
