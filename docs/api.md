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
| `POST` | `/admin/cache/clear` | Clears engine-owned reusable caches; currently a no-op because decode state is request-owned. |

Completion requests accept a string `prompt` and `max_tokens` from 1 through
256. Chat requests accept a non-empty OpenAI-style `messages` array and either
`max_completion_tokens` or `max_tokens`. The current engine is greedy-only;
`stream: true` returns 400. Responses include a non-standard `performance`
object with prompt time, generation time, and measured generation tok/s.

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
