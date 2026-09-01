# HTTP API

All endpoints bind to `127.0.0.1` by default. The current foundation handles one
HTTP/1.1 request per connection and closes it after the response. Concurrency,
streaming, inference, authentication, and cache mutation arrive in later
milestones.

| Method | Route | Current behavior |
|---|---|---|
| `GET` | `/healthz` | Process health; returns 200 while the server can respond. |
| `GET` | `/readyz` | Returns 200 only when a model is ready, otherwise 503. |
| `GET` | `/v1/status` | Runtime state, uptime, request counters, token counters, and last error. |
| `GET` | `/v1/models` | OpenAI-style model list; empty until a model is ready. |
| `GET` | `/metrics` | Prometheus text metrics. |
| `POST` | `/v1/chat/completions` | Reserved; returns structured 503 until inference exists. |
| `POST` | `/v1/completions` | Reserved; returns structured 503 until inference exists. |
| `POST` | `/admin/cache/clear` | Reserved; returns 503/501 until cache ownership exists. |

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
