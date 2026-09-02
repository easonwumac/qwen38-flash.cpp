# HTTP API

All endpoints bind to `127.0.0.1` by default. The server handles one HTTP/1.1
request per connection and closes it after the response. Four bounded HTTP
workers let health, status, and metrics run while generation is active; the
pending socket queue is capped at 128 and returns `503 server_busy` when full.
Native inference is serialized through one dedicated MLX executor thread, so
model construction, evaluation, cache mutation, and destruction retain stream
affinity without duplicating weights. Authentication and concurrent model
scheduling remain later milestones.

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

Completion requests accept a string `prompt` and a positive `max_tokens` up to
the server's `--max-generation-tokens` limit (4096 by default). The combined
prompt and requested generation must also fit the model context window. Chat
requests accept a non-empty OpenAI-style `messages` array and either
`max_completion_tokens` or `max_tokens`. The current engine is greedy-only;
`enable_thinking` and the mlx-serve-compatible alias `thinking` select the Qwen
thinking template; when both are present they must agree.
OpenAI-style `tools`, assistant `tool_calls`, and `role: "tool"` results follow
the checkpoint's native Qwen3.8 XML template. Responses translate valid calls
back to OpenAI `tool_calls` with `finish_reason: "tool_calls"`; parameter values
use the declared JSON schema to preserve string IDs and paths. Consecutive tool
results are grouped into the single user turn expected by the model.
`stream: true` returns live HTTP/1.1 chunked SSE for both completion routes.
Confirmed target-token text is emitted from the decode path, the final event
contains `finish_reason`, and the stream terminates with `data: [DONE]`.
`stream_options: {"include_usage": true}` adds usage to the finish event.
Closing an SSE connection cancels native generation at the next committed-token
boundary. The `requests.cancelled` status field and
`qwen38_requests_cancelled_total` metric count these requests.
Non-streaming responses include a non-standard `performance` object with prompt
time, generation time, measured generation tok/s, and an `mtp` object containing
round, proposal, acceptance, and fallback counts. The four-element
`proposed_by_position` and `accepted_by_position` arrays count draft positions
one through four; unused positions remain zero. They include learned and history
draft rounds, so disable history drafting when using them to tune the learned
depth policy.
With the diagnostic `QWEN38_MTP_TOP2_ORACLE=1`, the corresponding
`top2_rejected_by_position` and `top2_recovered_by_position` arrays report
whether the final-position second choice could have repaired the first rejected
draft. Use `all` instead of `1` to inspect every position. The oracle performs
extra vocabulary-wide selection and is intentionally unsuitable for throughput
measurements.
When `QWEN38_CONTEXT_COPY=1`, `performance.context_copy` reports its separate
round, proposal, acceptance, and suspension counters. Long copy blocks still
contribute to aggregate `mtp.proposed`/`mtp.accepted`; the four positional arrays
intentionally describe only positions one through four.
`mtp.profitability_cache_keep` marks a replay that retained cumulatively
profitable MTP despite a late fallback; `mtp.profitability_cache_skip` marks a
cached losing probe that was skipped.

```bash
curl -N http://127.0.0.1:11438/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"stream":true}'
```

Chat streams start with an assistant role delta. With thinking enabled, thought
tokens use `reasoning_content` and answer tokens use `content`; completion
streams use `text`. The server sets `Cache-Control: no-cache` and
`X-Accel-Buffering: no` so intermediaries do not intentionally buffer events.

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
