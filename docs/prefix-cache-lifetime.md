# SSD prefix-cache lifetime

For a newly prefetched cacheable prompt, the native engine writes the prompt
checkpoint before decode. If the store reports that the exact entry survived
quota eviction, both RAM snapshot owners are released. A capacity refusal or
write exception keeps both owners and preserves the end-of-request retry.

Scope and tradeoffs:

- Exact RAM hits and SSD-loaded hits bypass new-prompt snapshot creation, so this
  release applies only to newly computed prompt checkpoints.
- The early SSD write is on the response path and adds to time-to-first-token,
  and its synchronous elapsed time is included in `prompt_ms`. The layer prefill
  diagnostic remains compute-only, while API prompt/PP timing includes the write.
- Normal successful end-state extension recreates the RAM cache and computes
  the same completed MTP profitability metadata for that extended entry.
- If extension is impossible, including some cancellation or nonextendable MTP
  states, no prompt-only RAM snapshot remains; a later request must reload the
  prompt checkpoint from SSD.
- In that nonextended case, prompt-only MTP profitability learned during the
  request is not written back. A later SSD hit may repeat profitability probing
  or fallback. This is a cache-performance regression, not a token-semantic or
  model-quality change. Restoring it would require a metadata-only store update;
  retaining or reloading the full prompt state would defeat the lifetime goal.

Tests cover retained-save success and exact reload, quota refusal, empty input,
and an actual filesystem write exception. There is no direct NativeEngine
fixture for cancellation or extension lifecycle, so those paths are not claimed
as integration-tested here.
