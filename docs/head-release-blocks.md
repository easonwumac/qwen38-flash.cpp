# Head lifetime and segmented append feasibility

## MTP head lifetime change

The runner restores the pre-draft head snapshot immediately before target
verification, instead of after verification. Both lazy and eager draft paths
have already evaluated the draft token IDs before reaching this boundary.
Verification reads the target state, not the speculative head. Calibration
streams own their array handles independently. Head reconstruction, target
acceptance and correction logic, and all kernels are unchanged.

This avoids retaining unused speculative head KV/QSA during target verification.
The restore is timed within verify_ms (previously commit_ms), so phase timing
comparisons must account for that relocation. No added synchronization or cache
clear. On a verification exception the head remains at the pre-draft origin;
this is not a new whole-round transactional exception guarantee.

Ownership tests verify the restored origin values/position and independent
calibration-stream lifetime. All five CTests pass. These are model-free tests,
not end-to-end MTP parity or a demonstrated physical-memory/speed improvement.
Full-model validation is still required before deployment.

## Loader audit

Current QwenModel construction builds layers sequentially and loads quantized
arrays directly. MlxSafetensors uses MLX Load nodes; the inspected pinned MLX
source creates lazy array primitives, not eagerly dequantized full shards.
Fused MoE gate/up refers to a kernel, not a second packed full-model copy.
This inspection does not prove zero transient duplication, but it gives no
evidence for a blanket loader rewrite. No loader behavior changed.

## Segmented append prototype (not integrated attention)

Developer-only `qwen38-state-lifetime-probe blocks` retains the initial history
and adds independent BF16 blocks without concatenating them. `concat` retains
an origin snapshot and appends conventionally. Both now generate/evaluate each
new update inside the timing loop; this is a revised workload and its timings
must not be mixed with the older state-lifetime-results.json.

Synthetic shape: initial 32,768 x 256 BF16 history, 32 updates of 512 rows.
M5 Pro, 64 GiB; no model, sampling, MTP or attention; no thermal control.
A/B/B/A separate processes, seven repetitions each, discard the first:

| Mode | Warm allocator peak | Warm time range, 32 updates |
| --- | --- | --- |
| concat | 64 MiB | 31.0–41.7 ms |
| blocks | 24.75 MiB | 6.2–8.9 ms |

Full values of history and every appended block pass checks; origin is unchanged.
Peak is allocator memory for this tiny synthetic workload, not process RSS.
Both have the same logical token history, but different physical representations.
**This is not a PP/decode speedup.** The existing attention kernels require
contiguous state, so concatenating before attention would give back the benefit.
Block-aware QSA gathering, attention reads, rollback, and SSD serialization are
required before this can be used by the server. The append evidence justifies
that next prototype; it does not justify changing the serving defaults.

Safety: serial probes, strict MLX allocation cap 1 GiB/cache 16 MiB; memory guard
start 16 GiB, available floor 12 GiB, RSS/footprint caps 4 GiB. No server restart,
model loading, runtime-profile changes, or safety-threshold changes.
Raw outputs: head-release-blocks-results.json.
