# VQ throughput follow-up — 2026-09-19

Outcome: **no new runtime optimization promoted**. The three candidates failed
to establish an end-to-end gain; all experimental runtime hooks were removed.
Retain the safer, more comparable benchmark tools and the negative evidence.
This does not establish a hardware ceiling.

This round tests execution/layout changes, not new weights, pruning, lower
top-k or codebook precision. Source baseline: `dc33cd2`.
The existing 519.91 PP / 30.59 fixed-input serial / 57.94 favorable MTP
figures are historical workload-specific measurements, not results of these
component probes.

## Component gates

Every row compares complete MoE or complete attention, not just the changed
kernel. Lower milliseconds is better. The QSA tests use real layer-3 weights
with deterministic **synthetic** history; they are not retrieval-quality or
full-model throughput tests.

| Candidate | Workload | Control median ms [P10, P90] | Candidate median ms [P10, P90] | Result |
|---|---|---:|---:|---|
| Padded shared-memory tiles | MoE layer 2, 2,048 prompt rows | 54.484 [53.739, 55.207] | 54.320 [54.040, 55.450] | 0.3% movement, worse P90; removed |
| Flatten four QSA heads into matrix rows | 32K, one decode row | 0.9004 [0.8809, 1.1051] | 0.9915 [0.9716, 1.4460] | 10.1% slower; removed |
| Same head layout | 128K, one decode row | 1.6679 [0.8865, 1.9636] | 1.8712 [1.4352, 2.0516] | 12.2% slower; removed |
| Same head layout | 32K history, 512 prefill rows | 34.335 [34.046, 34.593] | 34.406 [34.159, 34.653] | No gain; removed |
| Retain QSA pools in FP32 after original BF16 rounding | 32K, one decode row | 0.9051 [0.8843, 1.4483] | 0.8266 [0.7993, 1.2493] | 8.7% lower latency; full-model gate below rejects it |
| Same pool storage | 128K, one decode row | 1.6737 [1.3380, 1.9183] | 1.6106 [1.2316, 2.1624] | 3.8% lower median, worse tail; not sufficient alone |

All rows retain exact FP32 output-value bits; QSA also checks the raw/pooled
index state and hot K/V values. FP32 pool storage intentionally changes its
storage dtype but not the represented values. A/B/B/A samples are interleaved
in one process. MoE uses two warmups and 20 measured samples per arm. QSA uses
20 warmups and 60 measured samples per arm, with phase barriers **off**.
Median averages the two central samples; P10/P90 use the sorted sample at
`round((N-1)*p)`. These are sample distributions, not confidence intervals.
Background scheduling noise remains visible, especially in 128K probes.

MoE input uses token `(9419 + row * 7919) % vocabulary`. QSA uses the same
embedding fixture, layer-3 attention mixing and deterministic synthetic BF16
history. Decode uses Q8 cold K/V, QSA budget 512 and raw window 64; prefill
uses its original 2,048-token budget. No sampling or MTP head runs in these
component tests. Their 1–2 GiB footprints are not model-serving RAM claims.

## Why these changes were tested

- **Tile padding:** keep identical weight reconstruction and matrix arithmetic,
  change threadgroup strides by eight elements to test layout conflicts.
  Complete MoE does not improve beyond noise, so no new runtime flag remains.
- **Head flattening:** use ordinary MLX matrix multiplication with four heads
  flattened into rows, rather than another custom fused score kernel. Outputs
  match, but the different GEMM shape is slower for decode and flat for PP.
- **FP32 pools:** norm and RoPE still finish at the original BF16 boundary.
  Widen those completed values once instead of converting the entire retained
  pool on every token. This is not higher-precision attention or a new numeric
  policy. At 128K, twelve layers of 128-dimensional, ratio-four pools require
  about 96 MiB of extra retained storage; the complete memory gate still applies.

Older local research had already rejected exact dual-SIMD gate/up tail sharing
(4–6% slower), OTILE64, stable CPU counting and bounded BF16 expert
materialization. Those experiments were not rerun. Their detailed evidence is
summarized in the research ledger so these dead ends remain searchable.

## Full-model gate

**Do not compare this 32K target-only test with the 35.28 tok/s IFBench figure:
that figure uses native MTP across thirty prompts.** The retained short-context
fixed-input target-only result is 30.59 tok/s; the favorable native-MTP coding
fixture is 57.94. These are separate workloads and execution policies, not a
single speed regressing from 35 to 21.

After restoring the original core, three fresh short-context processes measured
**29.28 / 29.63 / 29.49 tok/s** (29.49 median, about 36.3 GiB peak footprint).
This uses the same fixed-input token 9419 / 64-step probe, excludes its first
two warmup steps, and has no MTP. The historical 30.59 and today's 29.49 are
not a contemporaneous A/B; thermal/background conditions were not fixed.
The core source has no difference from `dc33cd2`. These numbers do not establish
a newly introduced 3.6% runtime regression, but the current measurements are
retained rather than silently replaced with the historical best.

The first long-context control accidentally used the core's BF16 KV fallback:
the developer probe did not apply the server CLI's Q8 defaults. The memory
guard stopped it at 40.224 GiB footprint, before a completed measurement. This
is a failed control run, **not** a candidate speed or memory result. The tool
now matches server defaults: Q8 KV from 8,192 tokens, flushing at 2,048 tokens.
The 40 GiB guard was not raised.

The corrected gate uses the fixed numbered-notes corpus (2,000 lines), greedy
non-thinking generation, max 128 tokens, MTP off, no prefix cache, automatic
bounded prefill, an A/B warmup and three measured samples per arm.
Results (three measured requests per arm; warmups excluded):

| Metric | Control | FP32 pool candidate |
|---|---:|---:|
| PP tok/s, each request | 422.08 / 374.26 / 383.01 | 412.53 / 380.83 / 358.89 |
| PP median | **383.01** | 380.83 |
| Decode tok/s, each request | 22.23 / 21.87 / 18.66 | 21.11 / 21.65 / 19.90 |
| Decode median | **21.87** | 21.11 |
| Aggregate decode (tokens / total generation time) | 20.79 | 20.86 |

The median decode rate decreases 3.5%, while aggregate changes only +0.35%.
Together with substantial run-to-run variation, this is **not a reliable
speedup**. All eight requests, including warmups, emit the same 29 tokens.
Prompt count is 32,024; the fixed prompt-token checksum is
`4883467297351314722` and generated-token checksum is `2434226475115966986`
(the probe's 64-bit rolling checksum, not a cryptographic identity).
The original 32K headline of 22.68 tok/s remains a historical measurement;
this round's two arms are compared only against each other. The lower PP
numbers are long-context results, not the 7K PP headline workload.

The corrected session exits normally: **39.611 GiB peak physical footprint**,
30.495 GiB peak RSS, 10.324 GiB minimum reclaimable memory. This is one combined
guarded session, not separate per-arm peak-memory attribution. Pool widening
and its opt-in hook are removed. No checkpoint, precision, default runtime
behavior, or historical quality score changes.

## Verification

- Clean release rebuild succeeds (existing duplicate-static-library linker warning).
- CTest: 6/6, including eight new model-free probe CLI safety tests.
- IFBench runner CLI: 5/5; long-context telemetry: 3/3; MTP benchmark tooling: 7/7.
- Rebuilt real-weight QSA rollback check: 16 cases, 1,152 continuation steps,
  exact output and QSA state across BF16/Q8, peak footprint 0.869 GiB.
- Rebuilt QSA A/B tool: 32K synthetic history plus 512 prefill rows, 60 samples
  per arm using the existing research-only score reduction; exact output/state
  bits, no phase barriers, same `8095078098800903043` checksum as the control.
  This validates the measurement tool, not a newly promoted score kernel.
- An initial CTest invocation used a model path relative to the wrong working
  directory; only tokenizer setup failed. The corrected path passes all six.
- The A/B probe explicitly validates output/state bits outside the measured
  interval and reports separate arm arrays. The runtime probe records prompt
  token checksums and server-equivalent Q8 defaults.

No new full IFBench/HumanEval score is claimed: runtime candidates were removed
before qualification, and the retained serving implementation is unchanged.

## Environment and evidence

Apple M5 Pro, 64 GiB, macOS 26.5, MLX 0.32.2, AC power. No active thermal control;
`pmset` reported no thermal/performance warning before testing. Normal OS
background activity was not disabled. Only one guarded GPU process ran at a
time. No model copy, new checkpoint, ANE backend or worktree was created.

Raw samples and exact integer hashes are in
[the machine-readable evidence](results/vq-throughput-round2-2026-09-19.json).
The QSA probe now makes phase barriers opt-in and tests interleaved output/state
parity. The runtime probe records prompt-token hashes and supports the fixed
long-context workload; these are developer diagnostics, not serving profiles.
