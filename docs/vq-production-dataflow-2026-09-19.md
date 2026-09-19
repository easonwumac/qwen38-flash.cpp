# VQ production dataflow — 2026-09-19

**Outcome: useful persistent-backend foundations, but no new production speed
record.** All three requested directions received an implementation and test.
The persistent VQ path remains opt-in: the three-case strict gate passes after
fixing an inherited EOS-policy bug, but broad native quality is not qualified. The
compact verifier and GPU-indirect prefill prototypes were slower and removed.
The original requirement to deliver three qualified improvements remains open.
This is not evidence that the hardware has reached its limit.

## Results

| Direction | Control | Candidate | Decision |
|---|---:|---:|---|
| Whole-token native activation flow | 30.989 ms/token [30.566, 31.609] | 30.473 [30.012, 31.141] | 1.7% lower median in native-vs-native diagnostic; not a production comparison |
| Persistent native IFBench development gate | MLX: 3/3 strict, 3/3 loose | Native: **3/3 strict, 3/3 loose** after EOS fix | Small regression gate passed; not full quality qualification |
| Compact shared-weight verifier MoE, S=5 | 0.831 ms/block | 2.643 ms/block | 3.18x latency; reject before full verifier/quality tests |
| GPU-indirect segmented MoE, 2,048 rows | 55.949 ms [55.666, 56.497] | 58.976 [58.366, 59.600] | Bit-exact but 5.4% slower |
| GPU-indirect full-model PP, 7,224 tokens | 497.61 tok/s [496.48, 499.13] | 478.39 [477.60, 478.86] | Same generated tokens, 3.9% less throughput; reject |

Brackets are P10/P90, not confidence intervals. Persistent timing uses three
measured arms per variant, 62 timed steps per arm. Segmented MoE uses 20
interleaved samples per variant. Full PP uses three samples per variant after
one warmup each. Compact-verifier medians come from separate processes, 43 warm
iterations after one cold iteration; raw distributions were not retained, so
they support rejection of this prototype, not a precise product-speed claim.

The production 519.91 PP / 30.59 fixed-input target-only / 57.94 favorable MTP
figures remain historical, workload-specific results. They are not replaced by
these diagnostics. In particular, neither the native 32.8 tok/s reciprocal
timing nor IFBench's different completion lengths establish a production gain.

## 1. Persistent backend: retained, experimental

Implemented:

- Keep embedding, 48-layer trunk, final mixer and head activations in existing
  Metal buffers; queue layer groups and wait at token completion. SSD PLE lookup
  still uses the host; this is not an all-GPU n-gram system.
- Two-stage GPU top-2 reduction. Only two IDs/values cross back to the host.
- Owning export of GDN recurrence/convolution, hot BF16 and cold Q8 KV, QSA
  pending/pooled keys, PLE convolution/ngram history, and the pre-mixer stream.
- Materialize MLX's deferred QSA pools before native import. Short prefill had
  legitimately deferred those pools, but native append assumed they existed.
- Validate state geometry/frontiers before import and reject native hot-slab
  exhaustion before a token mutates state. The server can export to MLX for its
  existing Q8 flush. Old cold buffers are released on BF16 import/reset.
- Release shared Metal weight wrappers on normal return, cancellation and
  exceptions; drain committed commands before an encoding exception unwinds.
- Target-only generated-prefix extension and cache compatibility fencing for
  native-vs-MLX execution. Native MTP state export is **not** qualified.

Native arithmetic also needed alignment: FP32 VQ route weights and route
accumulation, BF16 router/head rounding, reference RoPE multiplication
boundaries, and safe Metal math. These changes do **not** prove complete
native/MLX equivalence. Independent projection/reduction implementations still
need numerical investigation; a high cosine score is not a quality gate.

**The strict IF3 failure had a concrete serving-policy cause:** native greedy
historically forced at least eight output tokens. When the model selected EOS
earlier, the server substituted its second-best token. Key 100 therefore wrote
`maybe` followed by an unsolicited explanation. The default is now zero minimum
tokens, matching ordinary MLX greedy stopping. The otherwise identical native
build recovered 3/3 strict and loose; key 100 emits one token, `maybe`, and stops.
Explicit developer minimum/anchor overrides are included in cache fingerprints.
This bug was in the experimental native path, **not** the default MLX/MTP path;
it does not explain or revise historical production IF30/HumanEval results.

Verified:

- Eight exact state roundtrips: 3/4/7/511/513/8,192 tokens, 8,193 with 2,048
  Q8-cold tokens, and 2,048 entirely cold tokens. Q8 packed words are nonzero.
- Exported arrays remain unchanged after re-import. Full-slab refusal and a
  malformed final-layer import leave the previous state unchanged.
- Deferred pooling at 4/7/511/513 tokens is idempotent.
- Real 8,191-token prefill, native step to 8,192/full hot slab, export to MLX,
  first Q8 transition at 8,193 (8,192 cold tokens), re-import and native step to
  8,194. Peak footprint **38.3 GiB**, using bounded 512-row test chunks.
  This also exposed/fixed MLX rejecting a valid zero-length pending QSA tail.
- Split vs whole-token native runs match token, alternate token, top logit and
  complete final state; export/import continuation also matches exactly.
- A real serving fixture gives identical uninterrupted 64-token output and
  32-token + cached 32-token continuation (57 cached prompt tokens).
- Cancellation and an injected callback exception both permit a subsequent
  identical 64-token request. Serving peak footprint: **37.7 GiB**.
- Contemporary IFBench keys 20/70/100: MLX 3/3 strict and loose. Native initially
  scored 2/3 strict, 3/3 loose; after the EOS fix it scores 3/3 for both.
  Completion counts differ (MLX 223, corrected native 187), so these are not
  token-parity or comparable steady-state throughput measurements.

The native diagnostic's reported 0.7 GiB process footprint excludes much of
its read-only mmap-backed weight residency. System available RAM fell to
10.2 GiB. **It is not a 0.7 GiB model**, and it is not evidence of a RAM saving.
Shared-weight serving, rather than that standalone number, is the useful RAM
check. The actual Q8 state-boundary probe passes, but broad native quality,
long multi-turn HTTP stress and native MTP integration remain unqualified.

The first boundary probe used 2,048-row calls while both backend owners were
alive; the sampled guard stopped it at **40.113 GiB** (13.0 GiB still available).
It did not complete and is not a qualifying memory result. The guard was not
raised. The retained boundary fixture uses 512 rows to bound its graph; this
does not change the production automatic prefill policy or prove every native
workload stays below 40 GiB.

## 2. Compact target verifier: rejected prototype

The candidate used GPU stable route grouping, compact tile descriptors and
indirect gate/up/down dispatch. Unlike the older cross-row match scan, a weight
tile was explicitly decoded once and shared across up to eight route rows.
Compact INT8/U8 codebooks were reconstructed into small FP32 tables, without
materializing entire expert matrices. BF16 projection/activation boundaries
were retained, but FP32 matrix reduction reassociated the dot products.

S=5 layer-2 complete MoE cost increased from 0.831 to 2.643 ms, and output bits
changed. No full-model or quality claim follows. Small route groups, planning
overhead and the FP32 matrix execution are candidate explanations, not a
counter-based attribution. Do not repeat this implementation merely with more
drafts; first demonstrate a complete S=5 block gain under the same arithmetic.

## 3. GPU route planner + indirect prefill: exact but slower

Implemented block histograms, expert prefix sums, stable scatter, primary/tail
tile metadata and three GPU-written indirect grids. Tile counts never returned
to the CPU. Dispatch covered actual tiles, not a worst-case expert grid.
The original FP16 segmented math and slot-reduction order were retained.

The first planner scanned all routes per expert and was slower. A blocked
histogram version reduced that planning work but still lost in the complete
layer and full model. Full PP used the order warm-A/warm-B/A/B/B/A/A/B; all
eight outputs matched, including later controls. Peak footprint: **38.45 GiB**.

The prototype used a version-pinned MLX 0.32.2 custom primitive, a same-command-
buffer raw Metal encoder, GPU event ordering, indirect dispatch, and explicit
input/scratch lifetime retention. It did not patch an SDK or depend on private
object offsets. Those integration costs still matter. Eliminating a host
readback alone is insufficient evidence that the overall schedule improves.
No experimental planner switch, ABI bridge or slower verifier kernel remains.

## Protocol and reproduction

Model: `TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw`, revision
`64b0fb0f98a552d91fb9abd5531d547b2e78c8a8`. Packed-14 d8/K16384, d2/K256,
original FP16 prefill tables, signed INT8/affine U8 decode tables, Q8 backbone,
checkpoint SSD PLE, exact top-10. MTP off throughout this round.
Source baseline: `6af8216`; retained changes are in this report's commit.

Apple M5 Pro, 18 CPU cores, 64 GiB, macOS 26.5, MLX 0.32.2. One GPU job at a
time; full-model guard 40 GiB footprint / 38 GiB RSS / 6 GiB minimum available.
No active thermal controller or locked frequency. Post-test `pmset` reported
no recorded thermal/performance warning; background OS noise was uncontrolled.

Native pipeline: empty initial state, 64 repeated input IDs 9419 per arm; first
two steps excluded, first two arms warmups. This is not natural generation.
PP: `runtime_ab_probe`, 450 numbered notes, 7,224 tokens, token hash
`16262006632321229185`, cache off, 2,048-row automatic chunks, greedy/non-thinking,
max 64 output tokens (EOS at 27). IFBench: official scorer, keys 20/70/100,
greedy/non-thinking, max 512, concurrency 1; a development gate, not 300 tasks.

Retained probes, all requiring `devtools/memory_guard.py`:

```sh
qwen38-persistent-state-probe MODEL state
qwen38-persistent-state-probe MODEL pipeline
qwen38-persistent-state-probe MODEL parity
qwen38-persistent-state-probe MODEL boundary
qwen38-persistent-engine-probe MODEL
```

`parity` uses a repeated-token 127-token prefill and teacher-forced continuation;
passing it must not override real-prompt quality gates. The engine probe is target-only,
RAM-cache-only and tests continuation/cancellation/exception recovery.

Raw retained measurements and exact protocols:
[JSON results](results/vq-production-dataflow-2026-09-19.json).

## Final milestone checks

- Clean release build completed; existing duplicate-static-library linker
  warning only. CTest: **6/6 passed**.
- QSA rollback: **16 cases / 1,152 continuation steps**, exact output and QSA
  state, 0.9 GiB peak footprint.
- Updated engine lifecycle/EOS probe: cached continuation, cancellation and
  exception recovery all exact; one-token short response stops normally.
  Peak footprint 37.74 GiB, minimum available 9.75 GiB.
- State/pipeline and real Q8 boundary checks above passed under the guard.
- Official IF3 strict/loose: MLX 3/3, corrected native 3/3; no broad native
  HumanEval/IF30 rerun and no native-MTP speed claim.

Production VQ defaults and model files are unchanged. No new worktree was made.
Only the two slower prototypes created during this experiment were removed;
their evidence remains here. Existing unrelated evaluation-script edits were
not included in this milestone.
