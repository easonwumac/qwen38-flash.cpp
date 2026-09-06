# Fixed expert slots: synchronous fused-kernel feasibility probe

`qwen38-slot-pool-probe` is an isolated experiment, not a server option. It uses
16 fixed slots for one layer, nine page-aligned managed buffers for Q4/group64
weights and BF16 metadata. Original safetensor rows are read directly into a
selected slot after a GPU completion fence. Hot calls only pass slot indices to
the existing repository fused MoE kernels: no concatenation and no weight copy
on a hit. Existing kernel attribution remains in `src/moe_metal_kernels.hpp`.

This intentionally owns all buffers synchronously. It is NOT a thread-safe cache:
no graph/array escapes its owner, each invocation constructs a fresh graph, and
CPU writes only occur after GPU completion. A partial SSD read aborts the probe.
Production requires slot leases, atomic publication of successful loads,
generation validation, failure recovery and global memory accounting before this
can safely be connected to the asynchronous server.

## Checks

- Layers 0 and 47, 16 experts initially loaded, ten nonsequential selected slots.
- Replace selected slot 3 with expert 201, then restore expert 21.
- Compare the managed-buffer fused output with a freshly copied-buffer fused
  output after each change; all BF16 outputs must match exactly.
- Restoration must recover the original output; replacing an unselected slot
  must not change it.
- Report differences from the unfused MLX QMM path rather than hiding them. The
  existing fused arithmetic is not bit-identical to unfused QMM. This probe does
  not establish end-to-end language-model quality.

## Initial measurement (2026-09-06)

Apple M5 Pro /64 GiB; retained Qwen3.8-Flash-Next-REAP-288-MLX-4bit-repacked-MTP-Q8-REAP288-L47,
Q4/group64, BF16 metadata; MTP off. One synthetic 2560-element sinusoidal input,
ten fixed route weights, no router/shared expert/attention/PP included. Five
warmups and 50 timed fresh-graph calls per variant, including completion fence.
SSD fills are excluded from hot-call timing. Thermals/frequency are not locked;
this is not an end-to-end tok/s benchmark.

| Layer | Fused median / p95 ms | Unfused median / p95 ms |
| --- | --- | --- |
| 0 | 0.305 / 0.824 | 0.472 / 0.520 |
| 47 | 0.397 / 0.454 | 0.679 / 0.843 |

Initial median reduction is 35–41%, but layer 0 fused p95 is worse. Do not claim
a universal latency win. Managed-vs-copied fused parity passed all six cases.
Versus unfused QMM, max absolute differences were approximately 7.63e-5 (L0)
and 9.21e-5 (L47), RMSE approximately 1.35e-5 and 1.92e-5 for this small input.
Process peak was 0.2 GiB; MLX peak 84.5 MiB. No full model was loaded.

A clean-build rerun with reversed-order repeats confirmed replacement,
restoration and unselected-slot isolation. It also exposed substantial timing
variation: L0 fused/unfused medians were 0.611/0.485 ms first, then 0.293/0.464 ms
in reversed order; L47 was 0.593/0.823 ms, then 0.297/0.508 ms. Thus the initial
35–41% reduction is not a stable performance claim. The verified result is
zero-copy slot visibility and kernel compatibility; longer interleaved tests
and real hidden-state inputs are still needed. The unfused comparator includes
slot slicing and does not exactly reproduce the existing cache's object layout.

Run only with the private strict MLX library and `devtools/memory_guard.py`,
with 16 GiB starting headroom, 12 GiB available-memory floor and external 4 GiB
process cap. The probe additionally sets a 1 GiB MLX cap. Production unchanged.

## REAP ranking as a possible initial placement hint

The published [REAP-256 duo model](https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-256-duo-GGUF)
is a separate 512-to-256 pruning on code/agent calibration, not a verified nested
subset of our sh0wie REAP-288. Its author now warns that 256 prunes too deeply and
points to a 320 successor. Expert IDs must be mapped to the original model before
comparing lists; numerical IDs after pruning are not interchangeable.

If compatible rankings/maps become available, low-ranked experts can be an
initial SSD placement hint, not permanently excluded experts. Runtime demand
must be able to promote them. Keeping the existing router and loading every
selected expert preserves weights/routing, unlike pruning and renormalizing.
Merely moving 32 of 288 experts per layer to SSD saves roughly 4 GiB of expert
storage in this model, not enough by itself to halve the process footprint.
