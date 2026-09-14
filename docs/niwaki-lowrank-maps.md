# Niwaki low-rank healing maps

Status: retained candidate; 128K retrieval and directional quality checks pass,
but the normal broad quality suite and three cold-run distribution remain release
gates.

## Method

Niwaki's 48 BF16 `mlp.T` matrices are exactly identity on the diagonal and can
be written as `T = I + delta`. On layers 0, 12, 24, 36, and 47, `delta` was only
2.59--3.74% of the full-map Frobenius norm. The offline builder computes a
deterministic randomized SVD of each delta and stores BF16 left/right factors.
Rank 64 occupies 31,468,768 bytes for all 96 factors. At runtime the correction
is evaluated as:

```
output + (output @ delta_left) @ delta_right
```

When the sidecar tensors are indexed, the full 48 dense maps are not loaded.
`QWEN38_NIWAKI_FULL_MAPS=1` is the reference rollback, and
`QWEN38_SKIP_NIWAKI_MAPS=1` is diagnostic-only. Both settings participate in
the prefix-cache compatibility key.

Build the sidecar in place with:

```bash
python3 devtools/build_niwaki_lowrank_maps.py MODEL_DIRECTORY --rank 64
```

## Directional measurements

Environment: 2026-09-14; Apple M5 Pro MacBook Pro, 18 CPU cores, 64 GiB unified
memory; macOS 26.5; MLX 0.32.2; AC power; one guarded process; no active thermal
control. Model: Niwaki 99B, Q3/group-64 routed experts, Q4 backbone, top-10,
external REAP-288 Q4 SSD n-gram table. Sampling was greedy, temperature zero,
thinking off, MTP and prefix cache off.

| Workload | Result |
|---|---:|
| Fixed-input short decode, full BF16 maps | 33.95 tok/s |
| Fixed-input short decode, rank 128 | 40.48 tok/s |
| Natural 16-step decode, full maps | 37.02 tok/s |
| Natural 16-step decode, rank 64 | 41.16 tok/s; all 16 tokens matched the full-map trajectory |
| 16K needle, rank 64, Q8 KV, QSA budget 256 | `K7-MOON-491` recovered; 946.24 PP tok/s; 36.36 decode tok/s |
| 65K needle, same path | `R2-COMET-853` recovered; 757.55 PP tok/s; 31.07 decode tok/s |
| 128K needle, same path | `V1-NEBULA-128` recovered; 671.82 PP tok/s; 29.38 decode tok/s; 23.2 GiB guarded peak footprint |

The long runs used Q8 KV from 8,192 tokens, 2,048-token Q8 slabs, packed QSA
from 8,192, four-row prefill selection sharing, a 64-token raw QSA window, and
an adaptive 1,024-row configured prefill chunk. Decode selected 256 tokens. The
128K request contained 131,140 prompt tokens and returned 17 tokens.

A ten-case 256-token-output math/logic/Chinese/extraction/code subset scored
4/10 with both rank 64 and full maps; the passed cases differed. This establishes
no aggregate regression in that small sample, not model-quality equivalence.
Niwaki frequently spends the former 64-token evaluation allowance before
producing final content, so the quality tool now accepts a configurable output
limit and preserves performance/reasoning fields on grading failures.

## Rejected branches

- Purpose-built scalar and simd-shuffle Q3 Metal MoE kernels reached 29.01 and
  27.03 tok/s versus a 33.95 generic gather-QMM control. Q3 bit extraction and
  reduced occupancy outweighed removed graph nodes; the prototypes were removed.
- Routed top-8 retained the 16-token trajectory but fell to 38.98 tok/s versus
  41.16 at top-10. The generic Q3 kernel benefits from ten-slot occupancy.
- Rank 32 reached 40.94 tok/s and did not beat rank 64 materially, so rank 64
  retains more healing energy for the same practical dispatch floor.
- QSA budget 128 recovered the 65K needle but fell to 28.19 tok/s versus 31.07
  at budget 256. Smaller packed work reduces GPU occupancy.
- Alternating global QSA with recent-256 BF16 attention recovered the 16K needle
  but reached only 32.88 tok/s versus the 36.36 all-global control. MLX SDPA
  dispatch/layout cost exceeded packed-Q8 selection and attention; removed.

The remaining 128K gap is about 9.0 ms/token. It must come from a materially
different fused selector/direct-Q8 execution graph and further trunk dispatch
reduction; reducing candidate count or deleting global layers is not supported
by these end-to-end results.
