# VQ v2 checkpoint upgrade — 2026-09-19

Status: download and compatibility checks complete; full-model qualification
in progress. The first guarded start was refused at 41.1 GiB reclaimable versus
42 GiB required. The user then authorized a 40 GiB startup threshold, retaining
the 40 GiB footprint / 38 GiB RSS / 6 GiB availability runtime stops. Basic HTTP
generation succeeds; the three-case regression scores 2/3, so v2 is not yet
promoted. Its first warmed MTP fixture median is 47.66 tok/s, not a speed win.
The v1 daily model is retained until the v2 quality, performance and memory
checks pass. Do not apply the v1 README scores to v2.

## Model identity

Both versions are `TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw`:

| Version | Pinned revision |
|---|---|
| Retained v1 control | `64b0fb0f98a552d91fb9abd5531d547b2e78c8a8` |
| v2 candidate | `87d89bc328f8226deb95e26d9a69c7cd1f353007` |

The candidate is the author's mixed-codebook release, not our own re-quantization
or expert pruning. It retains 512 routed experts, top-10 selection, the Q8 dense
backbone and VQ PLE. The native Q6 MTP sidecar remains separate from the trunk.

| Projection family | v1 | v2 |
|---|---|---|
| Layers 0–1, all three projections | d2/K256, direct U8 | Unchanged |
| Layers 2–47, down | d8/K16384, packed-14 | d4/K256, packed-8 |
| Gate/up in layers 27–33, 35, 47 | d8/K16384, packed-14 | d4/K256, packed-8 |
| Other gate/up | d8/K16384, packed-14 | Unchanged geometry |

The author's card reports better approximation to its BF16 teacher at essentially
the same disk size. Those KL/top-1 agreement results are not IFBench/HumanEval
scores, and disk bytes are not a prediction of this engine's physical footprint.
[Pinned model card](https://huggingface.co/TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw/blob/87d89bc328f8226deb95e26d9a69c7cd1f353007/README.md).

## Engine changes

- Add d4/K256 packed-8 weight-tile reconstruction to segmented prefill, including
  fused gate/up and tail tiles. Permit gate/up and down to have different
  supported geometries; gate and up must still agree.
- Extend the existing ten-expert slot-packed down reduction to d4. Restrict the
  specialization to its actual 640-wide input; other widths use the generic
  reduction. The reduction order and dtype rounding remain unchanged.
- No routing, attention, KV, PLE or speculative acceptance policy changes.
- Fail closed if the old `QWEN38_PERSISTENT_VQ=1` experimental backend is
  requested for d4/mixed layers. Its legacy d2/d8-only kernels must not consume
  v2 data. The daily automatic MLX/Metal path supports the new geometry.
- v2 uses its original FP16 expert codebooks for both decode and wide prefill.
  The old d8 INT8/U8 preparation predicate requires all three projections to be
  d8, so it does not apply to v2's mixed layers. v1's numeric policy is unchanged.
  This difference must remain visible in v1/v2 comparisons.

## Verification

- Clean Release build, MLX 0.32.2.
- Eight CTest targets pass with both v1 and v2 tokenizer assets.
- Nine segmented CPU-oracle cases: d2, d4 and d8, row tiles 8/16/24, shuffled
  source rows and incomplete tails; check both projection and fused SwiGLU.
- Twenty-four slot-packed/scalar down comparisons: d4/d8, batch 1/2/5,
  FP16/BF16, with/without fused shared-expert addition. Exact output bits match.
- Manifest fixture accepts a mixed d8 gate/d4 down layer. Packed-8 index
  round-trip and supported-geometry predicates have CPU regression coverage.
- Real v2 layers 2 (mixed d8/d4) and 27 (d4) complete S=1, S=5 and 128-row
  smoke checks with finite outputs. Batch/serial maximum absolute differences
  are 0.000488/0.056641 at layer 2 and 0.004425/0.005493 at layer 27 for S=5/128;
  these are **not** bitwise whole-forward parity claims. Wide GEMM and serial
  routing/reduction use different floating-point execution paths.
- The explicit persistent-backend opt-in is rejected on real v2 metadata before
  backend allocation, with the expected unsupported-geometry diagnostic.
- HumanEval runner/grader unit tests: 13 cases, one optional-dependency skip;
  IFBench runner CLI tests: five passed. These test the tools, not model quality.

Full-model results and promotion decision will be recorded after guarded
evaluation. Platform: M5 Pro, 18 CPU
cores, 64 GiB, macOS 26.5, MLX 0.32.2; AC power, no active thermal conditioning.
Only one full-model process may run at a time. Use the user-approved 40 GiB
startup admission threshold and 40 GiB physical-footprint stop threshold;
thresholds are not measurements.

## Asset safety

The new revision is staged independently. Existing files reused across revisions
are checked against the pinned upstream SHA-256 before copying; no symlinks and
no deletion of the v1 checkpoint. Never reuse a derived Q4 draft LM head solely
because the model names match: verify its source tensors or regenerate it.

All 152 repository files are present with matching sizes; all 141 LFS files
pass pinned SHA-256 verification (47.913 GiB including MTP and auxiliary assets).
The locally derived Q4 draft head is additional, 357,581,270 bytes; its SHA-256 is
`e732a67b7eeb4d78dc47bebe2dec0851c6c6049275a7d7286c436f249833360d`.
It was regenerated from v2 `model-00019.safetensors`, source SHA-256
`7695d789f0d4f79eb09f58a1b88e2b32eb2d12575159a03949b31e025315a6e9`.
The conversion peaked at 2.25 GiB physical footprint, not a model-inference peak.

The SSD prefix-cache key incorporates model-directory identity, config/index
sizes and modification times. Keep revisions isolated during testing, and start
a fresh server after promotion; do not migrate live request state across them.
