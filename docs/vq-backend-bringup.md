# VQ-2.1bpw backend bring-up

Model: `TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw`, revision
`64b0fb0f98a552d91fb9abd5531d547b2e78c8a8`.

The native loader supports the checkpoint's self-contained `tokenizer.json`,
flat `model.*` tensor namespace, 144 routed VQ projections, and 128 mmap-backed
VQ PLE shards. The correctness path supports both shipped expert geometries:
d2/K256 byte codes in layers 0-1 and d8/K16384 14-bit packed codes elsewhere.
Selected experts are evaluated directly from their codebooks with FP32
accumulation; gate/up and down/route reduction each use one Metal dispatch.

Checkpoint parity smoke (raw token 9419, greedy, two decode steps) produces
tokens `0, 5606` in both the C++ backend and a compatible MLX reference after
the reference is made to honor the checkpoint's stored PLE multipliers.
The checkpoint multipliers correspond to n-gram seed 1234. The bundled
pre-fold MLX reference currently rebuilds them with its local default seed 0
and otherwise produces `11, 271`; do not use that path as a PLE oracle.

Bring-up measurement, not a release benchmark: Apple M5 Pro (18 CPU cores),
64 GB unified memory, macOS 26.5; single request; one-token raw prompt;
greedy sampling; MTP off; short context; no thermal conditioning; one run.
The correctness build measured a 36.2 GiB peak physical footprint and about
1.03 decode tok/s. The low decode speed is not a target result: decode still
synchronizes routing on every layer.

The VQ prefill path now keeps routing on device, evaluates all rows in a fused
gate/up kernel, and combines down projection with route reduction. The mmap
PLE path also resolves and requests the needed pages before decoding rows.
On the same machine, a 13-token non-thinking chat prompt improved from 812.63
ms serial to 273.98 ms batched (2.97x), with identical next token 9419 and a
36.3 GiB peak physical footprint. A separate 463-token sample from this file,
64-token chunks, measured 7.34 s (63.0 PP tok/s) and 38.9 GiB. Both are single
warm-cache observations, greedy, MTP off, single request, no thermal
conditioning; they establish bring-up and parity, not a release distribution
or the final performance target.
