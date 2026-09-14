# Niwaki 99B bring-up

Status: runtime-compatible candidate; not promoted over the retained REAP-288
checkpoint until the normal quality and needle suites pass.

## Runtime support

- Per-module affine Q3/Q4/Q8 geometry is read from the manifest.
- Missing routed banks become exact shared-only layers.
- Each BF16 `mlp.T` healing map is applied after the combined MoE output.
- Both `[C,1,K]` and `[C,K,1]` PLE convolution layouts are accepted.
- `--ngram-table-dir` reuses the row-major SSD Q4 table; the checkpoint's
  paired Q2 PLE shards are not added to the MLX graph.
- `--tokenizer-dir` reuses a compatible `vocab.json`, `merges.txt`, and
  tokenizer configuration when the checkpoint ships only `tokenizer.json`.

## Measurements

Environment: 2026-09-14; Apple M5 Pro MacBook Pro, 18 CPU cores, 64 GiB unified
memory; macOS 26.5 (25F71); AC power; no recorded thermal/performance warning;
MLX 0.32.2 through the pinned MLX-C ABI; one guarded process. Model:
`Qwen3.8-Flash-Next-99B-A5B-Niwaki-3bit-mlx`, Q4/group-32 backbone,
Q3/group-64 routed experts, Q8/group-64 routers, 512 experts/top-10, 24 routed
and 24 shared-only layers, BF16 healing maps. The external PLE was the retained
REAP-288 Q4/group-32 AoS table. Sampling was greedy/temperature 0, thinking off,
MTP off, and no prefix cache. Runs below are single directional samples, not
distributions.

| Probe | Configuration | Result |
|---|---|---:|
| Routed MoE layer 0, S=1 | generic grouped Q3 QMM | 1.149 ms warm median |
| Shared-only MoE layer 10, S=1 | shared Q4 path | 0.663 ms warm median |
| Short decode, 12 steps | grouped QMM, fused PP SwiGLU/reduce, O(n) inverse permutation, standard HC with predecoded dense injection | 34.81 steady tok/s; 20.9 GiB peak footprint, 20.8 GiB RSS |
| 8K prefill, 8,192 tokens | developer documentation corpus, chunk 1,024, prefill barrier 16, same PP candidates | 600.45 PP tok/s; next token unchanged; 25.7 GiB peak footprint |
| 128K capacity, 131,140 tokens | repeated developer corpus, chunk 512, Q8 KV at 65,536, flush 8,192, packed QSA from 32K, four-row selector, raw window 64, full 2,048-token decode budget | 379.75 PP tok/s; 80.41 ms / 12.44 decode tok/s; 1.60 GiB KV; 49.4 GiB sampled footprint, 20.6 GiB RSS |
| Synthetic 128K Q8 attention | layer 3, 512-token decode selector | 1.048 ms attention-only median; 2.073 ms complete-layer median |
| HTTP smoke | 57- and 64-token prompts, warm second request | health/ready/model list passed; 33.73 generation tok/s |

The 8K O(n) inverse-permutation kernel replaces a second `argsort` and preserves
the tested next token. Its unit oracle checks the entire inverse vector. The
dense-injection path removes repeated tiny quantized matmuls by predecoding about
8 MiB across the model; its one-layer displayed values and checksums matched the
standard path, but a longer greedy trajectory can still branch after accumulated
BF16 one-ULP differences. Keep it opt-in pending the full quality suite.

The first 128K run omitted the bounded raw-QSA window and reached only 3.09
tok/s; adding the exact 64-row raw-state bound raised it to 12.44 tok/s with the
same next token. A 512-token selector is previously needle-qualified on REAP and
is much faster in synthetic Niwaki attention probes, but it is approximate and
was not used for the reported 12.44 tok/s capacity result. Achieving 32--35
tok/s at 128K therefore remains an attention-backend milestone, not a model-swap
claim.
