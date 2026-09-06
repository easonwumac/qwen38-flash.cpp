# Q8 KV isolated attention probe

## Decision

Keep production BF16 KV and full Q8 MTP unchanged. Q8 KV is a possible storage
optimization, not yet a demonstrated runtime-memory or throughput improvement.
The next useful implementation would dequantize only QSA-selected blocks, rather
than reconstructing the complete BF16 KV on every verification forward pass.
It still requires full-model retrieval/output and MTP acceptance validation.

## Reproduction and scope

Build target `qwen38-kv8-layer-probe`; invoke under `devtools/memory_guard.py`
with `MODEL [8192|65536]`. Require startup16 GiB, available12 GiB,
RSS/footprint4 GiB; use the private strict MLX runtime. The executable caps MLX
allocation at3 GiB and allocator cache at64 MiB. Fresh Release build and all
five CTests passed. No running inference service or model was modified.

Hardware: M5 Pro64 GiB. Model: Qwen3.8-Flash-Next-REAP-288-MLX-4bit with
full Q8 REAP288 L47 drafter artifact (drafter is not exercised by this probe).
Main weights are unchanged. Actual layer3 attention weights, speed profile,
QSA enabled. Prefill uses512-token chunks. Inputs are deterministic token
embeddings fed directly into the isolated attention layer, NOT actual upstream
layer3 hidden states or natural-language full-model inputs. No sampling or MTP.

Each candidate packs K/V affine8-bit, group64, keeping QSA auxiliary state
unchanged. It then dequantizes the entire K/V to BF16 before the existing
16-row attention forward. Five pairs, control then candidate, in one process;
no thermal lock or randomized order. Reported times include candidate
dequantization, exclude initial packing. They are NOT PP/decode tok/s.

| Context | BF16 KV | Q8 including metadata | Packing | Control median | Q8 median | Relative L2 output error |
|---|---:|---:|---:|---:|---:|---:|
|8192|16 MiB|8.5 MiB|3.88017 ms|1.48429 ms|1.61204 ms|0.00541991|
|65536|128 MiB|68 MiB|4.92308 ms|10.9769 ms|12.1433 ms|0.00524772|

Logical KV storage falls46.875%. Median isolated-forward cost rises8.6%/10.6%
in these tiny samples; noisy timings do not establish a general slowdown.
Maximum absolute output difference is0.00012207 for both lengths. This is
not a task-quality score or a guarantee of unchanged generated tokens.

Original BF16, packed KV, and reconstructed BF16 coexist in the experiment.
Therefore this does NOT demonstrate physical-memory savings. Guard-reported
peak footprint was0.6/2.2 GiB, RSS0.4/0.4 GiB, minimum available45.3/43.4 GiB.

Raw times in ms, ordered by repeat0..4:

```
8192 control: 2.53242 1.74392 1.48429 1.32454 1.47283
8192 q8:      2.50708 1.70608 1.43504 1.57296 1.61204
65536 control: 11.9268 10.9769 10.9782 10.9 10.7555
65536 q8:      12.7542 12.1433 10.5485 10.6528 13.1151
```

## SSD checkpoint audit

The retained8K reuse experiment has an8220-token original-prompt snapshot
(312115882-byte safetensors) and8236-token extended snapshot (312608425 bytes).
These are different states, not identical duplicates. `load_longest` rejects
token manifests longer than the request, and recurrent GDN state cannot simply
be sliced backward. Keeping only the longer snapshot loses the shorter retry
hit; keeping only the shorter snapshot requires recomputing the continuation.
No snapshot was deleted or retention policy changed. Their overlapping KV
could potentially share an on-disk representation, but that is a separate
format/atomicity/eviction change, not a free RAM reduction.
