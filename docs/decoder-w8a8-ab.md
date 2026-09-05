# Layer-0 decoder W8A8 A/B

This developer benchmark measures a complete layer-0 decoder: both
hyper-connections, GDN, and sparse MoE. It forwards the default-null,
prefill-only GDN projection hook through `DecoderLayer`; decode, verify, and
user-facing profiles are unchanged. An explicit empty hook is required to be
bit-exact with the null-hook control before measurement begins.

Two diverse deterministic 512-token ID cohorts generate real embedding
streams. They are not a natural-language prompt. Every sample is the sum of
two consecutive M512 layer calls and forces the output stream, GDN convolution
state, and GDN recurrent state to evaluate.

## Reproduce

```sh
cmake -S . -B build-all
cmake --build build-all --target qwen38-decoder-w8a8-ab --parallel 2
python3 devtools/memory_guard.py \
  --min-start-gib 16 --min-available-gib 12 \
  --max-rss-gib 4 --max-footprint-gib 4 --interval .1 -- \
  build-all/qwen38-decoder-w8a8-ab /path/to/model
```

The harness explicitly enables GDN Metal prefill and grouped MoE prefill. It
uses lossless13 compact qmeta with request caching/deferred reduction only when
the required layer-0 sidecars exist. The measured base checkpoint had no qmeta
sidecars, so this run is labeled **exact-Q4 base, top-10, full metadata**. It is
not the retained product profile, nor the top-8/lossy9 turbo experiment.

## M5 Pro result

Three alternating warmup pairs and 15 alternating measured pairs ran on an
Apple M5 Pro with 64 GiB unified memory. Four shards were opened. Guarded peak
footprint was 2.6 GiB, peak RSS was 1.8 GiB, and minimum available memory was
29.5 GiB.

| W8 GDN banks | Sidecar | Mean Q4 / W8A8 | Mean speedup | Median Q4 / W8A8 |
| --- | ---: | ---: | ---: | ---: |
| z + out | 30.033 MiB | 28.462 / 27.826 ms | 1.023x | 28.417 / 27.732 ms |
| qkv + z + out | 55.072 MiB | 28.576 / 26.861 ms | 1.064x | 28.410 / 26.847 ms |

For `z + out`, the second complete-layer output had cosine 0.998994, maximum
absolute error 0.102539, and RMSE 0.001095. Convolution and recurrent states
were bit-exact. For all three banks, the second output had cosine 0.998962,
maximum error 0.122070, and RMSE 0.001113; convolution cosine was 0.999955 and
recurrent cosine was 0.999887. The full three-bank arm therefore still needs a
quality gate.

The complete-layer result shows that MoE and hyper-connection work dilute the
isolated GDN projection gain. It does not measure a full decoder stack, natural
prompt, retained compact-qmeta profile, 65K context, or full-model throughput,
and it must not be extrapolated to the 600 tokens/s product target. No W8 path
is enabled by default.
