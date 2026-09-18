# DeepSeek V4.1 transfer probes

This note records which V4.1 ideas transfer to the retained
Qwen3.8-Flash-Next-VQ-2.1bpw checkpoint without retraining. None of these
experiments changes the automatic serving policy.

## Result summary

| Direction | Result | Decision |
|---|---|---|
| Confidence-scheduled speculation | A 0.25 raw-logit-margin tail cutoff kept all four 64-token output hashes and reduced proposals by 0/1/1/4 rows. Reported throughput moved 2.1--3.7%, but the unchanged JSON arm also moved 2.2%, exposing system noise. A 0.5 cutoff added a creative round and lost one accepted draft | Rejected as an automatic policy. DSpark uses a trained semi-autoregressive drafter and calibrated scheduler; raw MTP margin is not an equivalent confidence model |
| CSA2-like cross-layer reindex | One global 4x candidate pool recalled only 51.6--68.8% of later-layer selected QSA blocks. Re-anchoring every three attention layers improved some groups, but 4x still ranged 60.9--100%; 8x, already one quarter of the 16K block context, still ranged 65.6--100% | Rejected. This checkpoint's per-layer indexers are not interchangeable; restricting selection would change attention semantics |
| Packed affine Q4 KV | Direct selected-row Q4 attention was 14.6% faster than packed Q8 at the 64K probe and used 48 versus 68 MiB per probed layer with group-16 metadata. Its BF16-relative output error was 3.72% versus 0.455% for Q8 | Retain the guarded probe, not a serving mode. Post-training Q4 is too approximate for the long-context retrieval gate |
| SWA bounded replay | The runtime already keeps a bounded raw QSA window after completed four-token groups enter the pooled index. The retained 64-row implementation previously recovered 65K and 128K needles | Already represented by `qsa_raw_start` and the bounded raw window. GDN recurrence still must be persisted; replaying only 128 tokens cannot reconstruct it exactly |
| Engram-style lookup | The checkpoint already has PLE n-gram lookup backed by SSD. Prior decoded-row LRU and mmap/random-advice A/B tests were noise-equivalent end to end | Already present. Storage plumbing is not the remaining decode bottleneck |
| Causal encoder-decoder split | Qwen's second half consumes every preceding layer's hidden trajectory; no projection exists that turns a midpoint representation into the later layers' trained global KV | Architecture/training change, not a checkpoint-preserving runtime optimization |

## Packed KV probe

`qwen38-kv4-attention-probe` uses the real layer-3 projections to construct a
KV state, selects 2,048 rows, and runs BF16, packed Q8, and packed affine-Q4
attention directly. The Q4 Metal kernel was checked against materialized Q4
dequantization and matched exactly for the reported output.

Apple M5 Pro 64 GiB, MLX 0.32.2, speed runtime policy, 15 measured iterations
after three warmups, no active thermal control:

| Context | Format | KV bytes | Median attention | Cosine vs BF16 | Relative L2 |
|---:|---|---:|---:|---:|---:|
| 8,192 | BF16 | 16 MiB | 1.100 ms | 1.000000 | 0% |
| 8,192 | Q8/g64 | 8.5 MiB | 0.786 ms | 0.999984 | 0.572% |
| 8,192 | Q4/g64 | 4.5 MiB | 0.662 ms | 0.998066 | 6.224% |
| 8,192 | Q4/g16 | 6 MiB | 0.669 ms | 0.998846 | 4.822% |
| 65,536 | BF16 | 128 MiB | 1.175 ms | 1.000000 | 0% |
| 65,536 | Q8/g64 | 68 MiB | 0.784 ms | 0.999990 | 0.455% |
| 65,536 | Q4/g64 | 36 MiB | 0.662 ms | 0.998359 | 5.727% |
| 65,536 | Q4/g16 | 48 MiB | 0.669 ms | 0.999314 | 3.716% |

At 8K, quantizing only values to Q4/g32 produced 3.142% relative L2 error;
quantizing only keys produced 5.211%. A hybrid cache therefore does not reduce
the approximation enough to justify an end-to-end needle run yet.

The probe uses affine nibbles with BF16 scale/bias metadata. It does not claim
equivalence to DeepSeek's trained FP4 cache or its E2M1/E4M3 representation.

## Reproduction

```bash
cmake --build build-v1-all --target qwen38-kv4-attention-probe -j6
python3 devtools/memory_guard.py \
  --min-start-gib 42 --min-available-gib 8 \
  --max-rss-gib 40 --max-footprint-gib 42 -- \
  ./build-v1-all/qwen38-kv4-attention-probe "$MODEL_DIR" 8192
```

Use `65536` for the second context point. This is a developer probe, not a
server configuration.
