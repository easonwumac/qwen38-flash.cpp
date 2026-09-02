# Selective Q4/Q8 MTP sidecar ablation

This experiment searches for a lower-bandwidth drafter without changing the
Q4 REAP-288 target. The source sidecars are the local Q4/g64 and Q8/g64 L47
packs; `devtools/build_selective_mtp_pack.py` copies selected projection groups
from Q8 into the Q4 sidecar without dequantizing or requantizing them.

All runtime samples below used an M5 Pro with 64 GiB unified memory,
temperature zero, thinking/history/prefix cache disabled, adaptive depth 4,
and the 40 GiB RSS / 42 GiB footprint / 10 GiB available-memory guard. Unless
noted, the verifier barrier stride and resident expert range were the speed
profile defaults (16 and `12:28`). Single-sample ablations identify proposal
paths; promoted comparisons use repeated warm samples.

| MTP precision | 128-token path | tok/s | draft ms | Result |
|---|---:|---:|---:|---|
| Full Q8 control | 31 rounds, 97/124 | 65.45 median | 272.2 | Retained control |
| Full Q4 control | 33 rounds, 94/131 | 62.41 median | not captured | Slower |
| Routed experts Q8 only | 34 rounds, 92/126 | 61.31 median | 255.3 | Rejected |
| FC Q8 only | 35 rounds, 91/129 | 60.08 | 251.4 | Rejected |
| Attention Q8 only | 32 rounds, 96/128 | 63.87 | 264.7 | Rejected alone |
| Hyper-Connection Q8 only | 33 rounds, 95/132 | 62.43 | 260.8 | Rejected |
| Shared expert Q8 only | 34 rounds, 92/126 | 61.37 | 246.6 | Rejected |
| All non-routed groups Q8 | 32 rounds, 96/128 | 63.83 median | 266.7 | Rejected |
| Attention + HC Q8 | 32 rounds, 98/128 | 63.77 | 266.3 | Rejected |
| Attention + shared Q8 | 30 rounds, 98/120 | 67.31 | 253.0 | Promising path, one sample |
| **Attention + FC Q8** | **30 rounds, 98/120** | **67.66 median** | **253.2** | Retained candidate |
| Attention + FC, barrier 48 | 30 rounds, 98/120 | 68.10 median | 248.4 | Better diagnostic setting |
| Attention + FC, barrier 48, resident `12:29` | 30 rounds, 98/120 | **68.19 median** | 248.5 | Best repeated result |
| Attention + FC + lossless16 qmeta, barrier 48, resident `12:29` | 30 rounds, 98/120 | **68.60 pooled median** | 249.9 | New bit-exact research candidate; lower footprint |
| Attention + FC, barrier 48, resident `12:30` | 30 rounds, 98/120 | 67.83 median | 252.6 | Rejected |
| Attention + FC + routed gate Q8 | 30 rounds, 98/120 | 64.92 | 346.4 | Rejected: mixed-bit generic MoE |

The retained candidate sidecar is 879,596,180 bytes with SHA-256
`7e3f33cfbaaa485920fcd5b59523252032e963549f2ff9788031294f7e8a5601`.
It was built with:

```bash
python devtools/build_selective_mtp_pack.py Q4_SIDECAR Q8_SIDECAR OUTPUT \
  --q8-groups attention fc
```

The 256-token retained fixture stayed at 60 rounds but accepted 195/240 rather
than the full-Q8 control's 197/240; two warm samples measured 66.43 and 66.47
tok/s. Thus the 128-token improvement is real but trajectory-dependent, not a
stable 75 tok/s result.

On the 128-token mixed-domain probe, attention+FC measured 56.39 tok/s for
code, 59.40 JSON, 40.53 creative, and 39.65 explanation. Compared with the
most recent full-Q8 cohort, code/JSON improved materially and explanation was
roughly flat-to-lower. Code and JSON hashes matched full Q8; explanation and
creative changed valid target-authoritative greedy trajectories. A broader
quality corpus is required before making this the production default.

The target's existing lossless16 qmeta sidecar is compatible with the hybrid
drafter and does not change decoded BF16 scale/bias values. In an adjacent
A/B/A comparison, four lossless16 samples measured 68.231--68.677 tok/s with a
68.599 pooled median, versus 68.236/68.244 tok/s for full metadata. The MTP
path remained exactly 30 rounds and 98/120; peak physical footprint fell from
about 40.1 to 38.1 GiB. Lossless13 was slower at a 67.215 median. This is a
small (~0.5%) verifier-bandwidth improvement, not the missing step to 75 tok/s.
The lossless16 kernel now reads its aligned tags directly as `ushort` values;
`QWEN38_QMETA_ALIGNED16=0` retains the generic bit-window rollback. The layer-0
S=5 probe improved from 2.710 to 2.355 ms with an identical bit hash. In the
same-binary HTTP comparison, verifier time improved from a 1,576.1 ms median
to 1,569.9 ms; total 128-token throughput was noise-equivalent at 68.87 versus
68.84 tok/s because drafter time moved in the opposite direction. At 256
tokens verifier time improved from the pre-change 3,210.7 ms median to 3,198.7
ms, while total throughput likewise remained noise-equivalent. The aligned
reader is retained as an exact verifier component improvement, not claimed as
an additional end-to-end speedup.

Do not selectively promote only one of routed gate/up/down with the current
runtime. Fused MTP MoE requires all three projections to have the same bit
width; a gate-only Q8 probe kept the useful proposal path but raised draft time
by 39% through the generic fallback. The builder now warns when asked to create
such a pack. A mixed-bit fused kernel would be prerequisite to revisiting this
axis.

The cost breakdown explains the remaining gap. The lossless16 128-token candidate
used about 1,577 ms in target verification, 250 ms in drafting, and 30 ms in
commit. Reaching 75 tok/s requires total generation below 1,707 ms, so another
roughly 150--160 ms must come primarily from verifier-wide execution or from a
general acceptance improvement that removes about three target rounds.
