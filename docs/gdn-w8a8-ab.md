# Layer-0 GDN W8A8 A/B

This developer benchmark injects the native MPP W8A8 projection only into
`GatedDeltaNet::forward_prefill`. The default hook is null, while decode and
verify never consult it. An explicit empty hook produced bit-exact outputs,
convolution state, and recurrent state versus the default path.

The test loads the real embedding, layer-0 attention hyper-connection, and
layer-0 GDN weights. Its two 512-token input cohorts use deterministic diverse
token IDs generated across the vocabulary; these are real embedding and HC
activations, but they are not a natural-language prompt. Each timing sample is
the sum of two consecutive M512 chunks and forces both outputs plus final
convolution and recurrent state to evaluate.

## Reproduce

```sh
cmake -S . -B build-all
cmake --build build-all --target qwen38-gdn-w8a8-ab --parallel 2
python3 devtools/memory_guard.py \
  --min-start-gib 16 --min-available-gib 12 \
  --max-rss-gib 4 --max-footprint-gib 4 --interval .1 -- \
  build-all/qwen38-gdn-w8a8-ab /path/to/model
```

The measured run used three alternating warmup pairs and 15 alternating
measurement pairs on an Apple M5 Pro with 64 GiB unified memory. Peak guarded
footprint was 1.5 GiB, peak RSS was 1.0 GiB, and two shards were opened.

| W8 banks | Sidecar | Mean Q4 / W8A8 | Mean speedup | Median Q4 / W8A8 | Median speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| z | 15.023 MiB | 8.734 / 8.459 ms | 1.033x | 8.711 / 8.403 ms | 1.037x |
| z + out | 30.033 MiB | 8.795 / 7.965 ms | 1.104x | 8.683 / 7.910 ms | 1.098x |
| qkv + z + out | 55.072 MiB | 8.824 / 7.233 ms | 1.220x | 8.729 / 7.138 ms | 1.223x |

For `z`, the second output had cosine 0.999984, maximum absolute error
0.015625, and RMSE 0.000566. For `z + out`, those values were 0.999242,
0.092773, and 0.003927. Both selective arms left convolution and recurrent
state bit-exact on these inputs. The full three-bank arm reached second-output
cosine 0.999160, maximum error 0.114258, and RMSE 0.004135; convolution cosine
was 0.999955 and recurrent cosine was 0.999887, so it needs a later quality
gate before any promotion.

Raw two-chunk samples in milliseconds:

```text
z q4:   9.026250 9.072000 8.605208 8.649750 8.742084 8.719333 8.760583 8.649375 8.688500 8.711000 8.733625 8.742958 8.616041 8.638417 8.659708
z w8a8: 8.402000 8.405333 8.357125 8.326416 8.329458 8.357042 8.645541 8.402958 8.435792 8.385125 8.477333 8.534500 8.349709 9.039750 8.433208
z+out q4:   8.598958 8.653833 8.682916 8.913625 8.654792 8.804000 9.392875 8.755583 8.744500 8.787250 8.671209 8.674875 9.251292 8.672833 8.664750
z+out w8a8: 7.860958 7.880417 7.910167 7.863125 8.512917 7.883541 7.928458 8.170333 7.926833 8.030500 7.861167 8.000083 7.873667 7.923541 7.846125
all3 q4:   8.729209 8.696375 9.322958 8.817041 8.763583 8.771708 8.712333 8.705584 8.866292 8.708000 8.596125 8.953917 8.583125 8.696417 9.435500
all3 w8a8: 7.093875 7.240833 7.233500 7.253417 7.106500 7.137917 7.862042 7.137833 7.132958 7.107333 7.101125 7.485083 7.078042 7.191042 7.337625
```

This is a layer-0 GDN microbenchmark, not a full decoder-layer or full-model
prefill result. Decoder MoE work is excluded, later-layer inputs would change
after any approximation, and these numbers do not imply a proportional gain at
65K context or establish the 600 tokens/s product target. No W8 arm is enabled
by default or exposed through a user-facing runtime profile.
