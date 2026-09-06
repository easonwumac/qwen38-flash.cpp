# First MTP allocation audit and retained hybrid recheck

Research instrumentation is compiled only into qwen38-mtp-round-smoke via its
own mtp_runner.cpp object. The normal qwen38_mlx library contains no phase
logging code/strings or runtime flag checks. Production kernels and defaults
are unchanged. Samples have no additional GPU fences and are not exact phase
physical peaks. Fresh Release build and all five CTests passed.

The recent admission tests used the full Q8 MTP capsule, not the previously
retained attention+FC-Q8 / remaining-Q4 hybrid. This distinction matters:

| Quantity | Full Q8 drafter | Retained hybrid |
| --- | ---: | ---: |
| Drafter file bytes | 1,600,492,544 | 879,596,180 |
| First draft active-memory increase | 1,601,462,280 | 880,566,280 |
| Guard sampled peak physical footprint | 38.0 GiB | 37.3 GiB |
| Minimum available | 10.2 GiB | 10.6 GiB |
| Four-token serial parity | pass | pass |

The near match of file size and first-draft allocation is strong evidence of
weight materialization, not a 1.5 GiB disposable verifier buffer. It is not an
allocation-by-allocation proof. Target verification adds about 64 MiB active
above restored-head state in the first round, most released at commit. Later
rounds do not show comparable persistent growth.

Both produced [271,40,599,264], three rounds, two accepted drafts. The hybrid
SHA-256 is 7e3f33cfbaaa485920fcd5b59523252032e963549f2ff9788031294f7e8a5601,
matching docs/mtp-hybrid-ablation.md. This recovers an existing approximately
0.67 GiB saving; it is not a new algorithm or a new improvement over that
historical best configuration. The target quantization is unchanged; the
drafter precision differs and broad acceptance/quality still require validation.

M5 Pro 64 GiB, REAP-288 affine Q4/group64 target, lossless16 metadata, resident
12:29, speed profile, depth 2, seed token 9419, no prefix cache. One full-Q8 run
then one hybrid run, no thermal control; cold short smoke with logging/cache
clears. Their printed token rates must not be interpreted as general speed or
compared to the historical warm 128-token benchmark. No long-context PP claim.

Safety: private strict runtime, allocator cap 40 GiB/cache 256 MiB; guard startup
40 GiB, available floor 8 GiB, process RSS/footprint ceilings 42 GiB, interval
0.1 seconds. Both runs exited normally. Existing service and model files were
not modified. Use the retained hybrid as an explicitly labeled research
baseline for subsequent no-regression checks, not full-Q8 admission numbers.
