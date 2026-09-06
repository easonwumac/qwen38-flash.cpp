# Mixed drafter acceptance gate and SSD idle-cache check

Decision: do not switch to the hybrid drafter on the evidence of this suite.
Preserve the full-Q8 option; SSD idle-cache reuse passed this bounded test.
No running service/model configuration was changed.

Same REAP-288 affine Q4/group64 target and lossless16 metadata, M5 Pro 64 GiB,
speed profile, resident 12:29, depth 4 with existing profitability fallback,
thinking off, greedy generation, max 64 output tokens. Four prompts from the
existing mixed-domain probe, two identical requests per case. Process order:
full Q8 without prefix cache, hybrid without prefix cache, full Q8 with SSD.
One process per arm; temperatures uncontrolled and no reverse-order replication.

| Case | Full-Q8 accepted/proposed | Hybrid accepted/proposed | Full-Q8 warm tok/s | Hybrid warm tok/s |
| --- | --- | --- | ---: | ---: |
| Code | 47/60 (78.3%) | 45/68 (66.2%) | 66.0 | 58.5 |
| JSON | 51/56 (91.1%) | 51/56 (91.1%) | 74.6 | 68.9 |
| Explain | 36/108 (33.3%) | 37/104 (35.6%) | 37.7 | 37.0 |
| Creative | 3/16 (18.8%) | 6/28 (21.4%) | 40.0 | 38.4 |

Acceptance counts repeated exactly within each arm. The code drop is 12.2
percentage points, adding two verification rounds. Creative falls back once
in both arms, after four versus seven rounds; accepted/proposed does not count
subsequent non-MTP output. Thus it is not an overall acceleration metric.
The code/JSON/explain output tokens agree across drafters; creative differs.
An unchanged target does not imply bit-identical trajectories across verifier
batch groupings, nor does this small suite establish broad quality equivalence.
These warm rates are single second-request samples, not robust speed medians.

Guard sampled footprints: full Q8 38.3 GiB, hybrid 37.7 GiB, full Q8+SSD 38.4 GiB.
This supports the known approximate 0.6–0.7 GiB drafter saving, not a new kernel
improvement. It is not enough to justify the observed code acceptance regression
under the user's preference. Historical longer-fixture wins remain workload-specific.

## SSD reuse

Full-Q8 SSD second requests report cached prefix lengths 25/29/25/25 tokens.
All eight generated token arrays AND accepted/proposed counts match the no-cache
full-Q8 arm. Second-request generation rates were 65.8/73.1/37.5/39.2 tok/s.
The persistent RAM checkpoint is dropped on successful SSD save by current
NativeEngine, so the subsequent reuse exercises its load_longest path. No
claim of uncached disk performance: macOS can retain file pages, and MLX loads
may be lazy. Logged prompt_ms is not an independently instrumented TTFT.
Short prompts do not demonstrate long-context memory savings. Peak is slightly
higher with SSD; idle savings versus RAM-only cache remain unmeasured.

An experiment-only 1 GiB SSD cache was created under the project experiments
directory (ssd-mtp-check.rBIExV), retaining about 406 MiB for inspection. No
existing user cache was cleared. It is not a serving cache configuration.

Fresh Release build, all five CTests passed. Full-model arms completed under
startup 40 GiB / available floor 8 GiB / process caps 42 GiB, private strict
allocator ceiling 40 GiB, cache 256 MiB. Minimum available was 8.8/8.7/8.5 GiB.
No long-context escalation. NativeEngine probe and structured results retained
for repetition; these results do not establish that 8 GiB is OOM-proof.
