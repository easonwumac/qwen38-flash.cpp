# Retained MTP benchmark

This benchmark reproduces the independent engine's retained 65+ tok/s path.
It is a high-predictability regression fixture, not a claim that every prompt
will reach the same throughput.

The historical run used an M5 Pro with 64 GiB unified memory, a REAP-288 Q4
target plus the Q8/g64 L47 MTP sidecar, temperature zero, thinking disabled,
the P271 depth-2 probe with a depth-3 cap, and no prefix cache. After warming
model pages and kernels,
the 128-token path accepted 89/103 proposals and measured 64.87--65.03 tok/s.
The 256-token path accepted 185/205 and measured 66.97 tok/s. Peak sampled RSS
was about 35.5--37.8 GiB.

Start the server through the memory guard. Use a model directory containing the
Q4 target and Q8 L47 sidecar:

```bash
QWEN38_HISTORY_DRAFT=0 ./devtools/memory_guard.py \
  --min-start-gib 42 --min-available-gib 6 --max-rss-gib 40 -- \
  ./build-all/qwen38-server \
  --host 127.0.0.1 --port 11438 --model /path/to/q4-target-q8-l47 \
  --profile speed --prefix-cache-tokens 0 --mtp-depth auto
```

In another terminal, collect one warmup and two measured samples per length:

```bash
python3 devtools/mtp_benchmark.py \
  --url http://127.0.0.1:11438 --tokens 128,256 \
  --warmups 1 --samples 2 --require-mtp --require-promotion \
  --min-acceptance 0.85 \
  --min-median-tps 128:64.0,256:65.0
```

The client exits zero when every gate passes, one for a measured regression,
and two for a request or telemetry error. JSON output includes every sample,
median/min/max throughput, acceptance, and the distinct
`rounds:accepted/proposed:depth` paths. The server's prefix cache is disabled;
"warm" here means resident weights/pages and compiled kernels, not prompt reuse.
The command uses regression floors rather than rounding the historical peaks
into hard requirements. The historical 64.87--65.03 and 66.97 tok/s values
remain the reference; passing 64.0/65.0 only proves that the retained path has
not materially regressed under ordinary run-to-run noise.

Thermal state and competing GPU or memory traffic can move throughput. A failed
speed gate is therefore a signal to repeat under controlled conditions, while
a changed proposal/acceptance path is a stronger numerical or policy regression
signal. The broader production decision still belongs to a mixed code,
creative, JSON, and long-context corpus.

## Current reproduction

On 2026-09-02 the same 64 GiB M5 Pro, Q4-target/Q8-L47 model directory,
depth-3 cap, disabled prefix/history cache, and memory guard reproduced the
exact historical numerical paths. Two measured warm samples after one warmup gave:

| Length | Acceptance | Warm tok/s | Median |
|---:|---:|---:|---:|
| 128 | 89/103 (86.4%) | 64.389, 64.458 | 64.424 |
| 256 | 185/205 (90.2%) | 66.537, 66.508 | **66.523** |

Both requests completed the requested token count with no fallback. The run was
guarded at 40 GiB process RSS and 6 GiB minimum available memory. Thermal state
was not externally controlled, so this validates the 65+ 256-token path and
exact acceptance trajectory, not the old 66.97 peak as a guaranteed floor.

A same-binary `--mtp-depth auto` retest produced the same promotion and token
paths, with medians of 64.367 and 66.429 tok/s. This confirms that `3` is a hard
cap over the P271 probe policy rather than an instruction to start every round
at depth 3. The benchmark now records promotion/demotion counts and can require
promotion explicitly, preventing a depth-2-only run from passing solely on
aggregate acceptance.

The report also aggregates four-element `proposed_by_position`,
`accepted_by_position`, and `acceptance_by_position` vectors. These distinguish
an inexpensive first/second-position drafter from a genuinely profitable third
or fourth position. Run with `QWEN38_HISTORY_DRAFT=0` when using the vectors to
tune learned-MTP depth; otherwise they intentionally include history rounds too.

## Experimental depth 4

The current optimized runtime makes a fourth draft position profitable on the
high-predictability fixture, although not yet broadly enough to change `auto`.
With fixed depth 4, two guarded warm samples reached 65.770/65.614 tok/s at 128
tokens (65.692 median) and 66.684/66.395 at 256 tokens (66.539 median). The
paths were exactly 97/124 in 31 rounds and 197/240 in 60 rounds. Fourth-position
acceptance was 61.3% and 66.7%, respectively. Compared with the retained auto
depth-3 medians, this is about +2.0% at 128 tokens and effectively flat at 256.

These results use the same Q4 target/Q8-L47 sidecar, M5 Pro 64 GiB, temperature
zero, disabled history/prefix caches, and 40 GiB RSS/10 GiB availability guard.
Peak physical footprint was 40.7 GiB. Fixed depth 4 remains an experiment:
general prompts with lower fourth-position acceptance can lose throughput, so
the next promotion gate is a mixed-domain adaptive-depth cohort.

The benchmark client also reports cumulative draft, target-verify, and commit
phase time. A later guarded 128-token Q8/adaptive-S4 repeat kept the exact
31-round, 97/124 path and measured a 65.45 tok/s median: target verification
used 1,644.6 ms (84.1% of generation), drafting 272.2 ms (13.9%), and accepted
state commit 30.7 ms (1.6%). Reaching 75 tok/s on this path requires removing
about 249 ms, so commit-only or barrier-only work cannot meet the target;
drafter bandwidth and verifier-wide execution remain the material levers.

An opt-in adaptive policy now starts short prompts at depth 4 and checks the
fourth position directly for eight rounds. On the retained fixture it kept
depth 4 and reached 65.852/65.684 tok/s at 128 tokens (65.768 median) and
66.893/66.804 at 256 (66.848 median), or about +2.1%/+0.5% versus retained auto.
A one-sample 128-token mixed probe measured code 54.09 versus 43.23 tok/s
(+25.1%), JSON 53.50 versus 54.22 (-1.3%), explanation 39.32 versus 39.90
(-1.4%), and creative 38.64 versus 39.83 (-3.0%). Code, JSON, and creative
output hashes matched their controls; explanation followed a different valid
greedy path. This is evidence for an experimental switch, not a default-policy
promotion or a broad quality claim.

The policy now adds a one-time four-round probation before the eight-round
window. Per-round traces showed 4/4 successful fourth positions on the retained
fixture, 3/4 on code, 4/4 on JSON, 1/4 on explanation, and 0/4 on the short
creative completion. The `<2/4` early exit therefore preserved the exact
97/124 and 197/240 fixture paths while cutting explanation's S=4 proposals from
eight to four. A repeated thermally continuous fixture measured 65.222/65.902
tok/s medians; these lower timings are not treated as a regression because the
proposal paths were identical and the machine was not thermally reset.
