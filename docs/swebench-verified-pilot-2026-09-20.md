# SWE-bench Verified three-task pilot, 2026-09-20

Status: complete directional pilot. The retained VQ v1 checkpoint resolved
2/3 tasks; the VQ v2 candidate resolved 1/3. This is not a full 500-task
SWE-bench Verified score and must not be compared with leaderboard percentages.

## Decision

Keep VQ v1 as the daily checkpoint. V2 was faster on the SymPy task, but used
all 40 agent steps without submitting a patch on both Django and scikit-learn.
Combined with its previously measured IFBench regression, this small agentic
pilot does not support replacing v1.

The pilot also exposed a real runtime defect: a restored or extended multi-turn
QSA path could pass FP32 activations to the packed attention kernel while still
selecting `TILE=32`. At head dimension 256, its two threadgroup arrays required
64 KiB and exceeded the Apple GPU's 32 KiB limit. FP32 packed QSA now defaults
to `TILE=16`; FP16 retains `TILE=32`. Both checkpoints then completed every
agent trajectory without a Metal or server error.

## Fixed protocol

The three instances were frozen before model answers and cover three distinct
repositories:

1. `sympy__sympy-15345`
2. `django__django-11451`
3. `scikit-learn__scikit-learn-9288`

This is a development subset of the public Verified test split, not a random
estimate with useful confidence bounds. Generation used the stock
mini-SWE-agent SWE-bench scaffold plus only the resource and model settings
below. Official SWE-bench evaluation determined whether submitted patches
resolved the tasks.

| Setting | Value |
|---|---|
| Hardware | Apple M5 Pro, 64 GiB unified memory |
| Host | macOS 27.0 (26A428), AC power, no active thermal control |
| Agent / evaluator | mini-SWE-agent 2.4.6 / SWE-bench 5.0.2 |
| Containers | official `swebench/sweb.eval.x86_64.*` images under Colima 0.10.3 and QEMU amd64 emulation |
| Agent budget | 40 API calls, 120-second command timeout |
| Generation | greedy, temperature 0, thinking off, seed 0, max 4,096 output tokens |
| Runtime | one worker, automatic native Q6 MTP plus Q4 draft LM head, prefix cache disabled |
| VQ v1 | revision `64b0fb0f98a552d91fb9abd5531d547b2e78c8a8` |
| VQ v2 | revision `87d89bc328f8226deb95e26d9a69c7cd1f353007` |

The official gold patch for the SymPy task was run first and resolved 1/1.
That smoke test, plus zero infrastructure failures in all model reports,
establishes that the emulated container path could apply patches and execute
the official tests. Apple Silicon required amd64 emulation because the official
evaluation images used here are x86_64. See the upstream
[evaluation guide](https://github.com/SWE-bench/SWE-bench/blob/main/docs/guides/evaluation.md),
[quickstart](https://github.com/SWE-bench/SWE-bench/blob/main/docs/guides/quickstart.md),
and the open [ARM64 evaluation issue](https://github.com/SWE-bench/SWE-bench/issues/520).

## Results

Wall time starts after each task container was ready and ends when its
trajectory was saved. It includes model generation and agent commands, but not
initial dataset loading or image download. `Empty` means the agent hit its
40-call limit without the scaffold's required final submission, so the official
evaluator had no patch to test.

| Instance | VQ v1 | Calls / wall | VQ v2 | Calls / wall |
|---|---:|---:|---:|---:|
| `sympy__sympy-15345` | **Resolved** | 33 / 14:53 | **Resolved** | 18 / 8:07 |
| `django__django-11451` | **Resolved** | 18 / 5:41 | Empty | 40 / 28:53 |
| `scikit-learn__scikit-learn-9288` | Empty | 40 / 34:56 | Empty | 40 / 32:00 |
| **Pilot total** | **2/3 (66.7%)** | **91 / 55:29** | **1/3 (33.3%)** | **98 / 69:00** |

Official report accounting:

| Checkpoint | Submitted | Completed | Resolved | Empty patches | Infra / evaluator errors |
|---|---:|---:|---:|---:|---:|
| VQ v1 | 3 | 2 | **2** | 1 | 0 / 0 |
| VQ v2 | 3 | 1 | **1** | 2 | 0 / 0 |

The 2/3 versus 1/3 difference is informative for this fixed subset, not enough
to estimate general SWE-bench capability. In particular, the result does not
show that v1 would score 66.7% on the full suite. It does show that v2's
HumanEval improvement did not transfer consistently to these multi-step agent
tasks.

## Runtime and memory observations

The agent sessions used a deliberately wider evaluation guard than the normal
40-GiB product goal: 30 GiB minimum reclaimable memory at startup, 4 GiB
minimum available memory, 40 GiB RSS, and 44 GiB process footprint. This kept
the fixed experiment running while still stopping an actual pressure event.

| Checkpoint | Peak footprint | Peak RSS | Minimum available |
|---|---:|---:|---:|
| VQ v1 | 41.92 GiB | 27.82 GiB | 7.45 GiB |
| VQ v2 | 41.90 GiB | 28.91 GiB | 7.90 GiB |

These peaks span the separate SymPy and two-task sessions. They are not new
sub-40-GiB claims. Prefix caching was disabled, and the service was shut down
normally after each checkpoint.

## Interpretation and next gate

- V1 remains the daily default: it resolved one more task, used fewer API calls,
  and finished the three effective task intervals about 20% sooner.
- V2 remains useful as a quality research candidate, not a production upgrade.
  Its 153/164 HumanEval result is real under that protocol, but it also has
  14/30 IFBench and 1/3 on this pilot versus v1's 18/30 and 2/3.
- Empty patches are agent-policy failures, not inference crashes. They do not
  prove incorrect numerical execution; they do matter to practical coding-agent
  usefulness.
- Do not tune on these same three tasks. A next model-selection gate should use
  additional predeclared Verified instances with the identical scaffold and
  limits. A full leaderboard claim requires the complete official suite.
