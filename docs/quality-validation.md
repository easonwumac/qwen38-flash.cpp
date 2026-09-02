# Quality validation

## 2026-09-02 pilot A/B

The deterministic probe in `devtools/quality_ab.py` compares the exact target
path with the decode-oriented turbo path. It covers arithmetic, logic, Chinese,
structured extraction, instruction following, code reasoning, and long-context
needle retrieval. Answers are machine graded through a strict JSON wrapper.

Test configuration:

- Hardware: Apple M5 Pro, 64 GiB unified memory.
- Model: Qwen3.8 Flash Next REAP-288, affine 4-bit/group-64 target weights,
  retained Q8 MTP weights.
- Sampling: temperature 0, thinking disabled, at most 64 completion tokens.
- Both paths: MTP depth 4, prefix cache disabled, resident expert range 12:29.
- Exact control: speed profile, lossless16 compact metadata, target top-10,
  64 MiB allocator cache.
- Turbo: turbo profile, lossy9 compact metadata, target top-8, 32 MiB allocator
  cache.

Results:

| Measurement | Exact control | Turbo |
| --- | ---: | ---: |
| Completed, comparable cases | 31 | 31 |
| Correct | 23/31 (74.2%) | 23/31 (74.2%) |
| Same answer between paths | 26/31 | 26/31 |
| Short-request generation TPS, mean | 37.34 | 45.46 |
| Short-request generation TPS, median | 38.44 | 43.92 |
| Approx. 16K-token needle retrieval | pass | pass |
| Approx. 65K-token needle retrieval | pass in 432.96 s | pass in 123.77 s after batched MTP prefill |

The paired differences were balanced: the exact path alone passed the compound
percentage question, while turbo alone passed the Chinese ordering question.
This small pilot therefore found no aggregate quality regression in completed
cases. It is not sufficient to claim general benchmark parity. After
vectorizing MTP prompt-state priming, the isolated turbo rerun retrieved the
exact `R2-COMET-853` needle from 65,601 prompt tokens. Prefill took 122.65 s
(534.86 prompt tok/s), versus the previous 600 s client timeout. A repeated
30-case short suite kept the same 22/30 per-case result as the pre-change turbo
run.

The promoted long-context run used the turbo profile, MTP depth 4, prefix cache
disabled, adaptive chunk 512, packed QSA above 16,384 tokens, and the memory
guard's 48 GiB RSS/52 GiB footprint/8 GiB availability limits. Peak physical
footprint was 39.7 GiB, peak RSS was 34.7 GiB, and minimum available memory was
8.7 GiB. No active thermal control was applied, so 534.86 is a single guarded
validation point; nearby development runs spanned approximately 525--541
prompt tok/s rather than establishing a distribution-level 600 tok/s claim.

The long cases tokenize to 16,451 and 65,601 prompt tokens respectively. The
original local artifact for the exact run used older 8K/32K case labels; the
runner now names them by their measured token scale.
