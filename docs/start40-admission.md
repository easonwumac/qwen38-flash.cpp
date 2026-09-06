# 40 GiB startup admission trial

Outcome: admitted, then safely aborted by the available-memory floor. Do not
promote 40 GiB as a validated daily startup threshold from this run.

M5 Pro, 64 GiB. Qwen3.8-Flash-Next REAP-288 affine Q4/group64 target,
lossless16 qmeta, MTP-Q8-REAP288-L47 model capsule, speed profile, resident
expert range 12:29. The developer smoke bootstraps token 9419, then intends to
compare four serial greedy tokens against depth-2 MTP. No prompt cache or
long-context prefill is involved. This is an admission/correctness smoke, not
the historical 128-token throughput fixture.

Single run: start threshold 40 GiB, available floor 12 GiB, process-tree RSS and
physical-footprint ceilings 42 GiB, polling interval 0.1 seconds. Private strict
MLX runtime; allocation-time ceiling 40 GiB and cache 256 MiB. Environment was
cleared before explicitly setting the profile overrides. No service restarted
and no persistent startup threshold was changed.

Observed output:

```
[phase] loading target and MTP head
[phase] bootstrap decode
[qmeta] compact 16-bit routed MoE engaged: rows=1 topk=10
memory_guard: stopping: footprint=36.5 GiB, rss=17.5 GiB, available=11.5 GiB
[phase] serial greedy oracle
```

Exit 76: the guard stopped the process. The final phase message can race with
the stop signal; no serial/MTP result JSON was produced. Therefore there is no
completed parity result or token-rate measurement. The observed footprint is
the stopping sample, not a proven absolute lifetime peak. No 8K/32K escalation.

Before/after system observations (not isolated process attribution):

- Swap used: 2806.94 MiB both times; swapout counter unchanged at 3,813,310.
- Compressor occupied pages: 165,816 -> 239,877 (16 KiB pages), about
  2.53 -> 3.66 GiB. Unchanged swap does not mean no memory pressure.
- After exit, available heuristic recovered to about 48.25 GiB, but some of
  that headroom accompanies increased system compression; it is not evidence
  of engine memory savings or a reason to immediately repeat the load.
- Process absence was checked. Polling protection cannot guarantee no OOM.

The smoke now has an explicit allocator cap, optional profile argument, and
phase markers for reproducible guarded admission tests. Production defaults
are unchanged. A fresh Release build and all five CTests passed; this does not
turn the aborted full-model run into a parity pass.
A safer next step is reducing baseline working-set overlap or
testing when fewer unrelated applications occupy memory, not reducing the
12 GiB available floor.

## User-approved 8 GiB floor follow-up

The user subsequently approved an experimental 8 GiB available-memory floor
with automatic termination, not alert-only behavior. Reused the verified
binary and identical model/profile/allocator settings above; start threshold
40 GiB and process ceilings 42 GiB remained unchanged. This was a separate
short-context smoke, not a long-context escalation.

```
[phase] loading target and MTP head
[phase] bootstrap decode
[qmeta] compact 16-bit routed MoE engaged: rows=1 topk=10
[phase] serial greedy oracle
[phase] MTP rounds
memory_guard: stopping: footprint=38.0 GiB, rss=26.2 GiB, available=7.8 GiB
```

Exit 76 again. The process reached MTP but did not emit completed parity or
throughput results. Its absence was checked after termination. No further
floor reduction or longer-context run was attempted. These are stopping
samples, not exact lifetime peaks; polling may overshoot a threshold.

Before/after swap use remained 2806.94 MiB and swapouts stayed 3,813,310.
Compressor occupancy increased from 205,257 to 312,906 pages, approximately
3.13 to 4.77 GiB. These system-wide measurements show additional compression,
not zero memory pressure. Initial available heuristic was about 46.53 GiB,
so merely passing a 40 GiB startup check clearly does not certify this workload.
The post-exit available heuristic recovered to about 49.40 GiB, with more
compressed system state. No persistent service/guard defaults were changed.
