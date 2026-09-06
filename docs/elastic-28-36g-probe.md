# 28 GiB steady / 36 GiB ceiling probe

This changes only the memory range of the [24/32 GiB experiment](elastic-24-32g-probe.md).
The current resident server is untouched. The experimental paged path is still
the reference per-expert QMM / rowwise PP route, with batched expert leases and
fences, not the existing fused decode or grouped prefill path. Therefore these
low throughput numbers do not establish the limit of a fully optimized paged
engine, nor mean the existing resident engine regressed to this speed.

## Configuration and safety

`--elastic-28-36` requests 28 GiB steady / 36 GiB peak **process** memory. The
short-workload planner leaves 4 GiB for base/state, so the expert allowance is
24 GiB steady and at most 28 GiB during work. Private strict MLX is limited to
32 GiB, with its free-block cache disabled. An external 36 GiB RSS/footprint
guard checks every 0.1 seconds, requires 40 GiB start headroom and 12 GiB system
headroom throughout. Internal phase/token checks stop above 34 GiB; after
trimming, footprint must be below 28 GiB. These layered checks are not a hard
process allocator cap, and the 4 GiB state estimate is not for long contexts.

Hardware/model/workload match the preceding experiment: Apple M5 Pro / 64 GiB,
macOS 26.5; retained REAP-288 Q4 / MTP-Q8-L47 artifact, no MTP enabled. Greedy
generation starts at token 9419 for 256 steps, without chat template/EOS stopping.
The eight-token PP test uses a fresh state after generation and shares the
expert cache. No weight or routing changes. Thermals and OS file caching are
uncontrolled; comparisons are observations, not causal RAM-speed estimates.

## Reproduce

```sh
python3 devtools/memory_guard.py --min-start-gib 40 --min-available-gib 12 \
  --max-rss-gib 36 --max-footprint-gib 36 --interval 0.1 -- \
  env MLX_STRICT_MEMORY_LIMIT=1 DYLD_LIBRARY_PATH=/path/to/strict-mlx/stage/lib \
  build-elastic36-verify/qwen38-paged-model-probe /path/to/model --long --elastic-28-36
```

The regular `--elastic` retains the previous 24/32 GiB range. This is a developer
probe option, not a deployed server setting.

## Result (2026-09-06)

| Metric | 28/36 GiB run |
|---|---:|
| Model load | 1.601 s |
| Decode, excluding first of 256 tokens | 2.879 tok/s |
| Step median / p95 | 0.298 / 0.774 s |
| Eight-token PP | 3.725 tok/s |
| Peak process footprint | 30.6 GiB |
| Footprint after steady trim | 27.327 GiB |
| Minimum system headroom | 18.0 GiB |
| Expert cache after decode | 26.725 GiB |
| Expert cache after trim | 23.999 GiB |
| Decode hits / misses / evictions | 112,344 / 10,536 / 470 |
| Cumulative miss loading | 50.778 s |

Growth and shrink completed, and the explicit below-28-GiB steady check passed.
The workload did not fill the 28 GiB burst expert allowance. No OOM occurred.
The 256-token hash `12818014199420254704` and full F32-converted PP logit hash
`12037094211002573605` match the preceding memory-range experiments. This does
not replace a full quality evaluation.

Compared with 24/32 GiB, decode misses declined from 10,918 to 10,536 (3.5%).
There was no observed material throughput improvement: the earlier repeat was
3.008 tok/s, with 46.492 seconds of miss loading. OS caching and SSD read
conditions differ, so this single run does not prove larger RAM makes inference
slower. It does show that increasing this allowance alone did not restore useful
speed. Fused selected-expert compute and grouped paged prefill remain missing;
the experiment is not a replacement for the existing resident server.

Fresh Release full build passed under a 4 GiB build guard (0.3 GiB footprint /
0.4 GiB RSS). CTest: three suites passed; tokenizer fixture suite skipped. The
pure memory-budget test now additionally covers the 28/36 GiB range.
