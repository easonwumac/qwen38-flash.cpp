# Dynamic expert cache: configurable steady target and ceiling

The target applies to **total process memory**, not just expert tensors. Requested
steady/ceiling pairs include 8/16, 12/20, 16/24, 20/30 and 24/32 GiB. A steady
target is not a mandatory minimum allocation. None is currently a verified
full-model configuration. The cache implementation and replay described here are an
integration prerequisite, not permission to lower the existing startup guard.

All experts remain on disk and all selected experts must be evaluated. Cache
misses wait for the original weights; no expert is silently dropped, substituted
or requantized. This preserves the weight artifact, not a guarantee of model
quality before full execution parity is tested.

## Ownership and policy

`ExpertCache<Value>` is single-inference-thread infrastructure. It tracks a
declared byte charge per entry, uses decaying frequency and recency for eviction,
and provides shared leases. An entry with an outstanding lease cannot be evicted.
The caller must hold that lease through GPU completion; a shared MLX array or a
lazy graph alone is not a lease recognized by this cache.

Admission makes space before invoking the loader. Failure or a null loader
result must not leak byte reservations. Reentrant loader calls are rejected.
Changing the budget can temporarily leave already-pinned entries above the new
desired limit; the operation reports that it could not finish shrinking. New
admissions cannot bypass the limit, and `trim()` retries when leases are released.
This behavior is suitable for a soft-target reduction, not an already-violated
hard process ceiling. Account pinned entries until they are really released.

## Full-runtime integration still required

The expert cache cannot be assigned the whole process allowance. Its budget must subtract
measured non-expert weights, KV/recurrent states, MTP verification states,
prefix-cache state, pending staging buffers and allocator/framework reserves.
Memory-pressure handling must shrink the expert tier before reserving growing
context state. If pinned work prevents safe admission, wait, split the work or
reject the request; do not skip experts or allocate beyond the total budget.

The earlier private MLX strict allocator only gates its own buffers. It is not
a process-wide ceiling, and the current production loader/MLX-C error handler
has not yet been integrated with this staged route. RSS, physical footprint and
system memory pressure still need independent observation.

`plan_expert_budget()` supplies an overflow-checked accounting plan for arbitrary
steady/ceiling pairs. It subtracts caller-supplied non-expert, transient and safety
costs, then applies an absolute expert allowance from system-pressure policy
(zero means none; the default means no additional pressure cap). It reports
steady feasibility separately from burst feasibility. Costs must use one
consistent conservative resident accounting domain without double-counting;
MLX logical counters cannot simply be subtracted from physical footprint.

The planner does not enforce a process ceiling or sample system pressure. Before
allocating, the caller must trim/fence and recheck actual headroom. Hysteresis,
observed memory costs, pressure sampling and staged production loading remain
integration work. CPU tests cover all five ranges, state growth, pinned shrink,
burst admission, zero pressure allowance, infeasible fixed costs and overflow.

## Bounded real-weight replay

The developer probe's `--cache-replay` mode exercises an 8 MiB expert cache,
shrinks it to 4 MiB, then grows it back to 8 MiB. It uses real layer-0 Q4/group64
expert gate/up/down weights and BF16 affine metadata. The replay holds two
leases, attempts a third admission, verifies the loader was not called, and
checks that a blocked shrink succeeds after a lease is released.

Metal allocations are page-rounded: one expert costs 2,850,816 bytes in the
pinned runtime, rather than its smaller tensor byte sum. The replay temporarily
disables the underlying MLX free-block cache so oversized reused buffers cannot
invalidate this declared charge. It asserts the actual active-byte increase of
each load. This is an accounting experiment, not a recommended production cache
setting or speed result.

Rows 1/4/32 and selections of 1/8/10 deterministic nonconsecutive experts compare
the complete selected branch after evictions/reloads against bounded copied
oracles. Imports and CPU validation oracles are temporary per load; the expert
cache owns evaluated aligned weight buffers, not complete shard tensor maps.
The replay excludes the real router, shared experts, fused kernels and MTP.

Run from the source directory with the private strict library available:

```sh
python3 devtools/memory_guard.py --min-start-gib 16 --min-available-gib 12 \
  --max-rss-gib 1 --max-footprint-gib 1 --interval 0.1 -- \
  env MLX_STRICT_MEMORY_LIMIT=1 \
  DYLD_LIBRARY_PATH=/path/to/strict-mlx/stage/lib \
  build-all/qwen38-managed-expert-region-probe /path/to/model --cache-replay
```

The replay is a correctness/ownership test with warm OS file caching possible.
Its synthetic hit rate is not representative of conversation workloads. No
20 GiB full-model success or token-speed estimate follows from it.

## Verified bounded result (2026-09-06)

On Apple M5 Pro / 64 GiB, the retained REAP-288 Q4 model's layer-0 replay passed
all nine exact BF16 selected-branch comparisons. It recorded 11 hits, 53 misses
(including one pinned refusal), 52 loads and 50 evictions. Expert-cache peak was
5,701,632 bytes; MLX peak was 11 MiB and lifetime physical-footprint peak 115 MiB.
After clearing, MLX active/cache bytes were zero. The external guard allowed at
most 1 GiB RSS/footprint and required 12 GiB system headroom. These are bounded
correctness results, not decode throughput or a full-model memory benchmark.

A clean Release build and CTest completed: three suites passed and the tokenizer
fixture suite was skipped. The existing three-round region/composition regression
also passed 513 QMM and 27 selected-branch comparisons. Its physical-footprint
peak was 191 MiB and final footprint 158 MiB despite MLX active/cache returning
to zero; framework/host residual memory must therefore remain in the process
budget rather than being treated as reclaimed MLX bytes.
