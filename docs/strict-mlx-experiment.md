# Strict MLX allocator experiment

This is an allocator-only experiment. It does not establish a process-memory
cap: non-MLX allocations, imported host buffers, framework overhead, and the
operating system remain outside the limit.

The probe installs a no-op MLX-C error handler so it can observe and recover
from an allocation refusal. MLX-C's default handler calls `exit(-1)`, and the
current server does not replace it. Therefore the private allocator must not be
linked into production until the server installs a nonterminating handler and
request-level allocation-failure tests prove graceful recovery.

## Provenance

The private sources under `../experiments/strict-mlx/` are normal local clones
of the MIT-licensed upstream sources already used by the installed runtime:

- MLX `1f8e74e3f12f31365464a6867c6579f0e9b29d85` (v0.32.2)
- MLX-C `56b2d39fc831f2c0eb5bb94d82ef7191f7b31fa6`

The machine lacked the optional Metal compiler component, so the private build
reused the existing metallib and generated JIT sources from that exact MLX
commit. Only `libmlx.dylib` was rebuilt; the unchanged installed MLX-C library
was used for the probe.

The private patch makes `MLX_STRICT_MEMORY_LIMIT=1` reinterpret the existing
memory limit as an allocation ceiling. Cache misses reserve aligned pending
bytes while holding the allocator mutex before it unlocks for `newBuffer`.
Failed or oversized allocations roll the reservation back. Cache reuse does
not increase committed bytes. Raw-pointer imports use the same reservation
path. Shrinking below active memory preserves existing buffers and rejects new
cache misses.

## Reproduction

From a clean checkout of the pinned MLX commit:

```sh
git checkout 1f8e74e3f12f31365464a6867c6579f0e9b29d85
git apply --check /path/to/qwen38-flash.cpp/source/devtools/patches/mlx-v0.32.2-strict-memory-limit.patch
git apply /path/to/qwen38-flash.cpp/source/devtools/patches/mlx-v0.32.2-strict-memory-limit.patch
```

Configure and build the private fork with at most two jobs, using
`devtools/memory_guard.py` with 16 GiB start, 12 GiB available, and 4 GiB RSS
and footprint limits. Configure MLX with:

```text
-DMLX_PREBUILT_METALLIB=<pinned-stage>/lib/mlx.metallib
-DMLX_PREBUILT_JIT_DIR=<same-commit-build>/mlx/backend/metal/jit
-DCMAKE_INSTALL_PREFIX=<project>/experiments/strict-mlx/stage
```

The two prebuilt inputs are optional experiment-only build aids. Omit them on
a machine with the matching Metal toolchain installed. When supplied, both
must come from MLX commit `1f8e74e3f12f`.

Run `qwen38-strict-allocator-probe baseline` against the installed runtime.
Run `qwen38-strict-allocator-probe strict` with
`MLX_STRICT_MEMORY_LIMIT=1` and the private stage first in
`DYLD_LIBRARY_PATH`.

The probe synchronizes the default GPU stream after evaluation and before
memory observations or final reclamation assertions. Without that completion
fence, recently freed arrays can remain owned by queued GPU work and produce a
false nonzero-active result.

## Result

With a 256 MiB cache limit and 512 MiB allocator limit, the installed allocator
accepted real nonzero 384 MiB and 256 MiB arrays concurrently (640 MiB active).
The private allocator accepted 384 MiB, rejected the next 256 MiB before its
allocation, rejected a 160 MiB managed import, admitted exactly one of two
concurrent 96 MiB requests, recovered after rejection, reused cache, and
rejected allocation after shrinking the limit to 320 MiB below 384 MiB active.
It exited with zero active/cache bytes and stayed below the 4 GiB guard.
Both baseline and strict modes repeated this result in three consecutive runs.

Private summary (MiB): peak MLX active 480; sampled physical footprint 708.
The import test deliberately allocated and touched its host backing before MLX
saw it; MLX rejected the import while physical footprint rose from 452 to 612.
That is direct evidence that a strict MLX allocator still cannot cap external
host allocation or total process footprint.

The concurrency test exercises pending reservations but is not an exhaustive
proof of all allocator interleavings. Before production use, add allocator-level
stress tests and explicitly budget process overhead. Weight staging must also
drop both projection owners and `MlxTensorStore` shard-map array owners.
