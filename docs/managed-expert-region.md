# Managed expert-region probe

`qwen38-managed-expert-region-probe` is a developer-only, bounded experiment.
It does not change model loading or production MoE execution and never asks
MLX to load a complete projection or a complete shard tensor.

The probe opens only the layer-0 safetensors header and obtains CPU views from
`SafetensorsFile`. For cumulative selections U1, U8 and U10 it imports a page-aligned U8
window around each selected gate/up/down weight, scale, and bias range, then
uses an MLX U8 slice for the exact expert bytes. This keeps every managed raw
pointer page-aligned while still exercising the real byte geometry. The probe
records whether MLX's managed raw-pointer constructor aliases the source
pointer or falls back to a copy, then compares every window and exact slice
with a separately copied bounded CPU oracle through `mlx_array_equal` on the
GPU stream. It drops the catalog owner and repeats fresh GPU comparisons to
prove the managed payload retains the mmap.

Each managed payload has an independent shared mapping owner and cleanup
counter. The probe also lowers the strict allocator limit to the currently
active amount and verifies a 1 MiB managed import is refused without a double
cleanup. The MLX-C error handler is installed as a no-op because the default
handler terminates the process on allocator errors.

## Run

Use the private strict MLX build first in the dynamic-library path. The
workflow's external guard should wrap the one GPU process with a 16 GiB start,
12 GiB available, and 1 GiB footprint/RSS ceiling:

```sh
# Run from the project capsule (the parent of source/).
python3 source/devtools/memory_guard.py \
  --min-start-gib 16 --min-available-gib 12 \
  --max-rss-gib 1 --max-footprint-gib 1 --interval 0.1 -- \
  env DYLD_LIBRARY_PATH="$PWD/experiments/strict-mlx/stage/lib" \
  MLX_STRICT_MEMORY_LIMIT=1 \
  "$PWD/source/build-all/qwen38-managed-expert-region-probe" \
  /path/to/Qwen3.8-Flash-Next-REAP-288-MLX-4bit-repacked-MTP-Q8-REAP288-L47 \
  --rounds 3
```

Configure/build with at most two jobs and the pinned MLX-C include/library
paths. The probe sets a 512 MiB MLX memory limit and 16 MiB MLX cache limit
before creating any MLX arrays. Host mmap pages and host backing for the
intentional refusal test are outside that allocator limit, so the output's
physical footprint remains a separate safety measurement.

The page-aligned window is bounded by the owner's full read-only mapped span,
so it can include adjacent tensor/header bytes. If its final page extends past
EOF, it uses an owned, zero-padded aligned buffer and copies only valid file
bytes; it never reads past EOF.
An alias result is evidence only for the tested pointer/length/MLX build and
hardware. A copy result is a supported fallback, not a failed correctness
case. No conclusion about production zero-copy paging follows without this
probe's result on the target machine.

## Initial byte-read result (a8b1611)

The retained layer-0 Q4/group-64 geometry has 819,200 weight bytes per expert
per projection, plus BF16 scales/biases. On Apple M5 Pro / 64 GiB / macOS 26.5
(25F71), with the pinned private strict allocator, all 171 windows aliased their mmap
source; none used the copy fallback. All window/exact-slice GPU byte comparisons
passed, including fresh comparisons after catalog owners were dropped. Weak
mapping owners expired after final GPU completion and array release. Every
successful payload callback ran once. The intentional 1 MiB import was refused
and its untransferred payload was cleaned manually once.

U1, U8 and U10 are **cumulative** within a round: 9 * (1 + 8 + 10) = 171
windows remain live, including overlapping/repeated experts. They are not 171
distinct experts or a cache eviction test. Three rounds in the same process
passed with MLX active peak about 54 MiB each, lifetime physical-footprint peak
about 153 MiB, and final active/cache 0/0 after each round. Physical footprint
settled near 130 MiB and RSS near 76 MiB each time: allocator zero does **not**
mean the process returned all physical memory. No monotonic growth appeared
in these three rounds; this is not a long-duration leak proof.

The short run can finish between external monitor samples, so the probe also
reports macOS's lifetime maximum physical footprint. Internal measurement
failure is fatal rather than silently treated as zero.

Limitations: the two layer-0 shard filenames are specific to the retained
131-shard pack. The managed-import copy fallback was not exercised on this hardware.
There is no real routing, shared-expert/fused execution, cold-SSD benchmark or
full-model run yet. Deterministic selected-expert arithmetic is now tested below.
This validates a managed GPU-readable region primitive, not 30 GiB
inference or preservation of decode/PP performance.

## Typed Q4 computation and alignment correction

The next test caught a real integration hazard: the retained layer-0 payload
starts at byte 845 in the shard. Expert weight/scales/biases therefore have an
odd byte offset (the tested addresses were 13 modulo 16). U8 window equality
passes, but viewing that same address as U32/BF16 and sending it directly to
stock QMM produced NaNs and incorrect outputs. Copied input bytes were identical
and the copied QMM outputs were finite. The pinned MLX `View` shares contiguous
buffers without checking typed-pointer alignment; `Slice` and `Reshape` retain
the offset. Consequently, byte-readable aliasing is **not** typed-QMM eligibility.

The probe now conservatively requires 16-byte alignment for direct typed views.
Misaligned expert slices are copied by a small U8 Metal kernel into a new aligned
MLX allocation, then viewed/reshaped as U32/BF16. Only exact expert bytes are
staged, never a whole projection. `mlx_copy` was tested and rejected as a means
to force alignment: it is a logical copy that can still share the odd-offset
backing. The explicit byte kernel verifies both a new address and alignment.
The reference independently copies CPU oracle bytes into an aligned U8 array
before viewing them; it never dereferences an odd CPU pointer as U32/BF16.

For each selected expert's gate/up/down projection, BF16 inputs with deterministic
signed values exercise rows 1, 4 and 32. The checks cover input byte equality,
output shape/dtype, finite results on both paths and exact BF16 output bits.
No tolerance was relaxed to obtain a pass. Each round passes 171 QMM comparisons;
three rounds pass 513 comparisons. Every tested typed projection required staging
(`typed_aliased=0`, `typed_staged=171` per round), although all 171 raw byte
windows still alias their mappings.

On the same M5 Pro / 64 GiB system, the corrected three-round run reached about
56 MiB peak MLX active and 159 MiB lifetime peak physical footprint. Each round
returned MLX active/cache to 0/0; process footprint settled at 133–134 MiB.
These figures include copied validation oracles. They are not a production cache
budget or throughput measurement. Routed MoE reductions and cache eviction are
the next gates. An aligned on-disk repack could remove this particular staging
copy, but no weights were repacked or modified in this experiment.

Verification: fresh developer-target build and coordinator full build passed
with two compiler jobs. CTest: core, memory guard and MLX tests passed;
the tokenizer fixture-dependent test was skipped. Existing dylib deployment
target warnings (26.0 executable versus 26.2 libraries) remain. These are
correctness/memory probes, with no prompt corpus or thermal-controlled timing
measurement and no MTP execution.

## Boundary experts and selected-expert composition

The current selections take the first 1, 8 or 10 entries from
`[0, 287, 31, 129, 7, 256, 63, 17, 201, 95]`. This includes both boundary experts
and nonconsecutive interior IDs. Each typed input's oracle is also compared to
the original tensor row for that ID, preventing a shared indexing mistake in
the staged and copied paths from giving false parity.

`SafetensorsFile::mapped_view()` exposes the read-only full file extent while
retaining the existing ownership requirement. A unit test checks its size and
tensor containment. Page windows use checked address extents. Leading windows
can include valid header bytes; trailing partial EOF pages use aligned owned
storage, zero-filled beyond the actual file. Managed payloads retain that storage,
and weak-owner assertions verify it is released after final GPU completion.
Each current round has 169 file-backed windows and 2 padded owned windows.
`alias=171` means MLX aliases all supplied window buffers, **not** that all 171
still alias the file itself.

For each U/row combination, the probe computes the full selected-expert branch:
gate QMM + SiLU, up QMM, hidden multiply, down QMM, BF16 scalar probability
multiply, then ordered BF16 additions. This matches the non-fused reference's
arithmetic order, not a new reduction tree. Inputs are `[1, rows, 2560]` BF16;
every intermediate is checked for shape, dtype and finite values. Fixed positive
weights `(slot+1) / (U*(U+1)/2)` are synthetic selections, not router outputs.

Three rounds pass 27 selected-expert output comparisons and 513 projection
comparisons with exact BF16 bits. With the added original-row identity checks,
the same-machine lifetime footprint peak is about 168–169 MiB, MLX active peak
about 56 MiB, and final MLX active/cache 0/0 each round. Footprint settles near
135–136 MiB. A fresh full Release build and its repeated probe passed; CTest
passed core, memory guard and MLX tests, with the tokenizer fixture test skipped.
This includes diagnostic oracles and repeated imports; it is not an optimized
runtime memory figure. No weights were changed. Real GPU routing, shared expert,
fused kernels, bounded cache eviction and full-model inference remain untested
on this staging route.
