# Bounded checkpoint serialization experiment

Result: sequential materialization reduces the synthetic save allocation peak,
but this writer prototype is slower and is not promoted to production.

M5 Pro 64 GiB; no model loaded. Twelve independent FP32 banks, each a
[2,8192,128] slice of a [2,8320,128] backing. The head stride creates real holes;
logical file payload is96 MiB and retained source backing97.5 MiB. Each result
is loaded by the existing MLX reader and compared in full (shape/dtype/data and
metadata). Every run passed. This is not a BF16/GDN/MTP full checkpoint test.

| Writer | MLX peak | Save duration samples |
| --- | ---: | --- |
| Existing MLX save | 193.5 MiB | 14.93–16.90 ms |
| Prototype one bank at a time | 105.5 MiB | 72.77–80.40 ms |
| Prototype four banks at a time | 129.5 MiB | 73.67–76.59 ms |
| Prototype twelve at a time | 193.5 MiB | 71.77 ms |

The same writer with all twelve banks remains slow, so the difference is not
explained solely by the number of eval barriers. Additional A/B/B/A runs after
explicit contiguous-kernel warmup gave existing19.20/14.98 ms and sequential
70.75/76.94 ms, with unchanged peaks and passing roundtrips. Writing mechanism,
copy stream, ordering and buffering are not fully matched: attributing the whole
slowdown to synchronization would be unjustified. No fsync/durable-SSD timing;
filesystem cache and thermal state uncontrolled. Separate processes per sample.

Only allocation-lifetime savings are established:88 MiB lower peak for this
96 MiB strided payload. Actual process guard samples can miss short peaks and
include post-save verification, so they are not save physical-footprint proof.
Do not scale this ratio blindly to the model: already contiguous state may not
need a copy, and actual bank geometry/dtype differs.

The narrow prototype writes a fixed F32 safetensors fixture. It does not support
arbitrary checkpoint dtypes, crash-safe atomic publication, failure recovery,
or production quota semantics. No existing serializer, service or default changed.
Next investigation should align the write path before introducing an incremental
serializer; a memory-bound save mode may be useful only if measured latency is
acceptable in full request timing.

Safety: strict allocator1 GiB/cache16 MiB; guard start16, floor12, RSS/footprint4
GiB. Fresh Release build and five CTests passed; all13 experiment roundtrips pass.
Files retained under experiments/stream-save.L63Egj (13 x96 MiB payload,
approximately1.22 GiB); no old user data deleted. Raw outputs in JSON companion.
