# Prefix snapshot memory probe

This developer-only probe compares a retained 32K prompt state with saving and
releasing that state before independently materializing a 65K live state. It
uses the retained model geometry but no model weights.

Run each mode separately so the memory guard lock and peak counters are clean:

```sh
python3 devtools/memory_guard.py --min-start-gib 16 --min-available-gib 12 \
  --max-rss-gib 4 --max-footprint-gib 4 -- \
  ./build/qwen38-prefix-memory-probe baseline
python3 devtools/memory_guard.py --min-start-gib 16 --min-available-gib 12 \
  --max-rss-gib 4 --max-footprint-gib 4 -- \
  ./build/qwen38-prefix-memory-probe release
```

Repeat three times, one process at a time. `prompt_logical_bytes` and
`live_logical_bytes` are BF16 payload accounting, while physical footprint is
the safety/reclamation observation. Save/load latency is not TTFT-neutral and
must be weighed against any measured footprint reduction. The tool owns and
validates its unique temporary directory and caps its state file at 3 GiB.

## 2026-09-06 measurements

Three guarded runs per mode passed target token count and array shape/dtype/count
parity; a subsequent guarded release run also passed first/last-value parity for
every persisted array. This does not check all state metadata or model quality.
The baseline final physical footprints were 3,219,505,824, 3,219,620,632, and
3,219,604,272 bytes. Release-mode results were 2,152,481,440, 2,152,612,632,
and 2,152,448,792 bytes. The median reduction was 1,067,122,832 bytes
(0.994 GiB), closely matching the 1,067,769,856-byte synthetic prompt payload.

Release-mode save times were 134.2, 97.2, and 117.0 ms (median 117.0 ms); load
times were 29.4, 30.0, and 29.3 ms (median 29.4 ms). The processes remained
under the 4 GiB cap and minimum reported availability stayed above 30 GiB.
Because these runs complete faster than every external guard sampling interval,
the probe's final `ri_phys_footprint` is the comparison metric; the guard remains
the safety boundary. This supports roughly 1 GiB—not 2–4 GiB—of directly
attributable reclamation for one retained 32K prompt snapshot in this geometry.

Load follows save immediately: these are warm filesystem-cache timings, not
cold SSD latency or durable-write throughput. Final footprint is not peak
footprint; transient serialization peaks can fall between guard samples.
The longer live state is independently materialized without real decode, so
production allocation sharing and service-level gains remain unverified.

Coordinator rebuild/recheck: baseline final 3,219,587,816 B; release final
2,153,562,856 B (1,066,024,960 B / 0.993 GiB lower). Save 108.126 ms,
warm load 33.674 ms; sampled parity passed. Guard minimum availability was
34/33 GiB respectively. The baseline finished before a footprint sample, so
the guard's reported zero peak is missing sampling evidence, not zero usage.
