# 256 hot + 32 SSD expert memory experiment

This measures the physical footprint of all 48 layers' expert weight buffers,
not a complete inference server. Model: retained Qwen3.8-Flash-Next REAP-288,
Q4/group64 with BF16 affine metadata; Apple M5 Pro, 64 GiB. No MTP, KV, dense
weights, PLE, routing, prompt or generation is executed in this layout mode.

`qwen38-slot-pool-probe MODEL --memory-layout` fills 256 expert slots per layer
from actual safetensor bytes and retains every buffer simultaneously. Then it
fills another eight slots per layer (384 cold experts total, 0.98877 GiB) to
represent a nearly full 1 GiB dynamic region. Buffers are page aligned and
registered with MLX. This is not an untouched virtual-memory reservation.

IDs 0..255 and 256..263 are geometry-only stand-ins. They are NOT a verified
REAP-256 subset or a learned hot/cold list. All experts within a layer have
equal geometry, so the memory result does not depend on which IDs are chosen.
Hit rates and throughput do depend on that choice and are not measured here.
The eight cold slots per layer are a layout test, not a global replacement policy.

## Measured candidate

| Quantity | GiB |
| --- | ---: |
| 256-expert hot payload across 48 layers | 31.6406 |
| Process footprint after loading hot weights | 31.6746 |
| Filled dynamic expert payload | 0.98877 |
| Process footprint with both retained | 32.6672 |
| 288-expert payload computed from same geometry | 35.5957 |
| Net payload saving after dynamic region | 2.96631 |

The separate all-288 baseline completed in the same representation at **35.6316
GiB process footprint** (35.5957 GiB payload). Thus the measured end-of-load
footprint difference is **2.9644 GiB**, approximately 8.32% of this expert-only
baseline. Baseline sampled peak was 35.6 GiB for both footprint and RSS. Swap
usage immediately before/after that run was unchanged at 2990.94 MiB; this is
system-wide existing swap, not memory attributable to this probe. Candidate
swap was not bracketed precisely, so no zero-swap claim is made for that run.

The sampled guard peak was 32.6 GiB footprint /32.7 GiB RSS. RSS and footprint
overlap and must not be added. Guard availability is a reclaimability heuristic,
not guaranteed allocation headroom. All MLX arrays were released (active bytes
zero), but process footprint remained 4.94 GiB until process exit; do not claim
all host allocator/framework memory immediately returned at clear().

This supports a roughly 3 GiB saving, not a reduction to 28 GiB. Subtracting
3 GiB from an otherwise identical 40 GiB process gives an illustrative 37 GiB,
NOT a measured server result. Historical approximately 40 GiB runs used varying
compact metadata, MTP and context settings; their peaks are not a directly paired
baseline for this raw BF16-metadata layout. Keeping compact metadata during
integration changes the saving and must be measured again.

## Reproduction and safety

Use the private strict MLX library and `devtools/memory_guard.py`, requiring
40 GiB starting availability, an external 36 GiB RSS/footprint limit and an
available-memory floor (candidate: 12 GiB; baseline: 20 GiB). Layout mode uses
a 36 GiB MLX cap and a 35.8 GiB internal early-stop check between layer loads.
Host allocation is not covered by the MLX cap; the external guard and bounded
per-layer allocation remain necessary. The default small kernel probe retains
its separate 1 GiB MLX limit and is run under a 4 GiB process guard.

`--memory-baseline` loads all 288 experts per layer, with no dynamic region, in
the same buffer representation. Run separately, never alongside the candidate.
Neither layout mode enables serving or creates new model files. Production
configuration is unchanged. Dynamic misses, PP, KV and MTP need later inference
integration and cannot be inferred from these memory measurements.

Validation: fresh Release build, default small slot replacement/parity probe,
CPU/MLX/tokenizer/memory-guard test suites, and both complete 48-layer layout
loads. No OOM or safety-limit termination in these layout runs.
