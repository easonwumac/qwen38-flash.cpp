# Bounded expert staging: integration gates

Research target: keep the current weights and target routing unchanged while
reducing **process physical-footprint peak** to 30 GiB. This is not yet a working
30 GiB model configuration. The private strict allocator and managed-region
probe are prerequisites, not an end-to-end inference result.

## Why a memory flag is insufficient

The current MLX safetensors path materializes tensors through allocator-backed
loads. `SparseMoe` owns complete projection arrays, and `MlxTensorStore` also
retains their shard-map owners. Releasing just one owner will not stage weights
out. Disabling `mlock` changes residency policy, not array ownership.

A bounded path must retain file metadata separately, select only required
expert regions, and drop every evaluated projection owner when its execution
finishes. Imported regions must stay alive through GPU completion. File-backed
mapping does not itself bound resident pages or prove zero-copy GPU access.

## Capacity is not throughput

For the base Q4, group-64 expert geometry (hidden width 2560, intermediate width
640), three projections with two BF16 affine parameters per group require:

```text
3 * 2560 * 640 * (4/8 + 4/64) = 2,764,800 bytes/expert
10 selected experts * 48 layers = 1.236 GiB/target decode step
288 experts * 48 layers = 35.596 GiB/all routed expert projections
```

These are geometry estimates, not measured traffic. They exclude dense weights,
PLE, embeddings, MTP, KV/state, caches and framework overhead. Hybrid Q8 layers
increase the total; compact qmeta changes the affine-metadata term.

At 45 target steps/s, even a **5% byte miss rate** on this baseline implies about
2.78 GiB/s of SSD reads, before other SSD users, read amplification and latency.
This is an illustrative demand calculation, not a measured SSD capability or
speed forecast. A 60% resident ratio does not imply a 60% cache hit rate. Measure
actual byte hits and reads on diverse prompts before predicting throughput.

## Required integration changes

1. Install a nonterminating MLX-C error handler and allocator/cache budgets
   **before model construction**. The current constructor sets cache size only
   after `model_` construction, which can already evaluate resident layers.
   MLX-C's default error handler exits the process, bypassing the executor's
   request exception handling. Prove failed requests can recover first.
2. Introduce byte-budgeted expert ownership separate from shard-map arrays.
   Include pending imports, temporary batches, GPU-pinned entries and mapping
   overhead; do not count logical cache removal as immediate physical release.
3. Preserve GPU router selections and weights exactly. The current fast path
   keeps expert IDs on the GPU; CPU-driven SSD selection introduces a route-ID
   readback/fence. Do not substitute CPU routing and silently change ties/order.
4. Remap original expert IDs into a compact staged batch for the existing
   kernels. Key entries by full tensor/layer identity, not expert ID alone.
   Keep MTP and target namespaces distinct.
5. Bound the lifetime of lazy graphs. Decode and multi-layer MTP verification
   may retain multiple layers of inputs. Evaluate and retire a bounded group
   before reusing its staging allocation; measure the extra barrier cost.
6. Handle prefill separately: many token routes can touch most experts in a
   layer. Bound temporary unique-expert batches and avoid evicting the entire
   decode cache for one prefill chunk. Any regrouped reduction needs parity
   testing, not just byte equality of source weights.

## Promotion ladder

- Managed-region import: exact GPU byte reads, alias/copy reporting, safe
  mapping lifetime and strict allocation refusal, under a 1 GiB process guard.
- One-layer staged MoE: exact selected IDs/weights and output parity against
  bounded reference inputs; report cold/warm I/O, peak, and synchronization cost.
- Repeated layer/cache replay: byte cap, pin accounting, eviction and measured
  physical reclamation; include low-locality and prefill-like selections.
- Short full-model inference only after the complete live-byte budget is
  established. Then increase context and enable MTP separately. Record RSS,
  physical footprint, MLX active/cache bytes, page-ins, swaps, PP and decode.

Typed projection gate now passes after a correction: this retained pack's odd
payload offset cannot feed stock U32/BF16 QMM directly. Exact expert U8 bytes
must first be copied into aligned GPU staging buffers. The probe checks this
address requirement and exact BF16 projection outputs; see
[managed expert-region results](managed-expert-region.md). Budget both imported
window and staging buffer while their GPU work overlaps. Do not equate the raw
window alias counter with zero-copy inference.

Boundary and selected-branch gate now passes for deterministic nonconsecutive
expert IDs including 0 and 287. Valid header/adjacent-tensor bytes can be included
inside the explicit full-file span; EOF tails use owned zero-padded windows.
Selected gate/SiLU/up/down/weighted ordered sums agree exactly with copied
oracles. The router/shared/fused path and byte-budgeted eviction remain separate
integration gates, not implied by these arithmetic checks.

No full-model launch is justified by the import probe alone. The 30 GiB process
budget needs a measured reserve outside MLX; polling guards are a secondary
stop mechanism, not an allocation-time guarantee.

Keep RSS beside physical footprint in every result: clean file-backed resident
pages and framework accounting can make the two diverge. A lower footprint
counter with unchanged system memory pressure or rising swap is not a successful
memory reduction. Report system reclaimability and swap deltas as well; do not
add RSS and footprint together, because their contents overlap.
