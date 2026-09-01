# Architecture

## Scope

The engine serves only Qwen3.8 Flash Next. Specialization is intentional: generic
model abstractions must not compromise hot-path scheduling, weight layout, state
reuse, or Metal fusion.

## Layers

1. **Server** — HTTP lifecycle, OpenAI-compatible schemas, streaming,
   cancellation, limits, maintenance APIs, and observability.
2. **Runtime** — request scheduling, batching policy, tokenization, sampling,
   model lifecycle, memory accounting, and cache ownership.
3. **Qwen3.8 graph** — PLE, hybrid linear/full attention, sparse n-gram
   embeddings, routed/shared MoE, and MTP verification.
4. **Execution backend** — correctness backend first, then Apple-specific
   Objective-C++/Metal command scheduling and fused quantized kernels.
5. **Storage** — memory-mapped safetensors, optional converted packed weights,
   resident expert tiers, and opt-in SSD-backed cache/paging.

## Language decision

C++20 is the primary language because the runtime requires deterministic object
lifetimes, direct memory mapping, low-overhead scheduling, interoperability with
Metal, and portable testing without a language runtime on the serving path.
Objective-C++ is restricted to Apple framework boundaries and Metal orchestration.
Metal Shading Language implements GPU kernels. Offline Python tools may inspect,
convert, or evaluate weights but never serve requests.

## Correctness policy

Optimization proceeds from fixed parity fixtures. Each specialized operation must
match reference layer outputs within a declared tolerance and preserve greedy
token sequences. Approximate attention, prompt-specific shortcuts, and unchecked
quantization are not accepted into the default path.

## Performance policy

The engine owns the decode loop so verification, cache transitions, command
buffers, and synchronization can be fused across layer boundaries. A change lands
only when controlled A/B evidence improves the relevant distribution without a
quality, memory, context, or stability regression.
