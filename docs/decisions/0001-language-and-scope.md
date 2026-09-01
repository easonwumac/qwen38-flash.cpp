# ADR 0001: C++20 server-only product

Status: accepted

## Decision

Build a new Qwen3.8 Flash Next-only inference server. C++20 owns all online
runtime behavior, Objective-C++/Metal implements Apple acceleration, and Python
is limited to offline tooling. No UI is included.

## Consequences

The engine can specialize its graph, storage, caches, verifier, and command
scheduling without generic-runtime overhead. In exchange, model support is
intentionally narrow and numerical correctness, tokenizer compatibility, API
behavior, memory safety, and operational lifecycle must all be implemented and
tested by this project.
