# Daily operations

This runbook operates the server-only engine without modifying another local-LLM
service. Use a distinct loopback port while validating a new build.

## Build

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DQWEN38_ENABLE_TOKENIZER=ON \
  -DQWEN38_MLXC_INCLUDE_DIR="$MLXC_INCLUDE_DIR" \
  -DQWEN38_MLX_LIBRARY_DIR="$MLX_LIBRARY_DIR"
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure
QWEN38_TEST_MODEL="$MODEL_DIR" \
  ctest --test-dir build-release --output-on-failure -R qwen38-tokenizer-tests
```

The checkpoint must contain `config.json`, `model.safetensors.index.json`, every
indexed shard, tokenizer files, and at least one supported n-gram table.
Row-addressable `ngram_table.bf16.aos`, affine `ngram_table.q8.aos`, and Q4
`ngram_table.bin.aos` are detected in that precision order and read with bounded
`pread` calls, so they remain SSD-backed. The original `ngram_table.bin`
safetensors file is the fallback and remains useful for oMLX compatibility.
Niwaki checkpoints can instead supply `niwaki_ple_pair` and
`niwaki_ple_quant`; their paired Q2 table is read directly from the indexed
model shards through mmap. Compact qmeta and MTP are optional sidecars; never
edit the upstream shard payloads to install them.

## Automatic tuning

The server exposes one production policy: exact top-10 routing, adaptive
1,024-row prefill, Q8 KV after 8K, QSA after its context threshold, four-slot
continuous batching, automatic lossless16 metadata when its sidecar is present,
and adaptive MTP when compatible assets are present.
Flags that remain in this runbook are resource limits or reproducibility
controls, not alternate performance profiles.

When a compatible MTP companion is supplied, automatic tuning probes its
economics and falls back to serial target decode after repeated unprofitable
rounds. Use `--mtp-depth off` only as a resource limit when the companion's
allocation does not fit.

## Launch and verify

```bash
DYLD_LIBRARY_PATH="$MLX_LIBRARY_DIR" \
./devtools/memory_guard.py -- ./build-release/qwen38-server \
  --host 127.0.0.1 --port 11438 --model "$MODEL_DIR" \
  --prefix-cache-tokens 8192 \
  --max-generation-tokens 4096

curl -fsS http://127.0.0.1:11438/healthz
curl -fsS http://127.0.0.1:11438/readyz
curl -fsS http://127.0.0.1:11438/v1/status
curl -fsS http://127.0.0.1:11438/metrics
```

For a guarded Q8 long-context run, keep MTP and the RAM prefix cache off:

```bash
QWEN38_RESIDENT_EXPERT_RANGE= DYLD_LIBRARY_PATH="$MLX_LIBRARY_DIR" \
./devtools/memory_guard.py --min-available-gib 6 -- \
  ./build-release/qwen38-server \
  --host 127.0.0.1 --port 11438 --model "$MODEL_DIR" \
  --mtp-depth off --prefix-cache-tokens 0 \
  --qmeta-cache-max-prompt-tokens 262144 --qmeta-cache-layers 8 \
  --qsa-packed-min-tokens 32768 --qsa-shared-rows 4 \
  --kv-cache q8 --kv-q8-min-tokens 65536 \
  --kv-q8-flush-tokens 8192 \
  --prefill-chunk 512 --prefill-chunk-fixed
```

See [Q8 KV cache](q8-kv-cache.md) for the current validation boundary.

Treat `readyz != 200`, a nonempty `last_error`, a rising cancellation count, or a
memory-guard exit as an operational signal. The guard exit codes are 75 for
admission refusal, 76 for a memory limit, and 77 for an unsafe measurement
failure. Signals 130/143/129 represent controlled SIGINT/SIGTERM/SIGHUP shutdown.
For long unattended capacity runs, add `--report-json "$REPORT_PATH"` before
`--`; the guard persists the exit reason and peak footprint/RSS/available-memory
summary even when it terminates the server.

Use SIGINT or SIGTERM for shutdown. The guard forwards termination to the
isolated server process group and reports peak footprint/RSS/available memory.

## Cache policy

The RAM prefix cache holds one complete recurrent state and is intended for an
append-only next chat turn. It is not expected to hit when the same user-only
request is replayed after the cache has been extended through assistant output.

Enable restart persistence deliberately:

```bash
--prefix-cache-tokens 8192 --ssd-prefix-cache-gib 8 \
--ssd-prefix-cache-dir "$CACHE_DIR"
```

The SSD cache is namespaced by model/runtime compatibility and bounded LRU. Clear the
active namespace only through `POST /admin/cache/clear`; this is destructive to
cached state and should not be part of routine health checks.

## Recovery

1. Save `/v1/status`, `/metrics`, the startup line, guard report, and request ID.
2. If admission was refused, close unrelated memory-heavy applications or select
   `memory`/`long-context`; do not weaken both guard limits at once.
3. If a request failed but the server remains ready, issue a one-token smoke and
   inspect `last_error` before restarting.
4. If the server is not ready, stop it normally, verify model files and sidecars,
   then restart on an isolated port.
5. Reproduce with MTP off and caches off before classifying a kernel/model fault.

## Reproducible checks

```bash
python3 devtools/mtp_benchmark.py --url http://127.0.0.1:11438 \
  --tokens 128,256 --warmups 1 --samples 3
python3 devtools/mtp_mixed_probe.py --url http://127.0.0.1:11438 --tokens 128
python3 devtools/prefix_cache_benchmark.py --url http://127.0.0.1:11438
python3 devtools/long_context_benchmark.py --url http://127.0.0.1:11438
python3 devtools/server_soak.py --url http://127.0.0.1:11438 \
  --cycles 64 --max-tokens 16
```

Performance reports must include every field required by
`docs/benchmark-contract.md`; a single high-acceptance MTP prompt is diagnostic,
not a product-wide speed claim.
