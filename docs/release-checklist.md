# Release checklist

- [ ] Source worktree and exact commit/dirty state recorded.
- [ ] Clean model-free and MLX/tokenizer Release builds pass.
- [ ] CTest and Python developer tests pass in a documented MLX environment.
- [ ] Real-checkpoint tokenizer, component, full-model, API, tool, streaming,
      cancellation, invalid-request recovery, and default-serial smokes pass.
- [ ] Greedy fixtures and the retained reference quality suite do not regress.
- [ ] Exact serial mixed-corpus p10/median meet the accepted release gate.
- [ ] Opt-in MTP meets both its median gate and p10-at-least-serial economics gate.
- [ ] Cold/warm PP and context sweep include retrieval correctness and the maximum
      feasible context with peak footprint and guard headroom.
- [ ] Warm exact PP is at least 600 tok/s through 8K; the 32K target and
      64K/128K/maximum-context degradation are reported separately.
- [ ] RAM next-turn reuse and SSD restart restore pass; cache clearing is verified.
- [ ] Mixed request/error/disconnect/cache soak shows no leak or stuck active state.
- [ ] Startup defaults, profiles, sidecar requirements, metrics, recovery, and
      rollback controls match the documentation.
- [ ] No model, credential, local absolute path, or benchmark secret is tracked.
- [ ] Benchmark report states hardware, OS/power/thermal conditions, checkpoint,
      quantization, corpus hash, context, sampling, MTP, cache, memory, and
      distribution statistics.
- [ ] Commit, push, tag, and deployment occur only after explicit approval.
