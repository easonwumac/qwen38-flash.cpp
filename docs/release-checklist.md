# Release checklist

This is a template for future runtime changes, not a completed test report.
The documentation-only daily-use closeout and the checks actually rerun are in
[2026-09-19 daily baseline](daily-use-2026-09-19.md). Performance stretch targets
remain deferred; accepting daily use does not mark them passed.

- [ ] Source worktree and exact commit/dirty state recorded.
- [ ] Clean model-free and MLX/tokenizer Release builds pass.
- [ ] CTest and Python developer tests pass in a documented MLX environment.
- [ ] Real-checkpoint tokenizer, component, full-model, API, tool, streaming,
      cancellation, invalid-request recovery, automatic-MTP and target-only smokes pass.
- [ ] Greedy fixtures and the retained reference quality suite do not regress.
- [ ] Serial mixed-corpus p10/median show no regression against the frozen baseline.
- [ ] Automatic MTP quality and economics pass against the frozen baseline;
      report p10, median, acceptance and fallback overhead separately.
- [ ] Cold/warm PP and context sweep include retrieval correctness and the maximum
      feasible context with peak footprint and guard headroom.
- [ ] Cold/warm PP does not regress; 600 PP / 40 target-only / 60 MTP stretch
      targets and unqualified 128K are explicitly distinguished from release status.
- [ ] RAM next-turn reuse and SSD restart restore pass; cache clearing is verified.
- [ ] Mixed request/error/disconnect/cache soak shows no leak or stuck active state.
- [ ] One automatic configuration, resource limits, sidecar requirements,
      metrics, recovery, and rollback controls match the documentation.
- [ ] No model, credential, local absolute path, or benchmark secret is tracked.
- [ ] Benchmark report states hardware, OS/power/thermal conditions, checkpoint,
      quantization, corpus hash, context, sampling, MTP, cache, memory, and
      distribution statistics.
- [ ] Commit, push, tag, and deployment occur only after explicit approval.
