#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace qwen38 {

struct ProcessMemoryRange {
    std::size_t steady_bytes;
    std::size_t ceiling_bytes;
};

struct ExpertBudgetRequest {
    std::size_t non_expert_bytes{0};
    std::size_t transient_bytes{0};
    std::size_t safety_reserve_bytes{0};
    std::size_t minimum_expert_bytes{0};
    // An absolute expert-tier allowance imposed by the caller's system-pressure
    // policy, not free system RAM. Convert observations before calling.
    std::size_t pressure_expert_limit{std::numeric_limits<std::size_t>::max()};
};

enum class BudgetFailure { none, base_exceeds_ceiling, working_set_exceeds_allowance };

struct ExpertBudgetPlan {
    bool feasible{false};
    bool steady_feasible{false};
    std::size_t accounted_overhead_bytes{0};
    std::size_t steady_expert_bytes{0};
    std::size_t burst_expert_bytes{0};
    BudgetFailure failure{BudgetFailure::none};
};

// Pure accounting plan, NOT permission to allocate. The runtime must also
// retire/trim leases, fence GPU work and recheck actual process/system headroom.
// In particular, do not estimate non-expert memory by blindly subtracting an
// MLX logical cache counter from RSS/physical footprint: those measures differ.
inline ExpertBudgetPlan plan_expert_budget(
    const ProcessMemoryRange range, const ExpertBudgetRequest request) {
    if (range.ceiling_bytes == 0 || range.steady_bytes > range.ceiling_bytes) {
        throw std::invalid_argument("invalid steady/ceiling memory range");
    }
    ExpertBudgetPlan plan;
    for (const std::size_t bytes : {request.non_expert_bytes, request.transient_bytes,
                                    request.safety_reserve_bytes}) {
        // Subtraction-first checks avoid overflow even at SIZE_MAX budgets.
        if (bytes > range.ceiling_bytes - plan.accounted_overhead_bytes) {
            plan.failure = BudgetFailure::base_exceeds_ceiling;
            return plan;
        }
        plan.accounted_overhead_bytes += bytes;
    }
    plan.steady_expert_bytes = range.steady_bytes > plan.accounted_overhead_bytes
        ? range.steady_bytes - plan.accounted_overhead_bytes : 0;
    plan.burst_expert_bytes = range.ceiling_bytes - plan.accounted_overhead_bytes;
    plan.steady_expert_bytes = std::min(plan.steady_expert_bytes, request.pressure_expert_limit);
    plan.burst_expert_bytes = std::min(plan.burst_expert_bytes, request.pressure_expert_limit);
    plan.feasible = request.minimum_expert_bytes <= plan.burst_expert_bytes;
    plan.steady_feasible = plan.accounted_overhead_bytes <= range.steady_bytes &&
        request.minimum_expert_bytes <= plan.steady_expert_bytes;
    if (!plan.feasible) plan.failure = BudgetFailure::working_set_exceeds_allowance;
    return plan;
}

} // namespace qwen38
