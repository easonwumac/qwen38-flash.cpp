#include "qwen38/memory_budget.hpp"
#include "qwen38/expert_cache.hpp"
#include "test.hpp"

#include <array>
#include <limits>

void run_memory_budget_tests() {
    using namespace qwen38;
    constexpr std::size_t gib = 1024ULL * 1024ULL * 1024ULL;
    // Arithmetic tests only: these do not instantiate a model at these sizes.
    for (const auto range : std::array<ProcessMemoryRange, 6>{
             {{8 * gib, 16 * gib}, {12 * gib, 20 * gib}, {16 * gib, 24 * gib},
              {20 * gib, 30 * gib}, {24 * gib, 32 * gib}, {28 * gib, 36 * gib}}}) {
        ExpertBudgetRequest request{.non_expert_bytes = 4 * gib,
            .transient_bytes = gib, .safety_reserve_bytes = 2 * gib,
            .minimum_expert_bytes = gib};
        const auto plan = plan_expert_budget(range, request);
        QWEN38_CHECK(plan.feasible && plan.failure == BudgetFailure::none);
        QWEN38_CHECK(plan.steady_feasible);
        QWEN38_CHECK(plan.accounted_overhead_bytes == 7 * gib);
        QWEN38_CHECK(plan.steady_expert_bytes == range.steady_bytes - 7 * gib);
        QWEN38_CHECK(plan.burst_expert_bytes == range.ceiling_bytes - 7 * gib);
        request.non_expert_bytes += 2 * gib; // KV/state growth shrinks the tier.
        const auto grown = plan_expert_budget(range, request);
        QWEN38_CHECK(grown.burst_expert_bytes + 2 * gib == plan.burst_expert_bytes);
        QWEN38_CHECK(grown.steady_expert_bytes <= plan.steady_expert_bytes);
        request.pressure_expert_limit = gib / 2;
        const auto pressured = plan_expert_budget(range, request);
        QWEN38_CHECK(!pressured.feasible);
        QWEN38_CHECK(pressured.failure == BudgetFailure::working_set_exceeds_allowance);
        QWEN38_CHECK(pressured.burst_expert_bytes == gib / 2);
    }
    {
        const auto burst = plan_expert_budget({8 * gib, 16 * gib},
            {.non_expert_bytes = 9 * gib, .minimum_expert_bytes = gib});
        QWEN38_CHECK(burst.feasible && burst.steady_expert_bytes == 0);
        QWEN38_CHECK(!burst.steady_feasible);
        QWEN38_CHECK(burst.burst_expert_bytes == 7 * gib);
        const auto impossible = plan_expert_budget({8 * gib, 16 * gib},
            {.non_expert_bytes = 16 * gib, .transient_bytes = 1});
        QWEN38_CHECK(!impossible.feasible && impossible.failure == BudgetFailure::base_exceeds_ceiling);
    }
    {
        const auto maximum = std::numeric_limits<std::size_t>::max();
        const auto overflow = plan_expert_budget({maximum, maximum},
            {.non_expert_bytes = maximum, .safety_reserve_bytes = 1});
        QWEN38_CHECK(!overflow.feasible);
        const auto exact = plan_expert_budget({maximum, maximum},
            {.non_expert_bytes = maximum - 1, .minimum_expert_bytes = 1});
        QWEN38_CHECK(exact.feasible && exact.burst_expert_bytes == 1);
    }
    {
        // Controller/cache contract in synthetic byte units, no large allocation.
        ExpertBudgetRequest request{.non_expert_bytes = 4, .transient_bytes = 2,
            .safety_reserve_bytes = 2, .minimum_expert_bytes = 4};
        auto plan = plan_expert_budget({16, 24}, request);
        ExpertCache<int> cache(plan.steady_expert_bytes);
        auto first = cache.acquire("first", 4, [] { return std::make_shared<const int>(1); });
        auto second = cache.acquire("second", 4, [] { return std::make_shared<const int>(2); });
        QWEN38_CHECK(cache.set_budget(plan.burst_expert_bytes));
        auto third = cache.acquire("third", 4, [] { return std::make_shared<const int>(3); });
        request.non_expert_bytes += 4; // Reserve context growth before allocation.
        plan = plan_expert_budget({16, 24}, request);
        QWEN38_CHECK(plan.feasible && plan.steady_expert_bytes == 4);
        QWEN38_CHECK(!cache.set_budget(plan.steady_expert_bytes));
        first.reset(); second.reset(); // Runtime must also fence GPU work first.
        QWEN38_CHECK(cache.trim() && cache.stats().resident_bytes == 4);
        request.pressure_expert_limit = 0; // Zero explicitly means no allowance.
        plan = plan_expert_budget({16, 24}, request);
        QWEN38_CHECK(!plan.feasible && !plan.steady_feasible);
        third.reset();
        QWEN38_CHECK(cache.set_budget(plan.burst_expert_bytes));
        QWEN38_CHECK(cache.stats().resident_bytes == 0);
    }
    for (const auto invalid : std::array<ProcessMemoryRange, 2>{{{0, 0}, {2, 1}}}) {
        bool refused = false;
        try { static_cast<void>(plan_expert_budget(invalid, {})); }
        catch (const std::invalid_argument&) { refused = true; }
        QWEN38_CHECK(refused);
    }
}
