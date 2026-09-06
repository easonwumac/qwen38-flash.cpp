#include "test.hpp"

#include "qwen38/expert_cache.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

using Cache = qwen38::ExpertCache<int>;

Cache::Handle value(const int number) {
    return std::make_shared<const int>(number);
}

template <class Function>
void expects_runtime_error(Function&& function) {
    bool failed = false;
    try {
        function();
    } catch (const std::runtime_error&) {
        failed = true;
    }
    QWEN38_CHECK(failed);
}

} // namespace

void run_expert_cache_tests() {
    {
        Cache cache(2, 4096, 4);
        static_cast<void>(cache.acquire("hot", 1, [] { return value(1); }));
        static_cast<void>(cache.acquire("hot", 1, [] { return value(1); }));
        QWEN38_CHECK(cache.set_budget(0));
        QWEN38_CHECK(cache.stats().resident_bytes == 0 && cache.stats().history_entries == 1);
        QWEN38_CHECK(cache.set_budget(2));
        auto hot = cache.acquire("hot", 1, [] { return value(1); });
        const std::weak_ptr<const int> owner = hot;
        hot.reset();
        for (int i = 0; i < 16; ++i)
            static_cast<void>(cache.acquire("cold" + std::to_string(i), 1, [] { return value(2); }));
        QWEN38_CHECK(!owner.expired() && cache.stats().history_hits == 1);
        QWEN38_CHECK(cache.stats().history_entries <= 4);
        cache.clear();
        QWEN38_CHECK(owner.expired() && cache.stats().history_entries == 0);
    }
    {
        // A hot entry followed by a long one-off scan. Short aging destroys
        // the frequency signal before the next inference step can revisit it.
        for (const auto period : {64ULL, 4096ULL}) {
            Cache cache(100, period);
            auto hot = cache.acquire("hot", 1, [] { return std::make_shared<const int>(1); });
            const std::weak_ptr<const int> owner = hot;
            hot.reset();
            static_cast<void>(cache.acquire("hot", 1, [] { return std::make_shared<const int>(1); }));
            for (int i = 0; i < 480; ++i)
                static_cast<void>(cache.acquire("scan" + std::to_string(i), 1,
                    [] { return std::make_shared<const int>(2); }));
            QWEN38_CHECK(owner.expired() == (period == 64));
            for (int i = 480; i < 8192; ++i)
                static_cast<void>(cache.acquire("scan" + std::to_string(i), 1,
                    [] { return std::make_shared<const int>(2); }));
            QWEN38_CHECK(owner.expired()); // Old popularity must eventually expire.
        }
        bool refused = false;
        try { Cache invalid(100, 0); }
        catch (const std::invalid_argument&) { refused = true; }
        QWEN38_CHECK(refused);
    }
    {
        Cache cache(10);
        int loads = 0;
        auto first = cache.acquire("a", 4, [&] { ++loads; return value(7); });
        auto second = cache.acquire("a", 4, [&] { ++loads; return value(8); });
        QWEN38_CHECK(*first == 7 && *second == 7 && loads == 1);
        QWEN38_CHECK(cache.stats().hits == 1 && cache.stats().misses == 1);
        QWEN38_CHECK(cache.stats().hit_bytes == 4 && cache.stats().miss_bytes == 4);
        QWEN38_CHECK(cache.stats().resident_bytes == 4 && cache.stats().entries == 1);
    }

    {
        Cache cache(2);
        auto a = cache.acquire("a", 1, [] { return value(1); });
        auto b = cache.acquire("b", 1, [] { return value(2); });
        const std::weak_ptr<const int> a_owner = a;
        const std::weak_ptr<const int> b_owner = b;
        for (int i = 0; i < 12; ++i) {
            auto hit = cache.acquire("a", 1, [] { return value(99); });
            QWEN38_CHECK(*hit == 1);
        }
        a.reset();
        b.reset();
        auto c = cache.acquire("c", 1, [] { return value(3); });
        QWEN38_CHECK(*c == 3 && cache.stats().evictions == 1);
        QWEN38_CHECK(!a_owner.expired() && b_owner.expired());
        c.reset();

        // Repeated decay operations allow a once-hot entry to age out rather
        // than remain permanently protected by its historical frequency.
        for (int i = 0; i < 256; ++i) {
            auto hit = cache.acquire("c", 1, [] { return value(4); });
            QWEN38_CHECK(*hit == 3);
        }
        auto d = cache.acquire("d", 1, [] { return value(4); });
        QWEN38_CHECK(*d == 4);
        QWEN38_CHECK(cache.stats().evictions >= 2);
        QWEN38_CHECK(a_owner.expired());
    }

    {
        Cache cache(4);
        int loads = 0;
        expects_runtime_error([&] {
            static_cast<void>(cache.acquire("bad", 2, [&]() -> Cache::Handle {
                ++loads;
                throw std::runtime_error("loader failure");
            }));
        });
        QWEN38_CHECK(loads == 1 && cache.stats().load_failures == 1);
        QWEN38_CHECK(cache.stats().resident_bytes == 0 && cache.stats().entries == 0);

        expects_runtime_error([&] {
            static_cast<void>(cache.acquire("null", 2, [] { return Cache::Handle{}; }));
        });
        QWEN38_CHECK(cache.stats().load_failures == 2);
        QWEN38_CHECK(cache.stats().resident_bytes == 0 && cache.stats().entries == 0);
        auto recovered = cache.acquire("recovered", 4, [] { return value(3); });
        QWEN38_CHECK(*recovered == 3 && cache.stats().resident_bytes == 4);
    }

    {
        Cache cache(4);
        bool called = false;
        expects_runtime_error([&] {
            static_cast<void>(cache.acquire("", 1, [&] { called = true; return value(1); }));
        });
        expects_runtime_error([&] {
            static_cast<void>(cache.acquire("zero", 0, [&] { called = true; return value(1); }));
        });
        expects_runtime_error([&] {
            static_cast<void>(cache.acquire("large", 5, [&] { called = true; return value(1); }));
        });
        QWEN38_CHECK(!called && cache.stats().refusals == 3);

        auto a = cache.acquire("a", 4, [] { return value(1); });
        int mismatch_loads = 0;
        expects_runtime_error([&] {
            static_cast<void>(cache.acquire("a", 3, [&] { ++mismatch_loads; return value(2); }));
        });
        QWEN38_CHECK(mismatch_loads == 0 && cache.stats().refusals == 4);
        QWEN38_CHECK(cache.stats().resident_bytes == 4 && cache.stats().entries == 1);
        a.reset();
    }

    {
        Cache cache(1);
        auto pinned = cache.acquire("a", 1, [] { return value(1); });
        int refused_loads = 0;
        expects_runtime_error([&] {
            static_cast<void>(cache.acquire("b", 1, [&] { ++refused_loads; return value(2); }));
        });
        QWEN38_CHECK(refused_loads == 0 && cache.stats().refusals == 1);
        pinned.reset();
        auto b = cache.acquire("b", 1, [] { return value(2); });
        QWEN38_CHECK(*b == 2 && cache.stats().evictions == 1);
    }

    {
        Cache cache(2);
        auto a = cache.acquire("a", 1, [] { return value(1); });
        auto b = cache.acquire("b", 1, [] { return value(2); });
        QWEN38_CHECK(!cache.set_budget(1));
        QWEN38_CHECK(cache.budget() == 1 && cache.stats().resident_bytes == 2);
        cache.clear();
        QWEN38_CHECK(cache.stats().resident_bytes == 2 && cache.stats().entries == 2);
        b.reset();
        QWEN38_CHECK(cache.trim());
        QWEN38_CHECK(cache.stats().resident_bytes == 1 && cache.stats().entries == 1);
        a.reset();
        cache.clear();
        QWEN38_CHECK(cache.stats().resident_bytes == 0 && cache.stats().entries == 0);
    }

    {
        Cache cache(4);
        bool reentered = false;
        auto outer = cache.acquire("outer", 2, [&]() {
            try {
                static_cast<void>(cache.acquire("inner", 1, [] { return value(2); }));
            } catch (const std::runtime_error&) {
                reentered = true;
            }
            return value(1);
        });
        QWEN38_CHECK(reentered && *outer == 1);
        QWEN38_CHECK(cache.stats().refusals == 1 && cache.stats().load_failures == 0);
    }

    {
        Cache cache(2);
        auto older = cache.acquire("older", 1, [] { return value(1); });
        auto newer = cache.acquire("newer", 1, [] { return value(2); });
        const std::weak_ptr<const int> old_owner = older;
        const std::weak_ptr<const int> new_owner = newer;
        older.reset();
        newer.reset();
        auto third = cache.acquire("third", 1, [] { return value(3); });
        QWEN38_CHECK(old_owner.expired() && !new_owner.expired());
    }

    {
        Cache cache(2);
        auto outer = cache.acquire("outer", 2, [&] {
            expects_runtime_error([&] { cache.set_budget(0); });
            expects_runtime_error([&] { cache.trim(); });
            expects_runtime_error([&] { cache.clear(); });
            return value(5);
        });
        QWEN38_CHECK(*outer == 5 && cache.budget() == 2);
    }

    {
        Cache cache(std::numeric_limits<std::size_t>::max());
        auto huge = cache.acquire("huge", std::numeric_limits<std::size_t>::max(), [] { return value(1); });
        QWEN38_CHECK(*huge == 1 && cache.stats().resident_bytes == std::numeric_limits<std::size_t>::max());
        expects_runtime_error([&] {
            static_cast<void>(cache.acquire("overflow", 1, [] { return value(2); }));
        });
        QWEN38_CHECK(cache.stats().refusals == 1);
        huge.reset();
    }
}
