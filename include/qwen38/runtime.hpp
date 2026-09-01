#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

namespace qwen38 {

enum class ModelState : std::uint8_t {
    unloaded,
    loading,
    ready,
    failed,
};

struct RuntimeSnapshot {
    ModelState model_state{ModelState::unloaded};
    std::string model_id;
    std::string model_path;
    std::string last_error;
    std::uint64_t requests_total{0};
    std::uint64_t requests_active{0};
    std::uint64_t requests_cancelled{0};
    std::uint64_t prompt_tokens_total{0};
    std::uint64_t generated_tokens_total{0};
    double uptime_seconds{0.0};
};

class RuntimeState final {
public:
    RuntimeState();

    [[nodiscard]] RuntimeSnapshot snapshot() const;
    [[nodiscard]] bool ready() const noexcept;
    void begin_loading(std::string model_path);
    void mark_ready(std::string model_id);
    void mark_failed(std::string message);
    void request_started() noexcept;
    void request_finished(
        std::uint64_t prompt_tokens,
        std::uint64_t generated_tokens,
        bool cancelled = false) noexcept;

private:
    const std::chrono::steady_clock::time_point started_at_;
    std::atomic<ModelState> model_state_{ModelState::unloaded};
    std::atomic<std::uint64_t> requests_total_{0};
    std::atomic<std::uint64_t> requests_active_{0};
    std::atomic<std::uint64_t> requests_cancelled_{0};
    std::atomic<std::uint64_t> prompt_tokens_total_{0};
    std::atomic<std::uint64_t> generated_tokens_total_{0};
    mutable std::mutex metadata_mutex_;
    std::string model_id_;
    std::string model_path_;
    std::string last_error_;
};

[[nodiscard]] std::string_view to_string(ModelState state) noexcept;

} // namespace qwen38
