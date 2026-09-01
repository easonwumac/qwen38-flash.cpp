#include "qwen38/runtime.hpp"

#include <utility>

namespace qwen38 {

RuntimeState::RuntimeState() : started_at_(std::chrono::steady_clock::now()) {}

RuntimeSnapshot RuntimeState::snapshot() const {
    RuntimeSnapshot result;
    result.model_state = model_state_.load(std::memory_order_acquire);
    result.requests_total = requests_total_.load(std::memory_order_relaxed);
    result.requests_active = requests_active_.load(std::memory_order_relaxed);
    result.requests_cancelled = requests_cancelled_.load(std::memory_order_relaxed);
    result.prompt_tokens_total = prompt_tokens_total_.load(std::memory_order_relaxed);
    result.generated_tokens_total = generated_tokens_total_.load(std::memory_order_relaxed);
    result.uptime_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started_at_).count();
    {
        std::scoped_lock lock(metadata_mutex_);
        result.model_id = model_id_;
        result.model_path = model_path_;
        result.last_error = last_error_;
    }
    return result;
}

bool RuntimeState::ready() const noexcept {
    return model_state_.load(std::memory_order_acquire) == ModelState::ready;
}

void RuntimeState::begin_loading(std::string model_path) {
    {
        std::scoped_lock lock(metadata_mutex_);
        model_path_ = std::move(model_path);
        model_id_.clear();
        last_error_.clear();
    }
    model_state_.store(ModelState::loading, std::memory_order_release);
}

void RuntimeState::mark_ready(std::string model_id) {
    {
        std::scoped_lock lock(metadata_mutex_);
        model_id_ = std::move(model_id);
        last_error_.clear();
    }
    model_state_.store(ModelState::ready, std::memory_order_release);
}

void RuntimeState::mark_failed(std::string message) {
    {
        std::scoped_lock lock(metadata_mutex_);
        last_error_ = std::move(message);
    }
    model_state_.store(ModelState::failed, std::memory_order_release);
}

void RuntimeState::request_started() noexcept {
    requests_total_.fetch_add(1, std::memory_order_relaxed);
    requests_active_.fetch_add(1, std::memory_order_relaxed);
}

void RuntimeState::request_finished(
    const std::uint64_t prompt_tokens,
    const std::uint64_t generated_tokens,
    const bool cancelled) noexcept {
    requests_active_.fetch_sub(1, std::memory_order_relaxed);
    if (cancelled) requests_cancelled_.fetch_add(1, std::memory_order_relaxed);
    prompt_tokens_total_.fetch_add(prompt_tokens, std::memory_order_relaxed);
    generated_tokens_total_.fetch_add(generated_tokens, std::memory_order_relaxed);
}

std::string_view to_string(const ModelState state) noexcept {
    switch (state) {
    case ModelState::unloaded: return "unloaded";
    case ModelState::loading: return "loading";
    case ModelState::ready: return "ready";
    case ModelState::failed: return "failed";
    }
    return "unknown";
}

} // namespace qwen38
