#include "qwen38/history_draft.hpp"

#include <algorithm>
#include <stdexcept>

namespace qwen38 {
namespace {

constexpr std::uint64_t fnv_offset = 1469598103934665603ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;
constexpr std::size_t learned_probe_rounds = 8;
constexpr std::size_t history_probe_rounds = 2;

void validate_observation(const std::size_t proposed, const std::size_t accepted) {
    if (accepted > proposed) {
        throw std::runtime_error("draft acceptance exceeds proposed tokens");
    }
}

} // namespace

void HistoryDraftPolicy::observe_learned(
    const std::size_t proposed,
    const std::size_t accepted) {
    validate_observation(proposed, accepted);
    if (mode_ != HistoryDraftMode::adaptive || active_ || rejected_ || proposed == 0) {
        return;
    }
    ++learned_rounds_;
    learned_proposed_ += proposed;
    learned_accepted_ += accepted;
    if (learned_rounds_ < learned_probe_rounds) return;
    if (learned_accepted_ * 2 < learned_proposed_) {
        active_ = true;
        ++activations_;
    }
    learned_rounds_ = 0;
    learned_proposed_ = 0;
    learned_accepted_ = 0;
}

void HistoryDraftPolicy::observe_history(
    const std::size_t proposed,
    const std::size_t accepted) {
    validate_observation(proposed, accepted);
    if (mode_ != HistoryDraftMode::adaptive || !active_ || proposed == 0) return;
    ++history_rounds_;
    history_proposed_ += proposed;
    history_accepted_ += accepted;
    if (history_rounds_ < history_probe_rounds) return;
    if (history_accepted_ * 2 < history_proposed_) {
        active_ = false;
        rejected_ = true;
        ++deactivations_;
    }
    history_rounds_ = 0;
    history_proposed_ = 0;
    history_accepted_ = 0;
}

HistoryDraftCache::HistoryDraftCache(
    const std::size_t minimum_order,
    const std::size_t maximum_order)
    : minimum_order_(minimum_order), maximum_order_(maximum_order) {
    if (minimum_order_ == 0 || minimum_order_ > maximum_order_) {
        throw std::runtime_error("invalid history draft order range");
    }
}

std::uint64_t HistoryDraftCache::key(
    const std::size_t start,
    const std::size_t order) const noexcept {
    std::uint64_t hash = fnv_offset ^ static_cast<std::uint64_t>(order);
    for (std::size_t index = 0; index < order; ++index) {
        hash ^= static_cast<std::uint64_t>(history_[start + index]);
        hash *= fnv_prime;
    }
    return hash;
}

void HistoryDraftCache::append(const std::uint32_t token) {
    history_.push_back(token);
    // The token immediately before the new one now has a known continuation.
    // Publish every suffix ending there, but never the current terminal suffix.
    for (std::size_t order = minimum_order_; order <= maximum_order_; ++order) {
        if (history_.size() < order + 1) break;
        const std::size_t start = history_.size() - order - 1;
        latest_.insert_or_assign(key(start, order), start);
    }
}

void HistoryDraftCache::append(const std::span<const std::uint32_t> tokens) {
    history_.reserve(history_.size() + tokens.size());
    latest_.reserve(latest_.size() + tokens.size() * (maximum_order_ - minimum_order_ + 1));
    for (const std::uint32_t token : tokens) append(token);
}

bool HistoryDraftCache::suffix_matches(
    const std::size_t start,
    const std::size_t order) const noexcept {
    if (start + order >= history_.size() || order > history_.size()) return false;
    const std::size_t suffix = history_.size() - order;
    return std::equal(
        history_.begin() + static_cast<std::ptrdiff_t>(start),
        history_.begin() + static_cast<std::ptrdiff_t>(start + order),
        history_.begin() + static_cast<std::ptrdiff_t>(suffix));
}

std::vector<std::uint32_t> HistoryDraftCache::propose(
    const std::size_t maximum_tokens) const {
    if (maximum_tokens == 0) return {};
    for (std::size_t order = maximum_order_; order >= minimum_order_; --order) {
        if (history_.size() >= order) {
            const std::size_t suffix = history_.size() - order;
            const auto found = latest_.find(key(suffix, order));
            if (found != latest_.end() && suffix_matches(found->second, order)) {
                const std::size_t continuation = found->second + order;
                const std::size_t count = std::min(
                    maximum_tokens, history_.size() - continuation);
                return std::vector<std::uint32_t>(
                    history_.begin() + static_cast<std::ptrdiff_t>(continuation),
                    history_.begin() + static_cast<std::ptrdiff_t>(continuation + count));
            }
        }
        if (order == minimum_order_) break;
    }
    return {};
}

ContextCopyCache::ContextCopyCache(
    const std::span<const std::uint32_t> prompt,
    const std::size_t minimum_order,
    const std::size_t maximum_order)
    : minimum_order_(minimum_order),
      maximum_order_(maximum_order),
      prompt_(prompt.begin(), prompt.end()) {
    if (minimum_order_ == 0 || minimum_order_ > maximum_order_) {
        throw std::runtime_error("invalid context-copy order range");
    }
    continuations_.reserve(prompt_.size());
    for (std::size_t end = minimum_order_; end < prompt_.size(); ++end) {
        const auto suffix = std::span<const std::uint32_t>(prompt_).subspan(
            end - minimum_order_, minimum_order_);
        continuations_[key(suffix)].push_back(end);
    }
}

std::uint64_t ContextCopyCache::key(
    const std::span<const std::uint32_t> tokens) noexcept {
    std::uint64_t hash = fnv_offset ^ static_cast<std::uint64_t>(tokens.size());
    for (const std::uint32_t token : tokens) {
        hash ^= static_cast<std::uint64_t>(token);
        hash *= fnv_prime;
    }
    return hash;
}

ContextCopyProposal ContextCopyCache::propose(
    const std::span<const std::uint32_t> history,
    const std::size_t maximum_tokens) const {
    if (maximum_tokens == 0 || history.size() < minimum_order_) return {};
    const auto suffix = history.last(minimum_order_);
    const auto found = continuations_.find(key(suffix));
    if (found == continuations_.end()) return {};

    std::size_t best_position = prompt_.size();
    std::size_t best_extension = 0;
    const std::size_t extension_cap = maximum_order_ - minimum_order_;
    const std::size_t candidate_begin = found->second.size() > 32
        ? found->second.size() - 32
        : 0;
    for (std::size_t candidate_index = found->second.size();
         candidate_index-- > candidate_begin;) {
        const std::size_t position = found->second[candidate_index];
        if (position >= prompt_.size()) continue;
        if (!std::equal(
                prompt_.begin() + static_cast<std::ptrdiff_t>(position - minimum_order_),
                prompt_.begin() + static_cast<std::ptrdiff_t>(position),
                suffix.begin())) {
            continue; // hash collision
        }
        std::size_t extension = 0;
        while (extension < extension_cap &&
               position > minimum_order_ + extension &&
               history.size() > minimum_order_ + extension &&
               prompt_[position - minimum_order_ - extension - 1] ==
                   history[history.size() - minimum_order_ - extension - 1]) {
            ++extension;
        }
        if (best_position == prompt_.size() || extension > best_extension) {
            best_position = position;
            best_extension = extension;
            if (extension == extension_cap) break;
        }
    }
    if (best_position == prompt_.size()) return {};
    const std::size_t count = std::min(maximum_tokens, prompt_.size() - best_position);
    return {
        .tokens = std::vector<std::uint32_t>(
            prompt_.begin() + static_cast<std::ptrdiff_t>(best_position),
            prompt_.begin() + static_cast<std::ptrdiff_t>(best_position + count)),
        .match_extension = best_extension,
    };
}

ContextCopyProposal ContextCopyCache::propose_completion(
    const std::span<const std::uint32_t> completion,
    const std::size_t maximum_tokens) const {
    const std::size_t history_size = prompt_.size() + completion.size();
    if (maximum_tokens == 0 || history_size < minimum_order_) return {};
    const auto history_token = [&](const std::size_t index) -> std::uint32_t {
        return index < prompt_.size()
            ? prompt_[index]
            : completion[index - prompt_.size()];
    };

    std::uint64_t suffix_key = fnv_offset ^ static_cast<std::uint64_t>(minimum_order_);
    for (std::size_t index = history_size - minimum_order_; index < history_size; ++index) {
        suffix_key ^= static_cast<std::uint64_t>(history_token(index));
        suffix_key *= fnv_prime;
    }
    const auto found = continuations_.find(suffix_key);
    if (found == continuations_.end()) return {};

    std::size_t best_position = prompt_.size();
    std::size_t best_extension = 0;
    const std::size_t extension_cap = maximum_order_ - minimum_order_;
    const std::size_t candidate_begin = found->second.size() > 32
        ? found->second.size() - 32
        : 0;
    for (std::size_t candidate_index = found->second.size();
         candidate_index-- > candidate_begin;) {
        const std::size_t position = found->second[candidate_index];
        if (position >= prompt_.size()) continue;
        bool exact = true;
        for (std::size_t offset = 0; offset < minimum_order_; ++offset) {
            if (prompt_[position - minimum_order_ + offset] !=
                history_token(history_size - minimum_order_ + offset)) {
                exact = false;
                break;
            }
        }
        if (!exact) continue;
        std::size_t extension = 0;
        while (extension < extension_cap &&
               position > minimum_order_ + extension &&
               history_size > minimum_order_ + extension &&
               prompt_[position - minimum_order_ - extension - 1] ==
                   history_token(history_size - minimum_order_ - extension - 1)) {
            ++extension;
        }
        if (best_position == prompt_.size() || extension > best_extension) {
            best_position = position;
            best_extension = extension;
            if (extension == extension_cap) break;
        }
    }
    if (best_position == prompt_.size()) return {};
    const std::size_t count = std::min(maximum_tokens, prompt_.size() - best_position);
    return {
        .tokens = std::vector<std::uint32_t>(
            prompt_.begin() + static_cast<std::ptrdiff_t>(best_position),
            prompt_.begin() + static_cast<std::ptrdiff_t>(best_position + count)),
        .match_extension = best_extension,
    };
}

} // namespace qwen38
