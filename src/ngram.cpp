#include "qwen38/ngram.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <unistd.h>

namespace qwen38 {
namespace {

constexpr std::uint64_t splitmix_gamma = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t splitmix_m1 = 0xBF58476D1CE4E5B9ULL;
constexpr std::uint64_t splitmix_m2 = 0x94D049BB133111EBULL;

std::uint64_t splitmix64(std::uint64_t value) {
    value += splitmix_gamma;
    value = (value ^ (value >> 30U)) * splitmix_m1;
    value = (value ^ (value >> 27U)) * splitmix_m2;
    return value ^ (value >> 31U);
}

bool is_prime(const std::uint64_t value) {
    if (value < 2) return false;
    if (value % 2 == 0) return value == 2;
    for (std::uint64_t divisor = 3; divisor <= value / divisor; divisor += 2) {
        if (value % divisor == 0) return false;
    }
    return true;
}

std::uint64_t nth_prime_after(const std::uint64_t start, const std::size_t count) {
    std::uint64_t value = start;
    for (std::size_t index = 0; index < count; ++index) {
        do {
            ++value;
        } while (!is_prime(value));
    }
    return value;
}

std::int64_t positive_remainder(const std::uint64_t bits, const std::int64_t divisor) {
    const std::int64_t value = std::bit_cast<std::int64_t>(bits);
    std::int64_t result = value % divisor;
    if (result < 0) result += divisor;
    return result;
}

template <typename Integer>
Integer read_integer(const std::byte* data) {
    Integer value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

float bf16_to_float(const std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

} // namespace

NgramHash::NgramHash(const ModelConfig& config) : eos_(config.end_of_sequence_token) {
    const std::uint64_t maximum = (std::uint64_t{1} << 63U) - 1U;
    const std::uint64_t half_bound = std::max<std::uint64_t>(
        1, (maximum / std::max<std::size_t>(config.vocabulary_size, 1)) / 2);
    for (std::size_t index = 0; index < multipliers_.size(); ++index) {
        const std::uint64_t value = config.ngram_seed + splitmix_gamma * (index + 1);
        multipliers_[index] = static_cast<std::int64_t>(
            2 * (splitmix64(value) % half_bound) + 1);
    }
    std::uint64_t total = 0;
    for (std::size_t index = 0; index < vocabulary_.size(); ++index) {
        const std::uint64_t size = nth_prime_after(
            config.ngram_vocabulary_base - 1, index + 1);
        vocabulary_[index] = static_cast<std::int64_t>(size);
        offsets_[index] = static_cast<std::int64_t>(total);
        total += size;
    }
    const std::uint64_t divisor = config.ngram_vocabulary_divisor;
    total_rows_ = ((total + divisor - 1) / divisor) * divisor;
}

std::array<std::int64_t, 16> NgramHash::row_ids(
    const std::uint32_t token,
    NgramState& state) const {
    if (!state.initialized) {
        state.previous = {eos_, eos_};
        state.initialized = true;
    }
    const std::uint32_t previous_one = state.segment_length >= 1 ? state.previous[1] : eos_;
    const std::uint32_t previous_two = state.segment_length >= 2 ? state.previous[0] : eos_;
    std::uint64_t mixed = static_cast<std::uint64_t>(token) *
        static_cast<std::uint64_t>(multipliers_[0]);
    mixed ^= static_cast<std::uint64_t>(previous_one) *
        static_cast<std::uint64_t>(multipliers_[1]);
    std::array<std::int64_t, 16> result{};
    for (std::size_t head = 0; head < 8; ++head) {
        result[head] = positive_remainder(mixed, vocabulary_[head]) + offsets_[head];
    }
    mixed ^= static_cast<std::uint64_t>(previous_two) *
        static_cast<std::uint64_t>(multipliers_[2]);
    for (std::size_t head = 8; head < 16; ++head) {
        result[head] = positive_remainder(mixed, vocabulary_[head]) + offsets_[head];
    }
    if (token == eos_) {
        state.previous = {eos_, eos_};
        state.segment_length = 0;
    } else {
        state.previous = {state.previous[1], token};
        state.segment_length = std::min<std::size_t>(2, state.segment_length + 1);
    }
    return result;
}

NgramTable::NgramTable(
    const std::filesystem::path& model_directory,
    const std::uint64_t rows,
    const bool prefer_aos)
    : rows_(rows) {
    const std::size_t row_bytes = packed_word_count_ * 4 + scale_count_ * 4;
    const std::filesystem::path aos_path = model_directory / "ngram_table.bin.aos";
    if (prefer_aos && std::filesystem::is_regular_file(aos_path) &&
        std::filesystem::file_size(aos_path) == rows_ * row_bytes) {
        aos_fd_ = ::open(aos_path.c_str(), O_RDONLY);
        if (aos_fd_ < 0) throw std::runtime_error("cannot open n-gram AoS table");
        return;
    }
    fallback_ = std::make_unique<SafetensorsFile>(model_directory / "ngram_table.bin");
    fallback_weight_ = fallback_->tensor("weight");
    fallback_scales_ = fallback_->tensor("scales");
    fallback_biases_ = fallback_->tensor("biases");
    const auto weight_shape = fallback_weight_.shape;
    const auto scales_shape = fallback_scales_.shape;
    const auto biases_shape = fallback_biases_.shape;
    if (weight_shape.size() != 2 || weight_shape[0] != rows_ ||
        weight_shape[1] != packed_word_count_ || scales_shape.size() != 2 ||
        scales_shape[0] != rows_ || scales_shape[1] != scale_count_ ||
        biases_shape.size() != 2 || biases_shape[0] != rows_ ||
        biases_shape[1] != scale_count_) {
        throw std::runtime_error("n-gram table geometry mismatch");
    }
}

NgramTable::~NgramTable() {
    if (aos_fd_ >= 0) static_cast<void>(::close(aos_fd_));
}

void NgramTable::decode_row(
    const std::span<const std::byte> packed,
    const std::span<float> output) const {
    const std::size_t weight_bytes = packed_word_count_ * 4;
    const std::size_t scale_bytes = scale_count_ * 2;
    if (packed.size() != weight_bytes + 2 * scale_bytes || output.size() != dimension_) {
        throw std::runtime_error("invalid n-gram row buffer");
    }
    const std::byte* scales = packed.data() + weight_bytes;
    const std::byte* biases = scales + scale_bytes;
    for (std::size_t index = 0; index < dimension_; ++index) {
        const std::uint32_t word = read_integer<std::uint32_t>(
            packed.data() + (index / 8) * 4);
        const std::uint32_t quantized = (word >> ((index % 8) * 4)) & 0xFU;
        const std::size_t group = index / group_size_;
        const float scale = bf16_to_float(read_integer<std::uint16_t>(scales + group * 2));
        const float bias = bf16_to_float(read_integer<std::uint16_t>(biases + group * 2));
        output[index] = static_cast<float>(quantized) * scale + bias;
    }
}

std::vector<float> NgramTable::gather(
    const std::span<const std::int64_t> row_ids) const {
    const std::size_t weight_bytes = packed_word_count_ * 4;
    const std::size_t scale_bytes = scale_count_ * 2;
    const std::size_t row_bytes = weight_bytes + 2 * scale_bytes;
    std::vector<float> result(row_ids.size() * dimension_);
    std::vector<std::byte> row(row_bytes);
    for (std::size_t index = 0; index < row_ids.size(); ++index) {
        if (row_ids[index] < 0 || static_cast<std::uint64_t>(row_ids[index]) >= rows_) {
            throw std::runtime_error("n-gram row index is out of range");
        }
        if (aos_fd_ >= 0) {
            const off_t offset = static_cast<off_t>(
                static_cast<std::uint64_t>(row_ids[index]) * row_bytes);
            const ssize_t count = ::pread(aos_fd_, row.data(), row.size(), offset);
            if (count != static_cast<ssize_t>(row.size())) {
                throw std::runtime_error("short read from n-gram AoS table");
            }
        } else {
            const std::size_t source_row = static_cast<std::size_t>(row_ids[index]);
            std::memcpy(row.data(),
                fallback_weight_.bytes.data() + source_row * weight_bytes, weight_bytes);
            std::memcpy(row.data() + weight_bytes,
                fallback_scales_.bytes.data() + source_row * scale_bytes, scale_bytes);
            std::memcpy(row.data() + weight_bytes + scale_bytes,
                fallback_biases_.bytes.data() + source_row * scale_bytes, scale_bytes);
        }
        decode_row(row, std::span<float>(result).subspan(index * dimension_, dimension_));
    }
    return result;
}

} // namespace qwen38
