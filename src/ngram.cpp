#include "qwen38/ngram.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <sys/mman.h>
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

float f16_to_float(const std::uint16_t value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
    const std::uint32_t exponent = (value >> 10U) & 0x1FU;
    const std::uint32_t fraction = value & 0x03FFU;
    std::uint32_t bits = 0;
    if (exponent == 0) {
        if (fraction == 0) {
            bits = sign;
        } else {
            std::uint32_t normalized = fraction;
            int shift = 0;
            while ((normalized & 0x0400U) == 0) {
                normalized <<= 1U;
                ++shift;
            }
            normalized &= 0x03FFU;
            bits = sign | (static_cast<std::uint32_t>(113 - shift) << 23U) |
                (normalized << 13U);
        }
    } else if (exponent == 0x1FU) {
        bits = sign | 0x7F800000U | (fraction << 13U);
    } else {
        bits = sign | ((exponent + 112U) << 23U) | (fraction << 13U);
    }
    return std::bit_cast<float>(bits);
}

std::size_t shard_number(const std::string& key) {
    const std::size_t separator = key.rfind('_');
    if (separator == std::string::npos || separator + 1 == key.size()) {
        throw std::runtime_error("invalid vector-quantized PLE shard key");
    }
    std::size_t result = 0;
    const char* begin = key.data() + separator + 1;
    const char* end = key.data() + key.size();
    const auto parsed = std::from_chars(begin, end, result);
    if (parsed.ec != std::errc{} || parsed.ptr != end) {
        throw std::runtime_error("invalid vector-quantized PLE shard number");
    }
    return result;
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
    const std::filesystem::path bf16_aos_path =
        model_directory / "ngram_table.bf16.aos";
    const std::uint64_t bf16_row_bytes = dimension_ * sizeof(std::uint16_t);
    if (prefer_aos && std::filesystem::is_regular_file(bf16_aos_path) &&
        std::filesystem::file_size(bf16_aos_path) == rows_ * bf16_row_bytes) {
        bf16_aos_fd_ = ::open(bf16_aos_path.c_str(), O_RDONLY);
        if (bf16_aos_fd_ < 0) {
            throw std::runtime_error("cannot open BF16 n-gram AoS table");
        }
        return;
    }
    constexpr std::size_t q8_bits = 8;
    const std::size_t q8_packed_word_count = dimension_ * q8_bits / 32;
    const std::size_t q8_row_bytes = q8_packed_word_count * 4 + scale_count_ * 4;
    const std::filesystem::path q8_aos_path =
        model_directory / "ngram_table.q8.aos";
    if (prefer_aos && std::filesystem::is_regular_file(q8_aos_path) &&
        std::filesystem::file_size(q8_aos_path) == rows_ * q8_row_bytes) {
        aos_fd_ = ::open(q8_aos_path.c_str(), O_RDONLY);
        if (aos_fd_ < 0) throw std::runtime_error("cannot open Q8 n-gram AoS table");
        bits_ = q8_bits;
        packed_word_count_ = q8_packed_word_count;
        return;
    }
    const std::size_t row_bytes = packed_word_count_ * 4 + scale_count_ * 4;
    const std::filesystem::path aos_path = model_directory / "ngram_table.bin.aos";
    if (prefer_aos && std::filesystem::is_regular_file(aos_path) &&
        std::filesystem::file_size(aos_path) == rows_ * row_bytes) {
        aos_fd_ = ::open(aos_path.c_str(), O_RDONLY);
        if (aos_fd_ < 0) throw std::runtime_error("cannot open n-gram AoS table");
        return;
    }
    const std::filesystem::path fallback_path = model_directory / "ngram_table.bin";
    if (!std::filesystem::is_regular_file(fallback_path)) {
        ModelManifest manifest = ModelManifest::load(model_directory);
        if (manifest.vector_quantized_ple().has_value()) {
            initialize_vector_quantized(std::move(manifest));
        } else {
            initialize_paired(std::move(manifest));
        }
        return;
    }
    fallback_ = std::make_unique<SafetensorsFile>(fallback_path);
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

void NgramTable::initialize_vector_quantized(ModelManifest manifest) {
    const VectorQuantizedPleSpec& spec = *manifest.vector_quantized_ple();
    if (spec.codebook_size != 256 || spec.vector_dimension != 8 ||
        spec.group_size != 32 || spec.row_bytes != dimension_ / spec.vector_dimension) {
        throw std::runtime_error("unsupported vector-quantized PLE table geometry");
    }
    std::vector<std::string> keys = spec.keys;
    std::sort(keys.begin(), keys.end(), [](const std::string& left, const std::string& right) {
        return shard_number(left) < shard_number(right);
    });
    for (std::size_t index = 0; index < keys.size(); ++index) {
        if (shard_number(keys[index]) != index) {
            throw std::runtime_error("vector-quantized PLE shard sequence is incomplete");
        }
    }

    vector_quantized_store_ = std::make_unique<TensorStore>(std::move(manifest));
    std::uint64_t logical_begin = 0;
    for (const std::string& base : keys) {
        TensorView codes = vector_quantized_store_->tensor(base + ".codes");
        TensorView codebook = vector_quantized_store_->tensor(base + ".codebook");
        TensorView scales = vector_quantized_store_->tensor(base + ".vq_scales");
        if (codes.dtype != "U8" || codebook.dtype != "F16" || scales.dtype != "F16" ||
            codes.shape.size() != 2 || codes.shape[1] != spec.row_bytes ||
            codebook.shape.size() != 2 ||
            codebook.shape[0] != spec.codebook_size ||
            codebook.shape[1] != spec.vector_dimension ||
            scales.shape.size() != 2 || scales.shape[0] != codes.shape[0] ||
            scales.shape[1] != dimension_ / spec.group_size) {
            throw std::runtime_error("vector-quantized PLE shard geometry mismatch: " + base);
        }
        const std::uint64_t logical_end = logical_begin + codes.shape[0];
        if (logical_end < logical_begin) {
            throw std::runtime_error("vector-quantized PLE row count overflows");
        }
        vector_quantized_shards_.push_back({
            .logical_begin = logical_begin,
            .logical_end = logical_end,
            .codes = codes,
            .codebook = codebook,
            .scales = scales,
        });
        logical_begin = logical_end;
    }
    if (vector_quantized_shards_.empty() || logical_begin != rows_) {
        throw std::runtime_error("vector-quantized PLE total row count mismatch");
    }
}

void NgramTable::initialize_paired(ModelManifest manifest) {
    const ModelConfig& config = manifest.config();
    pair_ = config.niwaki_ple_pair;
    bits_ = config.niwaki_ple_bits;
    group_size_ = config.niwaki_ple_group_size;
    if (pair_ < 2 || bits_ == 0 || group_size_ == 0) {
        throw std::runtime_error(
            "model has neither a legacy n-gram table nor Niwaki paired PLE metadata");
    }
    const std::size_t physical_dimension = dimension_ * pair_;
    if ((physical_dimension * bits_) % 32 != 0 ||
        physical_dimension % group_size_ != 0) {
        throw std::runtime_error("invalid Niwaki paired PLE quantization geometry");
    }
    packed_word_count_ = physical_dimension * bits_ / 32;
    scale_count_ = physical_dimension / group_size_;
    if (config.ple_layer_ids.size() != 1) {
        throw std::runtime_error("Niwaki paired PLE requires exactly one PLE layer");
    }
    const std::string prefix = "language_model.model.layers." +
        std::to_string(config.ple_layer_ids.front() - 1) +
        ".ple.ple_embedding.ngram_embedding.shards.";
    paired_store_ = std::make_unique<TensorStore>(std::move(manifest));
    std::uint64_t logical_begin = 0;
    for (std::size_t index = 0;; ++index) {
        const std::string base = prefix + std::to_string(index);
        if (!paired_store_->manifest().has_tensor(base + ".weight")) break;
        TensorView weight = paired_store_->tensor(base + ".weight");
        TensorView scales = paired_store_->tensor(base + ".scales");
        TensorView biases = paired_store_->tensor(base + ".biases");
        if (weight.dtype != "U32" || scales.dtype != "BF16" || biases.dtype != "BF16" ||
            weight.shape.size() != 2 || scales.shape.size() != 2 ||
            biases.shape.size() != 2 || weight.shape[1] != packed_word_count_ ||
            scales.shape[0] != weight.shape[0] || biases.shape[0] != weight.shape[0] ||
            scales.shape[1] != scale_count_ || biases.shape[1] != scale_count_) {
            throw std::runtime_error("Niwaki paired PLE shard geometry mismatch: " + base);
        }
        const std::uint64_t physical_rows = weight.shape[0];
        if (physical_rows > (std::numeric_limits<std::uint64_t>::max() - logical_begin) / pair_) {
            throw std::runtime_error("Niwaki paired PLE row count overflow");
        }
        const std::uint64_t logical_end = logical_begin + physical_rows * pair_;
        paired_shards_.push_back({
            .logical_begin = logical_begin,
            .logical_end = logical_end,
            .weight = weight,
            .scales = scales,
            .biases = biases,
        });
        logical_begin = logical_end;
    }
    if (paired_shards_.empty() || logical_begin != rows_) {
        throw std::runtime_error("Niwaki paired PLE total row count mismatch");
    }
}

NgramTable::~NgramTable() {
    if (aos_fd_ >= 0) static_cast<void>(::close(aos_fd_));
    if (bf16_aos_fd_ >= 0) static_cast<void>(::close(bf16_aos_fd_));
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
        const std::size_t values_per_word = 32 / bits_;
        const std::uint32_t word = read_integer<std::uint32_t>(
            packed.data() + (index / values_per_word) * 4);
        const std::uint32_t quantized =
            (word >> ((index % values_per_word) * bits_)) & ((1U << bits_) - 1U);
        const std::size_t group = index / group_size_;
        const float scale = bf16_to_float(read_integer<std::uint16_t>(scales + group * 2));
        const float bias = bf16_to_float(read_integer<std::uint16_t>(biases + group * 2));
        output[index] = static_cast<float>(quantized) * scale + bias;
    }
}

void NgramTable::decode_paired_row(
    const PairedShard& shard,
    const std::uint64_t logical_row,
    const std::span<float> output) const {
    if (output.size() != dimension_ || logical_row < shard.logical_begin ||
        logical_row >= shard.logical_end) {
        throw std::runtime_error("invalid Niwaki paired PLE row");
    }
    const std::uint64_t within = logical_row - shard.logical_begin;
    const std::size_t physical_row = static_cast<std::size_t>(within / pair_);
    const std::size_t piece = static_cast<std::size_t>(within % pair_);
    const std::size_t weight_bytes = packed_word_count_ * sizeof(std::uint32_t);
    const std::size_t scale_bytes = scale_count_ * sizeof(std::uint16_t);
    const std::byte* weights = shard.weight.bytes.data() + physical_row * weight_bytes;
    const std::byte* scales = shard.scales.bytes.data() + physical_row * scale_bytes;
    const std::byte* biases = shard.biases.bytes.data() + physical_row * scale_bytes;
    const std::size_t values_per_word = 32 / bits_;
    for (std::size_t index = 0; index < dimension_; ++index) {
        const std::size_t column = piece * dimension_ + index;
        const std::uint32_t word = read_integer<std::uint32_t>(
            weights + (column / values_per_word) * sizeof(std::uint32_t));
        const std::uint32_t quantized =
            (word >> ((column % values_per_word) * bits_)) & ((1U << bits_) - 1U);
        const std::size_t group = column / group_size_;
        const float scale = bf16_to_float(read_integer<std::uint16_t>(scales + group * 2));
        const float bias = bf16_to_float(read_integer<std::uint16_t>(biases + group * 2));
        output[index] = static_cast<float>(quantized) * scale + bias;
    }
}

void NgramTable::decode_vector_quantized_row(
    const VectorQuantizedShard& shard,
    const std::uint64_t logical_row,
    const std::span<float> output) const {
    if (output.size() != dimension_ || logical_row < shard.logical_begin ||
        logical_row >= shard.logical_end) {
        throw std::runtime_error("invalid vector-quantized PLE row");
    }
    const std::size_t row = static_cast<std::size_t>(logical_row - shard.logical_begin);
    const std::size_t codes_per_row = dimension_ / 8;
    const std::size_t scales_per_row = dimension_ / 32;
    const std::byte* codes = shard.codes.bytes.data() + row * codes_per_row;
    const std::byte* scales = shard.scales.bytes.data() +
        row * scales_per_row * sizeof(std::uint16_t);
    for (std::size_t code_index = 0; code_index < codes_per_row; ++code_index) {
        const std::uint8_t code = read_integer<std::uint8_t>(codes + code_index);
        const float scale = f16_to_float(read_integer<std::uint16_t>(
            scales + (code_index / 4) * sizeof(std::uint16_t)));
        const std::byte* vector = shard.codebook.bytes.data() +
            (static_cast<std::size_t>(code) * 8) * sizeof(std::uint16_t);
        for (std::size_t element = 0; element < 8; ++element) {
            output[code_index * 8 + element] = scale * f16_to_float(
                read_integer<std::uint16_t>(vector + element * sizeof(std::uint16_t)));
        }
    }
}

std::vector<float> NgramTable::gather(
    const std::span<const std::int64_t> row_ids) const {
    const std::size_t weight_bytes = packed_word_count_ * 4;
    const std::size_t scale_bytes = scale_count_ * 2;
    const std::size_t row_bytes = weight_bytes + 2 * scale_bytes;
    std::vector<float> result(row_ids.size() * dimension_);
    std::vector<std::byte> row(row_bytes);
    std::vector<std::uint16_t> bf16_row(dimension_);
    std::vector<const VectorQuantizedShard*> vector_locations;
    if (vector_quantized_store_ != nullptr) {
        vector_locations.reserve(row_ids.size());
        std::unordered_set<std::uintptr_t> advised_pages;
        const long raw_page_size = ::sysconf(_SC_PAGESIZE);
        const std::size_t page_size = raw_page_size > 0
            ? static_cast<std::size_t>(raw_page_size) : 4096;
        const auto advise = [&](const std::byte* address) {
            const std::uintptr_t page =
                reinterpret_cast<std::uintptr_t>(address) & ~(page_size - 1);
            if (advised_pages.insert(page).second) {
                static_cast<void>(::madvise(
                    reinterpret_cast<void*>(page), page_size, MADV_WILLNEED));
            }
        };
        for (const std::int64_t row_id : row_ids) {
            if (row_id < 0 || static_cast<std::uint64_t>(row_id) >= rows_) {
                throw std::runtime_error("n-gram row index is out of range");
            }
            const std::uint64_t logical_row = static_cast<std::uint64_t>(row_id);
            const auto shard = std::upper_bound(
                vector_quantized_shards_.begin(), vector_quantized_shards_.end(), logical_row,
                [](const std::uint64_t value, const VectorQuantizedShard& candidate) {
                    return value < candidate.logical_end;
                });
            if (shard == vector_quantized_shards_.end() || logical_row < shard->logical_begin) {
                throw std::runtime_error("vector-quantized PLE shard lookup failed");
            }
            vector_locations.push_back(&*shard);
            const std::size_t within =
                static_cast<std::size_t>(logical_row - shard->logical_begin);
            advise(shard->codes.bytes.data() + within * (dimension_ / 8));
            advise(shard->scales.bytes.data() +
                within * (dimension_ / 32) * sizeof(std::uint16_t));
        }
    }
    for (std::size_t index = 0; index < row_ids.size(); ++index) {
        if (row_ids[index] < 0 || static_cast<std::uint64_t>(row_ids[index]) >= rows_) {
            throw std::runtime_error("n-gram row index is out of range");
        }
        if (paired_store_ != nullptr) {
            const std::uint64_t logical_row = static_cast<std::uint64_t>(row_ids[index]);
            const auto shard = std::upper_bound(
                paired_shards_.begin(), paired_shards_.end(), logical_row,
                [](const std::uint64_t row_id, const PairedShard& candidate) {
                    return row_id < candidate.logical_end;
                });
            if (shard == paired_shards_.end() || logical_row < shard->logical_begin) {
                throw std::runtime_error("Niwaki paired PLE shard lookup failed");
            }
            decode_paired_row(
                *shard, logical_row,
                std::span<float>(result).subspan(index * dimension_, dimension_));
            continue;
        }
        if (vector_quantized_store_ != nullptr) {
            const std::uint64_t logical_row = static_cast<std::uint64_t>(row_ids[index]);
            decode_vector_quantized_row(
                *vector_locations[index], logical_row,
                std::span<float>(result).subspan(index * dimension_, dimension_));
            continue;
        }
        if (bf16_aos_fd_ >= 0) {
            const std::size_t bf16_bytes = dimension_ * sizeof(std::uint16_t);
            const off_t offset = static_cast<off_t>(
                static_cast<std::uint64_t>(row_ids[index]) * bf16_bytes);
            const ssize_t count = ::pread(
                bf16_aos_fd_, bf16_row.data(), bf16_bytes, offset);
            if (count != static_cast<ssize_t>(bf16_bytes)) {
                throw std::runtime_error("short read from BF16 n-gram AoS table");
            }
            const std::size_t target = index * dimension_;
            for (std::size_t column = 0; column < dimension_; ++column) {
                result[target + column] = bf16_to_float(bf16_row[column]);
            }
            continue;
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
