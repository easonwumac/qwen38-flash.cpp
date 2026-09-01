#include "qwen38/decode_state_io.hpp"

#include <charconv>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace qwen38 {
namespace {

using Metadata = std::vector<std::pair<std::string, std::string>>;
using Arrays = std::vector<MlxSafetensors::NamedArray>;

void add_metadata(Metadata& metadata, std::string key, const std::size_t value) {
    metadata.emplace_back(std::move(key), std::to_string(value));
}

void add_metadata(Metadata& metadata, std::string key, const bool value) {
    metadata.emplace_back(std::move(key), value ? "1" : "0");
}

std::string required_metadata(const MlxSafetensors& file, const std::string& key) {
    const std::optional<std::string> value = file.metadata(key);
    if (!value.has_value()) throw std::runtime_error("prefix state metadata missing: " + key);
    return *value;
}

std::size_t parse_size(const MlxSafetensors& file, const std::string& key) {
    const std::string text = required_metadata(file, key);
    std::size_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
        throw std::runtime_error("invalid prefix state metadata: " + key);
    }
    return value;
}

bool parse_bool(const MlxSafetensors& file, const std::string& key) {
    const std::string text = required_metadata(file, key);
    if (text == "0") return false;
    if (text == "1") return true;
    throw std::runtime_error("invalid prefix state boolean: " + key);
}

void append_decoder(
    const std::string& prefix,
    const DecoderLayerState& state,
    Arrays& arrays,
    Metadata& metadata) {
    add_metadata(metadata, prefix + ".linear.initialized", state.linear_attention.initialized);
    if (state.linear_attention.initialized) {
        arrays.push_back({prefix + ".linear.convolution", &state.linear_attention.convolution});
        arrays.push_back({prefix + ".linear.recurrent", &state.linear_attention.recurrent});
    }

    add_metadata(metadata, prefix + ".attention.token_count", state.full_attention.token_count);
    add_metadata(metadata, prefix + ".attention.position_base", state.full_attention.position_base);
    add_metadata(
        metadata, prefix + ".attention.qsa_pooled_count",
        state.full_attention.qsa_pooled_count);
    if (state.full_attention.token_count != 0) {
        arrays.push_back({prefix + ".attention.keys", &state.full_attention.keys});
        arrays.push_back({prefix + ".attention.values", &state.full_attention.values});
        arrays.push_back({prefix + ".attention.qsa_raw_keys", &state.full_attention.qsa_raw_keys});
        if (state.full_attention.qsa_pooled_count != 0) {
            arrays.push_back(
                {prefix + ".attention.qsa_pooled_keys", &state.full_attention.qsa_pooled_keys});
        }
    }

    add_metadata(metadata, prefix + ".ple.ngram.initialized", state.ple.ngram.initialized);
    add_metadata(
        metadata, prefix + ".ple.ngram.previous0",
        static_cast<std::size_t>(state.ple.ngram.previous[0]));
    add_metadata(
        metadata, prefix + ".ple.ngram.previous1",
        static_cast<std::size_t>(state.ple.ngram.previous[1]));
    add_metadata(metadata, prefix + ".ple.ngram.segment_length", state.ple.ngram.segment_length);
    add_metadata(
        metadata, prefix + ".ple.convolution_initialized",
        state.ple.convolution_initialized);
    if (state.ple.convolution_initialized) {
        arrays.push_back({prefix + ".ple.convolution", &state.ple.convolution});
    }
}

DecoderLayerState load_decoder(const std::string& prefix, const MlxSafetensors& file) {
    DecoderLayerState state;
    state.linear_attention.initialized = parse_bool(file, prefix + ".linear.initialized");
    if (state.linear_attention.initialized) {
        state.linear_attention.convolution = file.tensor(prefix + ".linear.convolution");
        state.linear_attention.recurrent = file.tensor(prefix + ".linear.recurrent");
    }

    state.full_attention.token_count = parse_size(file, prefix + ".attention.token_count");
    state.full_attention.position_base = parse_size(file, prefix + ".attention.position_base");
    state.full_attention.qsa_pooled_count =
        parse_size(file, prefix + ".attention.qsa_pooled_count");
    if (state.full_attention.token_count != 0) {
        state.full_attention.keys = file.tensor(prefix + ".attention.keys");
        state.full_attention.values = file.tensor(prefix + ".attention.values");
        state.full_attention.qsa_raw_keys = file.tensor(prefix + ".attention.qsa_raw_keys");
        if (state.full_attention.qsa_pooled_count != 0) {
            state.full_attention.qsa_pooled_keys =
                file.tensor(prefix + ".attention.qsa_pooled_keys");
        }
    }

    state.ple.ngram.initialized = parse_bool(file, prefix + ".ple.ngram.initialized");
    const std::size_t previous0 = parse_size(file, prefix + ".ple.ngram.previous0");
    const std::size_t previous1 = parse_size(file, prefix + ".ple.ngram.previous1");
    if (previous0 > std::numeric_limits<std::uint32_t>::max() ||
        previous1 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("prefix state ngram token is out of range");
    }
    state.ple.ngram.previous = {
        static_cast<std::uint32_t>(previous0), static_cast<std::uint32_t>(previous1)};
    state.ple.ngram.segment_length =
        parse_size(file, prefix + ".ple.ngram.segment_length");
    state.ple.convolution_initialized =
        parse_bool(file, prefix + ".ple.convolution_initialized");
    if (state.ple.convolution_initialized) {
        state.ple.convolution = file.tensor(prefix + ".ple.convolution");
    }
    return state;
}

} // namespace

void save_prefix_state(
    const std::filesystem::path& path,
    const PersistedPrefixState& state) {
    Arrays arrays;
    Metadata metadata;
    metadata.emplace_back("format", "qwen38-prefix-state-v1");
    add_metadata(metadata, "target.layer_count", state.target.layers.size());
    add_metadata(metadata, "target.token_count", state.target.token_count);
    for (std::size_t index = 0; index < state.target.layers.size(); ++index) {
        append_decoder(
            "target.layer." + std::to_string(index), state.target.layers[index], arrays, metadata);
    }

    append_decoder("mtp.layer", state.mtp.layer, arrays, metadata);
    add_metadata(metadata, "mtp.row_count", state.mtp.row_count);
    add_metadata(metadata, "mtp.position_base.present", state.mtp.position_base.has_value());
    add_metadata(metadata, "mtp.position_base", state.mtp.position_base.value_or(0));
    add_metadata(
        metadata, "previous_target_stream.present",
        state.previous_target_stream.has_value());
    if (state.previous_target_stream.has_value()) {
        arrays.push_back({"previous_target_stream", &*state.previous_target_stream});
    }
    add_metadata(metadata, "pending_mtp.count", state.pending_mtp_streams.size());
    if (state.pending_mtp_streams.size() != state.pending_mtp_tokens.size()) {
        throw std::runtime_error("pending MTP stream/token count mismatch");
    }
    for (std::size_t index = 0; index < state.pending_mtp_streams.size(); ++index) {
        arrays.push_back(
            {"pending_mtp.stream." + std::to_string(index), &state.pending_mtp_streams[index]});
        add_metadata(
            metadata, "pending_mtp.token." + std::to_string(index),
            static_cast<std::size_t>(state.pending_mtp_tokens[index]));
    }
    const std::size_t profitability = !state.mtp_profitable.has_value()
        ? 2
        : (*state.mtp_profitable ? 1 : 0);
    add_metadata(metadata, "mtp.profitable", profitability);
    add_metadata(
        metadata, "mtp.profitability_token.present",
        state.mtp_profitability_current_token.has_value());
    add_metadata(
        metadata, "mtp.profitability_token",
        static_cast<std::size_t>(state.mtp_profitability_current_token.value_or(0)));
    add_metadata(
        metadata, "mtp.cumulative_profitability_keep",
        state.mtp_cumulative_profitability_keep);
    MlxSafetensors::save(path, arrays, metadata);
}

PersistedPrefixState load_prefix_state(
    const std::filesystem::path& path,
    const std::size_t expected_layer_count) {
    MlxSafetensors file(path);
    if (required_metadata(file, "format") != "qwen38-prefix-state-v1") {
        throw std::runtime_error("unsupported prefix state format");
    }
    const std::size_t layer_count = parse_size(file, "target.layer_count");
    if (layer_count != expected_layer_count) {
        throw std::runtime_error("prefix state layer count does not match model");
    }
    PersistedPrefixState state(layer_count);
    state.target.token_count = parse_size(file, "target.token_count");
    for (std::size_t index = 0; index < layer_count; ++index) {
        state.target.layers[index] =
            load_decoder("target.layer." + std::to_string(index), file);
    }
    state.mtp.layer = load_decoder("mtp.layer", file);
    state.mtp.row_count = parse_size(file, "mtp.row_count");
    if (parse_bool(file, "mtp.position_base.present")) {
        state.mtp.position_base = parse_size(file, "mtp.position_base");
    }
    if (parse_bool(file, "previous_target_stream.present")) {
        state.previous_target_stream = file.tensor("previous_target_stream");
    }
    const std::size_t pending_count = parse_size(file, "pending_mtp.count");
    state.pending_mtp_streams.reserve(pending_count);
    state.pending_mtp_tokens.reserve(pending_count);
    for (std::size_t index = 0; index < pending_count; ++index) {
        state.pending_mtp_streams.push_back(
            file.tensor("pending_mtp.stream." + std::to_string(index)));
        const std::size_t token =
            parse_size(file, "pending_mtp.token." + std::to_string(index));
        if (token > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("pending MTP token is out of range");
        }
        state.pending_mtp_tokens.push_back(static_cast<std::uint32_t>(token));
    }
    const std::size_t profitability = parse_size(file, "mtp.profitable");
    if (profitability < 2) state.mtp_profitable = profitability == 1;
    if (profitability > 2) throw std::runtime_error("invalid MTP profitability state");
    if (parse_bool(file, "mtp.profitability_token.present")) {
        const std::size_t token = parse_size(file, "mtp.profitability_token");
        if (token > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("MTP profitability token is out of range");
        }
        state.mtp_profitability_current_token = static_cast<std::uint32_t>(token);
    }
    state.mtp_cumulative_profitability_keep =
        parse_bool(file, "mtp.cumulative_profitability_keep");
    return state;
}

} // namespace qwen38
