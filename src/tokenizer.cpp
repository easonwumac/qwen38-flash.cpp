#include "qwen38/tokenizer.hpp"

#include "qwen38/json.hpp"

#include <utf8proc.h>

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace qwen38 {
namespace {

struct Codepoint {
    std::uint32_t value;
    std::size_t begin;
    std::size_t end;
    utf8proc_category_t category;
};

std::string read_text(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    std::ostringstream result;
    result << stream.rdbuf();
    return result.str();
}

std::string utf8(const std::uint32_t value) {
    std::array<utf8proc_uint8_t, 4> bytes{};
    const utf8proc_ssize_t size = utf8proc_encode_char(static_cast<utf8proc_int32_t>(value), bytes.data());
    if (size <= 0) throw std::runtime_error("cannot encode Unicode code point");
    return {reinterpret_cast<const char*>(bytes.data()), static_cast<std::size_t>(size)};
}

std::string normalize_nfc(const std::string_view input) {
    utf8proc_uint8_t* output = nullptr;
    const utf8proc_ssize_t size = utf8proc_map(
        reinterpret_cast<const utf8proc_uint8_t*>(input.data()),
        static_cast<utf8proc_ssize_t>(input.size()),
        &output,
        static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
    if (size < 0) {
        throw std::runtime_error(std::string("Unicode normalization failed: ") + utf8proc_errmsg(size));
    }
    std::string result(reinterpret_cast<const char*>(output), static_cast<std::size_t>(size));
    std::free(output);
    return result;
}

std::vector<Codepoint> codepoints(const std::string_view input) {
    std::vector<Codepoint> result;
    std::size_t cursor = 0;
    while (cursor < input.size()) {
        utf8proc_int32_t value = 0;
        const utf8proc_ssize_t size = utf8proc_iterate(
            reinterpret_cast<const utf8proc_uint8_t*>(input.data() + cursor),
            static_cast<utf8proc_ssize_t>(input.size() - cursor),
            &value);
        if (size <= 0) throw std::runtime_error("input contains invalid UTF-8");
        const std::size_t end = cursor + static_cast<std::size_t>(size);
        result.push_back({
            .value = static_cast<std::uint32_t>(value),
            .begin = cursor,
            .end = end,
            .category = utf8proc_category(value),
        });
        cursor = end;
    }
    return result;
}

bool is_letter(const Codepoint& cp) {
    return cp.category >= UTF8PROC_CATEGORY_LU && cp.category <= UTF8PROC_CATEGORY_LO;
}

bool is_mark(const Codepoint& cp) {
    return cp.category >= UTF8PROC_CATEGORY_MN && cp.category <= UTF8PROC_CATEGORY_ME;
}

bool is_number(const Codepoint& cp) {
    return cp.category >= UTF8PROC_CATEGORY_ND && cp.category <= UTF8PROC_CATEGORY_NO;
}

bool is_newline(const Codepoint& cp) { return cp.value == '\r' || cp.value == '\n'; }

bool is_space(const Codepoint& cp) {
    if (cp.category >= UTF8PROC_CATEGORY_ZS && cp.category <= UTF8PROC_CATEGORY_ZP) return true;
    switch (cp.value) {
    case 0x0009:
    case 0x000A:
    case 0x000B:
    case 0x000C:
    case 0x000D:
    case 0x0085:
        return true;
    default:
        return false;
    }
}

std::size_t match_contraction(const std::vector<Codepoint>& cps, const std::size_t at) {
    if (cps[at].value != '\'' || at + 1 >= cps.size()) return at;
    static constexpr std::array<std::string_view, 7> suffixes{"s", "t", "re", "ve", "m", "ll", "d"};
    for (const std::string_view suffix : suffixes) {
        if (at + 1 + suffix.size() > cps.size()) continue;
        bool equal = true;
        for (std::size_t i = 0; i < suffix.size(); ++i) {
            std::uint32_t actual = cps[at + 1 + i].value;
            if (actual >= 'A' && actual <= 'Z') actual += 'a' - 'A';
            if (actual != static_cast<unsigned char>(suffix[i])) {
                equal = false;
                break;
            }
        }
        if (equal) return at + 1 + suffix.size();
    }
    return at;
}

std::vector<std::string_view> pretokenize(const std::string_view input) {
    const auto cps = codepoints(input);
    std::vector<std::string_view> result;
    std::size_t at = 0;
    auto emit = [&](const std::size_t end) {
        result.emplace_back(input.data() + cps[at].begin, cps[end - 1].end - cps[at].begin);
        at = end;
    };
    while (at < cps.size()) {
        if (const std::size_t end = match_contraction(cps, at); end != at) {
            emit(end);
            continue;
        }

        std::size_t cursor = at;
        if (!is_newline(cps[cursor]) && !is_letter(cps[cursor]) && !is_number(cps[cursor]) &&
            cursor + 1 < cps.size() && (is_letter(cps[cursor + 1]) || is_mark(cps[cursor + 1]))) {
            ++cursor;
        }
        if (is_letter(cps[cursor]) || is_mark(cps[cursor])) {
            while (cursor < cps.size() && (is_letter(cps[cursor]) || is_mark(cps[cursor]))) ++cursor;
            emit(cursor);
            continue;
        }
        if (is_number(cps[at])) {
            emit(at + 1);
            continue;
        }

        cursor = at;
        if (cps[cursor].value == ' ' && cursor + 1 < cps.size() && !is_space(cps[cursor + 1]) &&
            !is_letter(cps[cursor + 1]) && !is_mark(cps[cursor + 1]) && !is_number(cps[cursor + 1])) {
            ++cursor;
        }
        const std::size_t punctuation_start = cursor;
        while (cursor < cps.size() && !is_space(cps[cursor]) && !is_letter(cps[cursor]) &&
               !is_mark(cps[cursor]) && !is_number(cps[cursor])) {
            ++cursor;
        }
        if (cursor > punctuation_start) {
            while (cursor < cps.size() && is_newline(cps[cursor])) ++cursor;
            emit(cursor);
            continue;
        }

        if (is_space(cps[at])) {
            cursor = at;
            std::size_t last_newline = at;
            bool saw_newline = false;
            while (cursor < cps.size() && is_space(cps[cursor])) {
                if (is_newline(cps[cursor])) {
                    saw_newline = true;
                    last_newline = cursor;
                }
                ++cursor;
            }
            if (saw_newline) {
                emit(last_newline + 1);
                continue;
            }
            if (cursor == cps.size()) {
                emit(cursor);
                continue;
            }
            if (cursor - at > 1) {
                emit(cursor - 1);
                continue;
            }
            emit(cursor);
            continue;
        }
        throw std::runtime_error("tokenizer pre-split made no progress");
    }
    return result;
}

std::vector<std::string> split_symbols(const std::string_view input) {
    std::vector<std::string> result;
    for (const Codepoint& cp : codepoints(input)) {
        result.emplace_back(input.substr(cp.begin, cp.end - cp.begin));
    }
    return result;
}

std::string pair_key(const std::string_view left, const std::string_view right) {
    std::string result;
    result.reserve(left.size() + right.size() + 1);
    result.append(left);
    result.push_back('\0');
    result.append(right);
    return result;
}

} // namespace

Tokenizer Tokenizer::load(const std::filesystem::path& model_directory) {
    Tokenizer result;

    std::vector<std::uint32_t> visible;
    for (std::uint32_t value = 33; value <= 126; ++value) visible.push_back(value);
    for (std::uint32_t value = 161; value <= 172; ++value) visible.push_back(value);
    for (std::uint32_t value = 174; value <= 255; ++value) visible.push_back(value);
    std::array<bool, 256> direct{};
    for (const std::uint32_t value : visible) direct[value] = true;
    std::uint32_t extra = 0;
    for (std::uint32_t byte = 0; byte < 256; ++byte) {
        const std::uint32_t symbol = direct[byte] ? byte : 256U + extra++;
        result.byte_to_symbol_[byte] = utf8(symbol);
        result.symbol_to_byte_.emplace(symbol, static_cast<std::uint8_t>(byte));
    }

    const Json vocabulary = Json::parse(read_text(model_directory / "vocab.json"));
    std::size_t maximum_id = 0;
    for (const auto& [token, id_value] : vocabulary.as_object()) {
        const std::int64_t id = id_value.as_integer();
        if (id < 0 || static_cast<std::uint64_t>(id) > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("invalid tokenizer vocabulary ID");
        }
        const auto token_id = static_cast<std::uint32_t>(id);
        result.token_to_id_.emplace(token, token_id);
        maximum_id = std::max(maximum_id, static_cast<std::size_t>(token_id));
    }
    result.id_to_token_.resize(maximum_id + 1);
    for (const auto& [token, id] : result.token_to_id_) result.id_to_token_[id] = token;

    std::istringstream merges(read_text(model_directory / "merges.txt"));
    std::string line;
    std::uint32_t rank = 0;
    while (std::getline(merges, line)) {
        if (line.empty() || line.starts_with("#version")) continue;
        const auto separator = line.find(' ');
        if (separator == std::string::npos || separator == 0 || separator + 1 >= line.size()) {
            throw std::runtime_error("invalid tokenizer merge rule");
        }
        result.merge_rank_.emplace(pair_key(
            std::string_view(line).substr(0, separator),
            std::string_view(line).substr(separator + 1)), rank++);
    }

    const Json tokenizer_config = Json::parse(read_text(model_directory / "tokenizer_config.json"));
    for (const auto& [id_text, description] : tokenizer_config.at("added_tokens_decoder").as_object()) {
        std::uint32_t id = 0;
        const auto parsed = std::from_chars(id_text.data(), id_text.data() + id_text.size(), id);
        if (parsed.ec != std::errc{} || parsed.ptr != id_text.data() + id_text.size()) {
            throw std::runtime_error("invalid added token ID");
        }
        const std::string token = description.at("content").as_string();
        if (result.id_to_token_.size() <= id) result.id_to_token_.resize(static_cast<std::size_t>(id) + 1);
        result.id_to_token_[id] = token;
        result.token_to_id_[token] = id;
        if (description.at("special").as_boolean()) {
            result.special_to_id_.emplace(token, id);
            result.special_ids_.insert(id);
        }
    }
    return result;
}

std::vector<std::uint32_t> Tokenizer::encode(
    const std::string_view text,
    const bool recognize_special_tokens) const {
    if (!recognize_special_tokens || special_to_id_.empty()) return encode_ordinary(text);
    std::vector<std::uint32_t> output;
    std::size_t cursor = 0;
    while (cursor < text.size()) {
        std::size_t special_at = std::string_view::npos;
        const std::string* special = nullptr;
        for (const auto& [candidate, id] : special_to_id_) {
            static_cast<void>(id);
            const std::size_t found = text.find(candidate, cursor);
            if (found < special_at || (found == special_at && special != nullptr && candidate.size() > special->size())) {
                special_at = found;
                special = &candidate;
            }
        }
        if (special == nullptr) {
            auto ordinary = encode_ordinary(text.substr(cursor));
            output.insert(output.end(), ordinary.begin(), ordinary.end());
            break;
        }
        if (special_at > cursor) {
            auto ordinary = encode_ordinary(text.substr(cursor, special_at - cursor));
            output.insert(output.end(), ordinary.begin(), ordinary.end());
        }
        output.push_back(special_to_id_.at(*special));
        cursor = special_at + special->size();
    }
    return output;
}

std::vector<std::uint32_t> Tokenizer::encode_ordinary(const std::string_view text) const {
    const std::string normalized = normalize_nfc(text);
    std::vector<std::uint32_t> output;
    for (const std::string_view piece : pretokenize(normalized)) {
        std::string byte_piece;
        for (const char raw : piece) {
            const auto byte = static_cast<unsigned char>(raw);
            byte_piece += byte_to_symbol_[byte];
        }
        for (const std::string& token : bpe(byte_piece)) {
            const auto found = token_to_id_.find(token);
            if (found == token_to_id_.end()) throw std::runtime_error("BPE emitted an unknown token");
            output.push_back(found->second);
        }
    }
    return output;
}

std::vector<std::string> Tokenizer::bpe(const std::string_view piece) const {
    {
        std::scoped_lock lock(*cache_mutex_);
        const auto cached = bpe_cache_.find(std::string(piece));
        if (cached != bpe_cache_.end()) return cached->second;
    }
    std::vector<std::string> symbols = split_symbols(piece);
    while (symbols.size() > 1) {
        std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
        std::string best_left;
        std::string best_right;
        for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto found = merge_rank_.find(pair_key(symbols[i], symbols[i + 1]));
            if (found != merge_rank_.end() && found->second < best_rank) {
                best_rank = found->second;
                best_left = symbols[i];
                best_right = symbols[i + 1];
            }
        }
        if (best_rank == std::numeric_limits<std::uint32_t>::max()) break;
        std::vector<std::string> merged;
        for (std::size_t i = 0; i < symbols.size();) {
            if (i + 1 < symbols.size() && symbols[i] == best_left && symbols[i + 1] == best_right) {
                merged.push_back(symbols[i] + symbols[i + 1]);
                i += 2;
            } else {
                merged.push_back(std::move(symbols[i++]));
            }
        }
        symbols = std::move(merged);
    }
    {
        std::scoped_lock lock(*cache_mutex_);
        if (bpe_cache_.size() < 65536) bpe_cache_.emplace(std::string(piece), symbols);
    }
    return symbols;
}

std::string Tokenizer::decode(const std::span<const std::uint32_t> token_ids) const {
    std::string output;
    for (const std::uint32_t id : token_ids) {
        if (id >= id_to_token_.size() || id_to_token_[id].empty()) {
            throw std::out_of_range("token ID is not in the vocabulary: " + std::to_string(id));
        }
        if (special_ids_.contains(id)) {
            output += id_to_token_[id];
            continue;
        }
        for (const Codepoint& cp : codepoints(id_to_token_[id])) {
            const auto byte = symbol_to_byte_.find(cp.value);
            if (byte == symbol_to_byte_.end()) throw std::runtime_error("token contains an invalid byte symbol");
            output.push_back(static_cast<char>(byte->second));
        }
    }
    return output;
}

} // namespace qwen38
