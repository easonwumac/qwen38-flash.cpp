#include "qwen38/token_embedding.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen38 {
namespace {
int dimension(const std::size_t value, const char* name) {
    if (value == 0 || value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("invalid token embedding dimension: ") + name);
    }
    return static_cast<int>(value);
}
} // namespace

MlxArray embed_token_batch(
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray& biases,
    const std::span<const std::uint32_t> tokens,
    const std::size_t vocabulary_size,
    const std::size_t hidden_size,
    const int group_size,
    const int bits) {
    if (tokens.empty() || tokens.size() > 1024) {
        throw std::runtime_error("token embedding batch must contain 1 to 1024 ids");
    }
    std::vector<std::int32_t> values;
    values.reserve(tokens.size());
    for (const std::uint32_t token : tokens) {
        if (token >= vocabulary_size ||
            token > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::runtime_error("token id is out of range");
        }
        values.push_back(static_cast<std::int32_t>(token));
    }
    const std::vector<int> id_shape{dimension(tokens.size(), "rows")};
    MlxArray ids = MlxArray::from_int32(values, id_shape);
    MlxArray embedding = MlxArray::dequantize(
        MlxArray::take_axis(weight, ids, 0),
        MlxArray::take_axis(scales, ids, 0),
        MlxArray::take_axis(biases, ids, 0),
        group_size,
        bits);
    const std::vector<int> shape{
        1, dimension(tokens.size(), "rows"), dimension(hidden_size, "hidden size")};
    return embedding.reshape(shape);
}

} // namespace qwen38
