#include "qwen38/ngram.hpp"
#include "test.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

void run_ngram_tests() {
    qwen38::ModelConfig config;
    config.vocabulary_size = 248320;
    config.ngram_vocabulary_base = 20000000;
    config.ngram_vocabulary_divisor = 128;
    config.ngram_seed = 1234;
    config.end_of_sequence_token = 248044;
    qwen38::NgramHash hash(config);
    QWEN38_CHECK(hash.total_rows() == 320001536);
    qwen38::NgramState state;
    const auto first = hash.row_ids(5, state);
    QWEN38_CHECK(first[0] == 15389869);
    QWEN38_CHECK(first[7] == 155458159);
    QWEN38_CHECK(first[8] == 179763390);
    QWEN38_CHECK(first[15] == 315720898);
    const auto second = hash.row_ids(7, state);
    QWEN38_CHECK(second[0] == 12441580);
    QWEN38_CHECK(second[15] == 314484167);
    static_cast<void>(hash.row_ids(248044, state));
    const auto after_eos = hash.row_ids(9, state);
    QWEN38_CHECK(after_eos[0] == 18043673);
    QWEN38_CHECK(after_eos[15] == 307951937);

    std::array<char, 128> directory_template{};
    const std::string temporary =
        (std::filesystem::temp_directory_path() / "qwen38-ngram-XXXXXX").string();
    QWEN38_CHECK(temporary.size() < directory_template.size());
    std::copy(temporary.begin(), temporary.end(), directory_template.begin());
    char* directory = ::mkdtemp(directory_template.data());
    QWEN38_CHECK(directory != nullptr);
    const std::filesystem::path table_path =
        std::filesystem::path(directory) / "ngram_table.bf16.aos";
    std::array<std::uint16_t, 320> table{};
    for (std::size_t column = 0; column < 160; ++column) {
        table[160 + column] = column % 2 == 0 ? 0x3f80 : 0xc000;
    }
    {
        std::ofstream output(table_path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(table.data()), sizeof(table));
    }
    {
        qwen38::NgramTable bf16_table(directory, 2, true);
        QWEN38_CHECK(bf16_table.uses_aos());
        QWEN38_CHECK(bf16_table.uses_bf16_aos());
        const std::array<std::int64_t, 1> row{1};
        const auto values = bf16_table.gather(row);
        QWEN38_CHECK(values.size() == 160);
        QWEN38_CHECK(values[0] == 1.0F);
        QWEN38_CHECK(values[1] == -2.0F);
        QWEN38_CHECK(values[158] == 1.0F);
        QWEN38_CHECK(values[159] == -2.0F);
    }
    std::filesystem::remove_all(directory);
}
