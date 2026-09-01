#include "qwen38/ngram.hpp"
#include "test.hpp"

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
}
