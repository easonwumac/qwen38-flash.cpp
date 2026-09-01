#include "qwen38/tokenizer.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " MODEL_DIRECTORY TEXT\n";
        return EXIT_FAILURE;
    }
    try {
        const qwen38::Tokenizer tokenizer = qwen38::Tokenizer::load(argv[1]);
        const auto ids = tokenizer.encode(argv[2]);
        std::cout << '[';
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (i != 0) std::cout << ',';
            std::cout << ids[i];
        }
        std::cout << "]\n" << tokenizer.decode(ids) << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-tokenize: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
