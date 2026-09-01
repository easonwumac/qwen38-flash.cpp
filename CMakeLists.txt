cmake_minimum_required(VERSION 3.24)

project(qwen38_flash_cpp VERSION 0.1.0 LANGUAGES CXX)

option(QWEN38_BUILD_TESTS "Build qwen38-flash.cpp tests" ON)
option(QWEN38_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(QWEN38_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
set(QWEN38_MLXC_INCLUDE_DIR "" CACHE PATH "Directory containing mlx/c/mlx.h")
set(QWEN38_MLX_LIBRARY_DIR "" CACHE PATH "Directory containing libmlxc and libmlx")
option(QWEN38_ENABLE_TOKENIZER "Build the native Qwen tokenizer" OFF)
set(QWEN38_UTF8PROC_INCLUDE_DIR "" CACHE PATH "Directory containing utf8proc.h")
set(QWEN38_UTF8PROC_LIBRARY_DIR "" CACHE PATH "Directory containing libutf8proc")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(qwen38_core
    src/api.cpp
    src/chat_template.cpp
    src/http_server.cpp
    src/history_draft.cpp
    src/json.cpp
    src/mapped_file.cpp
    src/model_manifest.cpp
    src/mtp_depth_policy.cpp
    src/mtp_lifecycle.cpp
    src/ngram.cpp
    src/quantization_geometry.cpp
    src/runtime.cpp
    src/safetensors.cpp
    src/tool_call.cpp
)
target_include_directories(qwen38_core PUBLIC include)
target_compile_features(qwen38_core PUBLIC cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
    target_compile_options(qwen38_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

if(QWEN38_ENABLE_ASAN)
    target_compile_options(qwen38_core PUBLIC -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(qwen38_core PUBLIC -fsanitize=address)
endif()

if(QWEN38_ENABLE_UBSAN)
    target_compile_options(qwen38_core PUBLIC -fsanitize=undefined -fno-omit-frame-pointer)
    target_link_options(qwen38_core PUBLIC -fsanitize=undefined)
endif()

add_executable(qwen38-server src/main.cpp)
target_link_libraries(qwen38-server PRIVATE qwen38_core)

add_executable(qwen38-inspect src/inspect.cpp)
target_link_libraries(qwen38-inspect PRIVATE qwen38_core)

add_executable(qwen38-ngram-smoke src/ngram_smoke.cpp)
target_link_libraries(qwen38-ngram-smoke PRIVATE qwen38_core)

install(TARGETS qwen38-server qwen38-inspect qwen38-ngram-smoke RUNTIME DESTINATION bin)

if(QWEN38_ENABLE_TOKENIZER)
    find_path(QWEN38_UTF8PROC_HEADER utf8proc.h
        HINTS "${QWEN38_UTF8PROC_INCLUDE_DIR}" /opt/homebrew/include /usr/local/include)
    find_library(QWEN38_UTF8PROC_LIBRARY NAMES utf8proc
        HINTS "${QWEN38_UTF8PROC_LIBRARY_DIR}" /opt/homebrew/lib /usr/local/lib)
    if(NOT QWEN38_UTF8PROC_HEADER OR NOT QWEN38_UTF8PROC_LIBRARY)
        message(FATAL_ERROR "QWEN38_ENABLE_TOKENIZER requires utf8proc")
    endif()
    add_library(qwen38_tokenizer src/tokenizer.cpp)
    target_include_directories(qwen38_tokenizer PUBLIC include)
    target_include_directories(qwen38_tokenizer SYSTEM PRIVATE "${QWEN38_UTF8PROC_HEADER}")
    target_link_libraries(qwen38_tokenizer PUBLIC qwen38_core "${QWEN38_UTF8PROC_LIBRARY}")
    if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
        target_compile_options(qwen38_tokenizer PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()

    add_executable(qwen38-tokenize src/tokenize.cpp)
    target_link_libraries(qwen38-tokenize PRIVATE qwen38_tokenizer)
    install(TARGETS qwen38-tokenize RUNTIME DESTINATION bin)
endif()

if(QWEN38_MLXC_INCLUDE_DIR OR QWEN38_MLX_LIBRARY_DIR)
    if(NOT QWEN38_MLXC_INCLUDE_DIR OR NOT QWEN38_MLX_LIBRARY_DIR)
        message(FATAL_ERROR "Both QWEN38_MLXC_INCLUDE_DIR and QWEN38_MLX_LIBRARY_DIR are required")
    endif()
    find_path(QWEN38_MLXC_HEADER mlx/c/mlx.h PATHS "${QWEN38_MLXC_INCLUDE_DIR}" NO_DEFAULT_PATH)
    find_library(QWEN38_MLXC_LIBRARY NAMES mlxc PATHS "${QWEN38_MLX_LIBRARY_DIR}" NO_DEFAULT_PATH)
    find_library(QWEN38_MLX_LIBRARY NAMES mlx PATHS "${QWEN38_MLX_LIBRARY_DIR}" NO_DEFAULT_PATH)
    if(NOT QWEN38_MLXC_HEADER OR NOT QWEN38_MLXC_LIBRARY OR NOT QWEN38_MLX_LIBRARY)
        message(FATAL_ERROR "The configured MLX C include/library directories are incomplete")
    endif()

    add_library(qwen38_mlx
        src/decoder_layer.cpp
        src/gated_delta_net.cpp
        src/hyper_connection.cpp
        src/mlx_backend.cpp
        src/model.cpp
        src/mtp_head.cpp
        src/mtp_runner.cpp
        src/mtp_verifier.cpp
        src/ple.cpp
        src/self_attention.cpp
        src/sparse_moe.cpp
    )
    target_include_directories(qwen38_mlx PUBLIC include)
    target_include_directories(qwen38_mlx SYSTEM PUBLIC "${QWEN38_MLXC_HEADER}")
    target_link_libraries(qwen38_mlx PUBLIC qwen38_core "${QWEN38_MLXC_LIBRARY}" "${QWEN38_MLX_LIBRARY}")
    set_target_properties(qwen38_mlx PROPERTIES BUILD_RPATH "${QWEN38_MLX_LIBRARY_DIR}")
    target_compile_definitions(qwen38_mlx PUBLIC QWEN38_HAS_MLX=1)
    if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
        target_compile_options(qwen38_mlx PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()

    add_executable(qwen38-mlx-smoke src/mlx_smoke.cpp)
    target_link_libraries(qwen38-mlx-smoke PRIVATE qwen38_mlx)
    set_target_properties(qwen38-mlx-smoke PROPERTIES BUILD_RPATH "${QWEN38_MLX_LIBRARY_DIR}")
    add_executable(qwen38-qmm-smoke src/qmm_smoke.cpp)
    target_link_libraries(qwen38-qmm-smoke PRIVATE qwen38_mlx)
    set_target_properties(qwen38-qmm-smoke PROPERTIES BUILD_RPATH "${QWEN38_MLX_LIBRARY_DIR}")
    add_executable(qwen38-embedding-smoke src/embedding_smoke.cpp)
    target_link_libraries(qwen38-embedding-smoke PRIVATE qwen38_mlx)
    set_target_properties(qwen38-embedding-smoke PROPERTIES BUILD_RPATH "${QWEN38_MLX_LIBRARY_DIR}")
    add_executable(qwen38-hyper-connection-smoke src/hyper_connection_smoke.cpp)
    target_link_libraries(qwen38-hyper-connection-smoke PRIVATE qwen38_mlx)
    set_target_properties(qwen38-hyper-connection-smoke PROPERTIES BUILD_RPATH "${QWEN38_MLX_LIBRARY_DIR}")
    add_executable(qwen38-sparse-moe-smoke src/sparse_moe_smoke.cpp)
    target_link_libraries(qwen38-sparse-moe-smoke PRIVATE qwen38_mlx)
    set_target_properties(qwen38-sparse-moe-smoke PROPERTIES BUILD_RPATH "${QWEN38_MLX_LIBRARY_DIR}")
    add_executable(qwen38-gated-delta-net-smoke src/gated_delta_net_smoke.cpp)
    target_link_libraries(qwen38-gated-delta-net-smoke PRIVATE qwen38_mlx)
    set_target_properties(qwen38-gated-delta-net-smoke PROPERTIES BUILD_RPATH "${QWEN38_MLX_LIBRARY_DIR}")
    add_executable(qwen38-self-attention-smoke src/self_attention_smoke.cpp)
    target_link_libraries(qwen38-self-attention-smoke PRIVATE qwen38_mlx)
    set_target_properties(qwen38-self-attention-smoke PROPERTIES BUILD_RPATH "${QWEN38_MLX_LIBRARY_DIR}")
    add_executable(qwen38-ple-smoke src/ple_smoke.cpp)
    target_link_libraries(qwen38-ple-smoke PRIVATE qwen38_mlx)
    set_target_properties(qwen38-ple-smoke PROPERTIES BUILD_RPATH "${QWEN38_MLX_LIBRARY_DIR}")
    add_executable(qwen38-decoder-layer-smoke src/decoder_layer_smoke.cpp)
    target_link_libraries(qwen38-decoder-layer-smoke PRIVATE qwen38_mlx)
    set_target_properties(qwen38-decoder-layer-smoke PROPERTIES BUILD_RPATH "${QWEN38_MLX_LIBRARY_DIR}")
    add_executable(qwen38-model-smoke src/model_smoke.cpp)
    target_link_libraries(qwen38-model-smoke PRIVATE qwen38_mlx)
    set_target_properties(qwen38-model-smoke PROPERTIES BUILD_RPATH "${QWEN38_MLX_LIBRARY_DIR}")
    add_executable(qwen38-model-bench src/model_bench.cpp)
    target_link_libraries(qwen38-model-bench PRIVATE qwen38_mlx)
    set_target_properties(qwen38-model-bench PROPERTIES BUILD_RPATH "${QWEN38_MLX_LIBRARY_DIR}")

    add_executable(qwen38-prefill-smoke src/prefill_smoke.cpp)
    target_link_libraries(qwen38-prefill-smoke PRIVATE qwen38_mlx)
    set_target_properties(qwen38-prefill-smoke PROPERTIES BUILD_RPATH "${QWEN38_MLX_LIBRARY_DIR}")
    add_executable(qwen38-mtp-head-smoke src/mtp_head_smoke.cpp)
    target_link_libraries(qwen38-mtp-head-smoke PRIVATE qwen38_mlx)
    set_target_properties(qwen38-mtp-head-smoke PROPERTIES BUILD_RPATH "${QWEN38_MLX_LIBRARY_DIR}")
    add_executable(qwen38-mtp-fixture-smoke src/mtp_fixture_smoke.cpp)
    target_link_libraries(qwen38-mtp-fixture-smoke PRIVATE qwen38_mlx)
    set_target_properties(qwen38-mtp-fixture-smoke PROPERTIES BUILD_RPATH "${QWEN38_MLX_LIBRARY_DIR}")

    add_executable(qwen38-mtp-head-compare src/mtp_head_compare.cpp)
    target_link_libraries(qwen38-mtp-head-compare PRIVATE qwen38_mlx)
    set_target_properties(qwen38-mtp-head-compare PROPERTIES BUILD_RPATH "${QWEN38_MLX_LIBRARY_DIR}")
    add_executable(qwen38-mtp-verifier-smoke src/mtp_verifier_smoke.cpp)
    target_link_libraries(qwen38-mtp-verifier-smoke PRIVATE qwen38_mlx)
    set_target_properties(qwen38-mtp-verifier-smoke PROPERTIES BUILD_RPATH "${QWEN38_MLX_LIBRARY_DIR}")
    add_executable(qwen38-mtp-verifier-bench src/mtp_verifier_bench.cpp)
    target_link_libraries(qwen38-mtp-verifier-bench PRIVATE qwen38_mlx)
    set_target_properties(qwen38-mtp-verifier-bench PROPERTIES BUILD_RPATH "${QWEN38_MLX_LIBRARY_DIR}")
    add_executable(qwen38-mtp-round-smoke src/mtp_round_smoke.cpp)
    target_link_libraries(qwen38-mtp-round-smoke PRIVATE qwen38_mlx)
    set_target_properties(qwen38-mtp-round-smoke PROPERTIES BUILD_RPATH "${QWEN38_MLX_LIBRARY_DIR}")
    install(TARGETS qwen38-mlx-smoke qwen38-qmm-smoke qwen38-embedding-smoke
        qwen38-hyper-connection-smoke qwen38-sparse-moe-smoke qwen38-gated-delta-net-smoke
        qwen38-self-attention-smoke
        qwen38-ple-smoke
        qwen38-decoder-layer-smoke
        qwen38-model-smoke
        qwen38-model-bench
        qwen38-prefill-smoke
        qwen38-mtp-head-smoke
        qwen38-mtp-fixture-smoke
        qwen38-mtp-head-compare
        qwen38-mtp-verifier-smoke
        qwen38-mtp-verifier-bench
        qwen38-mtp-round-smoke
        RUNTIME DESTINATION bin)

    if(TARGET qwen38_tokenizer)
        add_library(qwen38_inference src/native_engine.cpp)
        target_include_directories(qwen38_inference PUBLIC include)
        target_link_libraries(qwen38_inference PUBLIC qwen38_mlx qwen38_tokenizer)
        if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
            target_compile_options(qwen38_inference PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
        endif()
        target_link_libraries(qwen38-server PRIVATE qwen38_inference)
        target_compile_definitions(qwen38-server PRIVATE QWEN38_HAS_INFERENCE=1)

        add_executable(qwen38-prefill-prompt-smoke src/prefill_prompt_smoke.cpp)
        target_link_libraries(qwen38-prefill-prompt-smoke PRIVATE qwen38_mlx qwen38_tokenizer)
        set_target_properties(qwen38-prefill-prompt-smoke PROPERTIES
            BUILD_RPATH "${QWEN38_MLX_LIBRARY_DIR}")
    endif()
endif()

if(QWEN38_BUILD_TESTS)
    include(CTest)
    enable_testing()

    add_executable(qwen38-tests
        tests/api_test.cpp
        tests/chat_template_test.cpp
        tests/history_draft_test.cpp
        tests/json_test.cpp
        tests/model_manifest_test.cpp
        tests/mtp_depth_policy_test.cpp
        tests/mtp_lifecycle_test.cpp
        tests/ngram_test.cpp
        tests/quantization_geometry_test.cpp
        tests/runtime_test.cpp
        tests/safetensors_test.cpp
        tests/test_main.cpp
        tests/tool_call_test.cpp
    )
    target_link_libraries(qwen38-tests PRIVATE qwen38_core)
    add_test(NAME qwen38-tests COMMAND qwen38-tests)

    if(APPLE)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            add_test(
                NAME qwen38-memory-guard-tests
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/memory_guard_test.py
            )
        endif()
    endif()

    if(TARGET qwen38_mlx)
        add_executable(qwen38-mlx-tests tests/mlx_backend_test.cpp)
        target_link_libraries(qwen38-mlx-tests PRIVATE qwen38_mlx)
        set_target_properties(qwen38-mlx-tests PROPERTIES BUILD_RPATH "${QWEN38_MLX_LIBRARY_DIR}")
        add_test(NAME qwen38-mlx-tests COMMAND qwen38-mlx-tests)
    endif()
    if(TARGET qwen38_tokenizer)
        add_executable(qwen38-tokenizer-tests tests/tokenizer_test.cpp)
        target_link_libraries(qwen38-tokenizer-tests PRIVATE qwen38_tokenizer)
        add_test(NAME qwen38-tokenizer-tests COMMAND qwen38-tokenizer-tests)
        set_tests_properties(qwen38-tokenizer-tests PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()
