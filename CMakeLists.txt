cmake_minimum_required(VERSION 3.24)

project(qwen38_flash_cpp VERSION 0.1.0 LANGUAGES CXX)

option(QWEN38_BUILD_TESTS "Build qwen38-flash.cpp tests" ON)
option(QWEN38_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(QWEN38_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
set(QWEN38_MLXC_INCLUDE_DIR "" CACHE PATH "Directory containing mlx/c/mlx.h")
set(QWEN38_MLX_LIBRARY_DIR "" CACHE PATH "Directory containing libmlxc and libmlx")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(qwen38_core
    src/api.cpp
    src/http_server.cpp
    src/json.cpp
    src/mapped_file.cpp
    src/model_manifest.cpp
    src/runtime.cpp
    src/safetensors.cpp
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

install(TARGETS qwen38-server qwen38-inspect RUNTIME DESTINATION bin)

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

    add_library(qwen38_mlx src/mlx_backend.cpp)
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
    install(TARGETS qwen38-mlx-smoke RUNTIME DESTINATION bin)
endif()

if(QWEN38_BUILD_TESTS)
    include(CTest)
    enable_testing()

    add_executable(qwen38-tests
        tests/api_test.cpp
        tests/json_test.cpp
        tests/model_manifest_test.cpp
        tests/runtime_test.cpp
        tests/safetensors_test.cpp
        tests/test_main.cpp
    )
    target_link_libraries(qwen38-tests PRIVATE qwen38_core)
    add_test(NAME qwen38-tests COMMAND qwen38-tests)

    if(TARGET qwen38_mlx)
        add_executable(qwen38-mlx-tests tests/mlx_backend_test.cpp)
        target_link_libraries(qwen38-mlx-tests PRIVATE qwen38_mlx)
        set_target_properties(qwen38-mlx-tests PROPERTIES BUILD_RPATH "${QWEN38_MLX_LIBRARY_DIR}")
        add_test(NAME qwen38-mlx-tests COMMAND qwen38-mlx-tests)
    endif()
endif()
