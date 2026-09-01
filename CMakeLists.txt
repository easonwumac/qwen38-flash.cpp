cmake_minimum_required(VERSION 3.24)

project(qwen38_flash_cpp VERSION 0.1.0 LANGUAGES CXX)

option(QWEN38_BUILD_TESTS "Build qwen38-flash.cpp tests" ON)
option(QWEN38_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(QWEN38_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(qwen38_core
    src/api.cpp
    src/http_server.cpp
    src/runtime.cpp
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

install(TARGETS qwen38-server RUNTIME DESTINATION bin)

if(QWEN38_BUILD_TESTS)
    include(CTest)
    enable_testing()

    add_executable(qwen38-tests
        tests/api_test.cpp
        tests/runtime_test.cpp
        tests/test_main.cpp
    )
    target_link_libraries(qwen38-tests PRIVATE qwen38_core)
    add_test(NAME qwen38-tests COMMAND qwen38-tests)
endif()
