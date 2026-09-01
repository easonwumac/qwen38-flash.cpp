#pragma once

#include "qwen38/api.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

namespace qwen38 {

struct ServerConfig {
    std::string host{"127.0.0.1"};
    std::uint16_t port{11438};
    std::size_t max_header_bytes{64 * 1024};
    std::size_t max_body_bytes{4 * 1024 * 1024};
};

class HttpServer final {
public:
    HttpServer(ServerConfig config, const Api& api);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    void run();
    void stop() noexcept;

private:
    void handle_client(int client_fd) const;

    ServerConfig config_;
    const Api& api_;
    std::atomic<bool> stopping_{false};
    std::atomic<int> listen_fd_{-1};
};

} // namespace qwen38
