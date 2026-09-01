#pragma once

#include "qwen38/api.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace qwen38 {

struct ServerConfig {
    std::string host{"127.0.0.1"};
    std::uint16_t port{11438};
    std::size_t max_header_bytes{64 * 1024};
    std::size_t max_body_bytes{4 * 1024 * 1024};
    std::size_t worker_threads{4};
    std::size_t max_pending_connections{128};
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
    void worker_loop();
    void start_workers();
    void stop_workers() noexcept;
    void join_workers() noexcept;

    ServerConfig config_;
    const Api& api_;
    std::atomic<bool> stopping_{false};
    std::atomic<int> listen_fd_{-1};
    std::mutex queue_mutex_;
    std::condition_variable queue_ready_;
    std::deque<int> pending_clients_;
    std::vector<std::thread> workers_;
};

} // namespace qwen38
