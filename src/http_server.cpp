#include "qwen38/http_server.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cctype>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

namespace qwen38 {
namespace {

class FileDescriptor final {
public:
    explicit FileDescriptor(const int value = -1) noexcept : value_(value) {}
    ~FileDescriptor() { reset(); }
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;
    FileDescriptor(FileDescriptor&& other) noexcept : value_(other.release()) {}
    FileDescriptor& operator=(FileDescriptor&& other) noexcept {
        if (this != &other) {
            reset(other.release());
        }
        return *this;
    }
    [[nodiscard]] int get() const noexcept { return value_; }
    [[nodiscard]] int release() noexcept {
        const int result = value_;
        value_ = -1;
        return result;
    }
    void reset(const int value = -1) noexcept {
        if (value_ >= 0) {
            ::close(value_);
        }
        value_ = value;
    }

private:
    int value_;
};

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string_view trim(std::string_view value) {
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
        value.remove_prefix(1);
    }
    while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) {
        value.remove_suffix(1);
    }
    return value;
}

std::string reason_phrase(const int status) {
    switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 413: return "Payload Too Large";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 503: return "Service Unavailable";
    default: return "Error";
    }
}

HttpResponse parse_error(const int status, const std::string_view message) {
    return {
        .status = status,
        .content_type = "application/json",
        .body = "{\"error\":{\"code\":\"bad_request\",\"message\":\"" +
            json_escape(message) + "\",\"type\":\"invalid_request_error\"}}",
    };
}

bool send_all(const int fd, const std::string_view data) {
    std::size_t offset = 0;
    while (offset < data.size()) {
        const auto remaining = data.size() - offset;
        const auto chunk = std::min(
            remaining,
            static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
        const ssize_t sent = ::send(fd, data.data() + offset, chunk, 0);
        if (sent < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (sent == 0) {
            return false;
        }
        offset += static_cast<std::size_t>(sent);
    }
    return true;
}

void write_response(const int fd, const HttpResponse& response) {
    std::ostringstream header;
    header << "HTTP/1.1 " << response.status << ' ' << reason_phrase(response.status)
           << "\r\nContent-Type: " << response.content_type;
    if (!response.body_stream) {
        header << "\r\nContent-Length: " << response.body.size()
               << "\r\nConnection: close\r\nX-Content-Type-Options: nosniff\r\n\r\n";
        const std::string wire = header.str() + response.body;
        static_cast<void>(send_all(fd, wire));
        return;
    }

    header << "\r\nTransfer-Encoding: chunked"
           << "\r\nCache-Control: no-cache"
           << "\r\nX-Accel-Buffering: no"
           << "\r\nConnection: close\r\nX-Content-Type-Options: nosniff\r\n\r\n";
    if (!send_all(fd, header.str())) return;
    bool connected = true;
    const auto sink = [&](const std::string_view fragment) {
        if (!connected) return false;
        std::ostringstream chunk_header;
        chunk_header << std::hex << fragment.size() << "\r\n";
        connected = send_all(fd, chunk_header.str()) && send_all(fd, fragment) &&
            send_all(fd, "\r\n");
        return connected;
    };
    response.body_stream(sink);
    if (connected) static_cast<void>(send_all(fd, "0\r\n\r\n"));
}

} // namespace

HttpServer::HttpServer(ServerConfig config, const Api& api)
    : config_(std::move(config)), api_(api) {}

HttpServer::~HttpServer() {
    stop();
}

void HttpServer::run() {
    FileDescriptor socket_fd(::socket(AF_INET, SOCK_STREAM, 0));
    if (socket_fd.get() < 0) {
        throw std::runtime_error("socket failed: " + std::string(std::strerror(errno)));
    }

    const int reuse = 1;
    if (::setsockopt(socket_fd.get(), SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) != 0) {
        throw std::runtime_error("setsockopt failed: " + std::string(std::strerror(errno)));
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(config_.port);
    if (::inet_pton(AF_INET, config_.host.c_str(), &address.sin_addr) != 1) {
        throw std::runtime_error("host must be an IPv4 address: " + config_.host);
    }
    if (::bind(socket_fd.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        throw std::runtime_error("bind failed: " + std::string(std::strerror(errno)));
    }
    if (::listen(socket_fd.get(), 128) != 0) {
        throw std::runtime_error("listen failed: " + std::string(std::strerror(errno)));
    }

    stopping_.store(false, std::memory_order_release);
    listen_fd_.store(socket_fd.get(), std::memory_order_release);
    std::cerr << "qwen38-server listening on http://" << config_.host << ':' << config_.port << '\n';

    while (!stopping_.load(std::memory_order_acquire)) {
        const int client = ::accept(socket_fd.get(), nullptr, nullptr);
        if (client < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (stopping_.load(std::memory_order_acquire) && (errno == EBADF || errno == EINVAL)) {
                break;
            }
            throw std::runtime_error("accept failed: " + std::string(std::strerror(errno)));
        }
        FileDescriptor client_fd(client);
        handle_client(client_fd.get());
    }

    listen_fd_.store(-1, std::memory_order_release);
    static_cast<void>(socket_fd.release());
}

void HttpServer::stop() noexcept {
    stopping_.store(true, std::memory_order_release);
    const int fd = listen_fd_.exchange(-1, std::memory_order_acq_rel);
    if (fd >= 0) {
        ::shutdown(fd, SHUT_RDWR);
        ::close(fd);
    }
}

void HttpServer::handle_client(const int client_fd) const {
    const timeval timeout{.tv_sec = 30, .tv_usec = 0};
    static_cast<void>(::setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)));
    static_cast<void>(::setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)));
#ifdef SO_NOSIGPIPE
    const int no_sigpipe = 1;
    static_cast<void>(::setsockopt(
        client_fd, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, sizeof(no_sigpipe)));
#endif

    std::string wire;
    wire.reserve(8192);
    std::array<char, 8192> buffer{};
    std::size_t header_end = std::string::npos;
    while ((header_end = wire.find("\r\n\r\n")) == std::string::npos) {
        if (wire.size() >= config_.max_header_bytes) {
            write_response(client_fd, parse_error(431, "Request headers are too large"));
            return;
        }
        const ssize_t received = ::recv(client_fd, buffer.data(), buffer.size(), 0);
        if (received <= 0) {
            return;
        }
        wire.append(buffer.data(), static_cast<std::size_t>(received));
    }

    HttpRequest request;
    const std::string_view headers(wire.data(), header_end);
    const auto first_line_end = headers.find("\r\n");
    if (first_line_end == std::string_view::npos) {
        write_response(client_fd, parse_error(400, "Malformed request line"));
        return;
    }
    {
        std::istringstream line(std::string(headers.substr(0, first_line_end)));
        std::string version;
        if (!(line >> request.method >> request.target >> version) || version != "HTTP/1.1") {
            write_response(client_fd, parse_error(400, "Expected an HTTP/1.1 request"));
            return;
        }
    }

    std::size_t cursor = first_line_end + 2;
    while (cursor < headers.size()) {
        const auto line_end = headers.find("\r\n", cursor);
        const auto end = line_end == std::string_view::npos ? headers.size() : line_end;
        const std::string_view line = headers.substr(cursor, end - cursor);
        const auto colon = line.find(':');
        if (colon == std::string_view::npos) {
            write_response(client_fd, parse_error(400, "Malformed request header"));
            return;
        }
        request.headers.emplace(
            lowercase(std::string(trim(line.substr(0, colon)))),
            std::string(trim(line.substr(colon + 1))));
        cursor = end + 2;
    }

    std::size_t content_length = 0;
    if (const auto it = request.headers.find("content-length"); it != request.headers.end()) {
        const auto* begin = it->second.data();
        const auto* end = begin + it->second.size();
        const auto parsed = std::from_chars(begin, end, content_length);
        if (parsed.ec != std::errc{} || parsed.ptr != end) {
            write_response(client_fd, parse_error(400, "Invalid Content-Length"));
            return;
        }
    }
    if (content_length > config_.max_body_bytes) {
        write_response(client_fd, parse_error(413, "Request body is too large"));
        return;
    }

    const std::size_t body_start = header_end + 4;
    while (wire.size() - body_start < content_length) {
        const ssize_t received = ::recv(client_fd, buffer.data(), buffer.size(), 0);
        if (received <= 0) {
            write_response(client_fd, parse_error(400, "Incomplete request body"));
            return;
        }
        wire.append(buffer.data(), static_cast<std::size_t>(received));
        if (wire.size() - body_start > config_.max_body_bytes) {
            write_response(client_fd, parse_error(413, "Request body is too large"));
            return;
        }
    }
    request.body.assign(wire.data() + body_start, content_length);

    try {
        write_response(client_fd, api_.handle(request));
    } catch (const std::exception& error) {
        write_response(client_fd, {
            .status = 500,
            .content_type = "application/json",
            .body = "{\"error\":{\"code\":\"internal_error\",\"message\":\"" +
                json_escape(error.what()) + "\",\"type\":\"server_error\"}}",
        });
    }
}

} // namespace qwen38
