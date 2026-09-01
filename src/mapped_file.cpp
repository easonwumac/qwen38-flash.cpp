#include "qwen38/mapped_file.hpp"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace qwen38 {

MappedFile::MappedFile(const std::filesystem::path& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error("cannot open " + path.string() + ": " + std::strerror(errno));
    }
    struct stat metadata {};
    if (::fstat(fd, &metadata) != 0) {
        const std::string message = std::strerror(errno);
        ::close(fd);
        throw std::runtime_error("cannot stat " + path.string() + ": " + message);
    }
    if (metadata.st_size < 0) {
        ::close(fd);
        throw std::runtime_error("negative file size: " + path.string());
    }
    size_ = static_cast<std::size_t>(metadata.st_size);
    if (size_ != 0) {
        data_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            const std::string message = std::strerror(errno);
            ::close(fd);
            throw std::runtime_error("cannot mmap " + path.string() + ": " + message);
        }
    }
    ::close(fd);
}

MappedFile::~MappedFile() { reset(); }

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        reset();
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

std::span<const std::byte> MappedFile::bytes() const noexcept {
    return {static_cast<const std::byte*>(data_), size_};
}

void MappedFile::reset() noexcept {
    if (data_ != nullptr) {
        ::munmap(data_, size_);
        data_ = nullptr;
        size_ = 0;
    }
}

} // namespace qwen38
