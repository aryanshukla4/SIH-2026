#include "io/detail/MappedFile.hpp"

#include <cstdio>
#include <cstring>
#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef SOVSOLVE_WITH_ZLIB
#include <zlib.h>
#endif

namespace sovsolve::io::detail {

using core::ErrorCode;
using core::make_error;
using core::Status;

bool looks_gzipped(std::string_view bytes) noexcept {
  return bytes.size() >= 2 &&
         static_cast<unsigned char>(bytes[0]) == 0x1f &&
         static_cast<unsigned char>(bytes[1]) == 0x8b;
}

Status inflate_gzip(std::string_view compressed, std::vector<char>& out) {
#ifdef SOVSOLVE_WITH_ZLIB
  z_stream zs{};
  // 15 + 16 selects gzip framing rather than raw deflate or zlib framing.
  if (inflateInit2(&zs, 15 + 16) != Z_OK) {
    return make_error(ErrorCode::FileReadFailed, "zlib initialization failed");
  }

  zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(compressed.data()));
  zs.avail_in = static_cast<uInt>(compressed.size());

  // Text compresses ~4x, so this usually avoids any regrowth at all.
  out.clear();
  out.resize(compressed.size() * 4 + 1024);

  std::size_t written = 0;
  int rc = Z_OK;
  do {
    if (written == out.size()) out.resize(out.size() * 2);
    zs.next_out = reinterpret_cast<Bytef*>(out.data() + written);
    zs.avail_out = static_cast<uInt>(out.size() - written);
    rc = inflate(&zs, Z_NO_FLUSH);
    if (rc != Z_OK && rc != Z_STREAM_END) {
      inflateEnd(&zs);
      return make_error(ErrorCode::FileReadFailed,
                        "gzip stream is corrupt or truncated");
    }
    written = out.size() - zs.avail_out;
  } while (rc != Z_STREAM_END);

  inflateEnd(&zs);
  out.resize(written);
  return Status::Ok();
#else
  (void)compressed;
  (void)out;
  return make_error(ErrorCode::UnsupportedFeature,
                    "input is gzip-compressed but zlib support was not compiled "
                    "in; decompress the file or rebuild with zlib");
#endif
}

MappedFile::~MappedFile() { close(); }

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      handle_(std::exchange(other.handle_, nullptr)),
      file_(std::exchange(other.file_, nullptr)),
      owned_(std::move(other.owned_)),
      compressed_(std::exchange(other.compressed_, false)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    close();
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    handle_ = std::exchange(other.handle_, nullptr);
    file_ = std::exchange(other.file_, nullptr);
    owned_ = std::move(other.owned_);
    compressed_ = std::exchange(other.compressed_, false);
  }
  return *this;
}

#if defined(_WIN32)

Status MappedFile::open(const std::string& path) {
  close();

  HANDLE file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    return make_error(ErrorCode::FileNotFound, "cannot open " + path);
  }

  LARGE_INTEGER file_size{};
  if (!GetFileSizeEx(file, &file_size)) {
    CloseHandle(file);
    return make_error(ErrorCode::FileReadFailed, "cannot size " + path);
  }
  if (file_size.QuadPart == 0) {
    CloseHandle(file);
    return make_error(ErrorCode::FileReadFailed, path + " is empty");
  }

  HANDLE mapping =
      CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (mapping == nullptr) {
    CloseHandle(file);
    return make_error(ErrorCode::FileReadFailed, "cannot map " + path);
  }

  const void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  if (view == nullptr) {
    CloseHandle(mapping);
    CloseHandle(file);
    return make_error(ErrorCode::FileReadFailed, "cannot view " + path);
  }

  data_ = static_cast<const char*>(view);
  size_ = static_cast<std::size_t>(file_size.QuadPart);
  handle_ = mapping;
  file_ = file;

  if (looks_gzipped(std::string_view{data_, size_})) {
    compressed_ = true;
    std::vector<char> inflated;
    const Status st = inflate_gzip(std::string_view{data_, size_}, inflated);
    // The mapping is released either way: on success the inflated buffer
    // replaces it, on failure it is no longer needed.
    if (!st.ok()) {
      close();
      return st;
    }
    owned_ = std::move(inflated);
    UnmapViewOfFile(data_);
    CloseHandle(mapping);
    CloseHandle(file);
    data_ = nullptr;
    size_ = 0;
    handle_ = nullptr;
    file_ = nullptr;
  }
  return Status::Ok();
}

void MappedFile::close() noexcept {
  if (data_ != nullptr) UnmapViewOfFile(data_);
  if (handle_ != nullptr) CloseHandle(static_cast<HANDLE>(handle_));
  if (file_ != nullptr && file_ != INVALID_HANDLE_VALUE) {
    CloseHandle(static_cast<HANDLE>(file_));
  }
  data_ = nullptr;
  size_ = 0;
  handle_ = nullptr;
  file_ = nullptr;
  owned_.clear();
  owned_.shrink_to_fit();
  compressed_ = false;
}

#else  // POSIX

Status MappedFile::open(const std::string& path) {
  close();

  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return make_error(ErrorCode::FileNotFound, "cannot open " + path);

  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    return make_error(ErrorCode::FileReadFailed, "cannot size " + path);
  }
  if (st.st_size == 0) {
    ::close(fd);
    return make_error(ErrorCode::FileReadFailed, path + " is empty");
  }

  const auto len = static_cast<std::size_t>(st.st_size);
  void* view = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
  if (view == MAP_FAILED) {
    ::close(fd);
    return make_error(ErrorCode::FileReadFailed, "cannot map " + path);
  }

  data_ = static_cast<const char*>(view);
  size_ = len;
  file_ = reinterpret_cast<void*>(static_cast<std::intptr_t>(fd));

  if (looks_gzipped(std::string_view{data_, size_})) {
    compressed_ = true;
    std::vector<char> inflated;
    const Status status = inflate_gzip(std::string_view{data_, size_}, inflated);
    if (!status.ok()) {
      close();
      return status;
    }
    owned_ = std::move(inflated);
    ::munmap(const_cast<char*>(data_), size_);
    ::close(fd);
    data_ = nullptr;
    size_ = 0;
    file_ = nullptr;
  }
  return Status::Ok();
}

void MappedFile::close() noexcept {
  if (data_ != nullptr) ::munmap(const_cast<char*>(data_), size_);
  if (file_ != nullptr) {
    ::close(static_cast<int>(reinterpret_cast<std::intptr_t>(file_)));
  }
  data_ = nullptr;
  size_ = 0;
  handle_ = nullptr;
  file_ = nullptr;
  owned_.clear();
  owned_.shrink_to_fit();
  compressed_ = false;
}

#endif

}  // namespace sovsolve::io::detail
