// Memory-mapped file access, with transparent gzip.
//
// Reading a 500 MB MPS file through iostreams means the kernel copies it into
// the page cache and the runtime copies it again into a userspace buffer.
// Mapping it hands the parser a pointer straight at the page cache, so the
// second copy never happens and pages fault in on demand.
//
// It also gives the tokenizer a single flat address range, which is what lets
// the symbol table key on `string_view`s into the file rather than owning
// copies of every name.
//
// Gzip input cannot be mapped (MIPLIB ships `.mps.gz`), so it is inflated into
// one exactly-sized owned buffer instead. `contents()` returns the same flat
// view in both cases, so nothing downstream has to care which path was taken.

#ifndef SOVSOLVE_IO_DETAIL_MAPPED_FILE_HPP
#define SOVSOLVE_IO_DETAIL_MAPPED_FILE_HPP

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "sovsolve/core/Status.hpp"

namespace sovsolve::io::detail {

class MappedFile {
 public:
  MappedFile() = default;
  ~MappedFile();

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;

  /// Map `path`, or inflate it if it is gzip-compressed.
  ///
  /// Compression is detected from the magic bytes rather than the extension,
  /// since benchmark archives are not consistent about naming.
  [[nodiscard]] core::Status open(const std::string& path);

  /// The whole file as one contiguous range. Valid until this object dies.
  [[nodiscard]] std::string_view contents() const noexcept {
    return owned_.empty() ? std::string_view{data_, size_}
                          : std::string_view{owned_.data(), owned_.size()};
  }

  [[nodiscard]] std::size_t size() const noexcept {
    return owned_.empty() ? size_ : owned_.size();
  }

  [[nodiscard]] bool was_compressed() const noexcept { return compressed_; }

 private:
  void close() noexcept;

  const char* data_ = nullptr;   ///< mapping base, when mapped
  std::size_t size_ = 0;
  void* handle_ = nullptr;       ///< OS mapping handle
  void* file_ = nullptr;         ///< OS file handle
  std::vector<char> owned_;      ///< inflated bytes, when compressed
  bool compressed_ = false;
};

/// True when the buffer begins with the gzip magic number.
[[nodiscard]] bool looks_gzipped(std::string_view bytes) noexcept;

/// Inflate a gzip stream. Returns an error when zlib support is not compiled
/// in, so the caller can report that rather than mis-parsing binary data.
[[nodiscard]] core::Status inflate_gzip(std::string_view compressed,
                                        std::vector<char>& out);

}  // namespace sovsolve::io::detail

#endif  // SOVSOLVE_IO_DETAIL_MAPPED_FILE_HPP
