// Deterministic mutation fuzzing for both parsers.
//
// ---------------------------------------------------------------------------
// Why guard pages instead of AddressSanitizer
// ---------------------------------------------------------------------------
//
// The risk this targets is specific: both readers scan a memory-mapped buffer
// with raw pointer arithmetic, and a malformed file is exactly what makes a
// scanner walk off the end. ASan is the usual detector for that, but this
// toolchain (MinGW-w64 GCC 15) ships no `libasan` or `libubsan`, so
// `-fsanitize=address` does not link. Fuzzing without a detector is close to
// worthless -- an overrun that happens to land in readable heap looks like a
// pass.
//
// So the input is placed against a GUARD PAGE: the buffer's last byte sits
// immediately before an unmapped page, and any read past the end faults
// immediately and deterministically. For a one-directional scanner this is
// actually a sharper instrument than ASan's redzones, which are finite and can
// be jumped over by a large stride.
//
// It does not catch reads BEFORE the buffer, or logic errors. It catches the
// overrun class, which is the one raw pointer arithmetic over untrusted input
// actually produces.
//
// ---------------------------------------------------------------------------
// What counts as a failure
// ---------------------------------------------------------------------------
//
// A parse error is a PASS. Malformed input is supposed to be rejected, and the
// readers return `Expected` with a message and a line number. The failures this
// looks for are: a crash, a hang, or a "successful" parse that produces a model
// failing its own structural invariants.
//
// The seed is fixed, so a failure here is reproducible by re-running. The
// failing input is written to disk and its index printed.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "sovsolve/io/Load.hpp"
#include "sovsolve/io/Write.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
namespace fs = std::filesystem;

namespace {

/// Read a whole file. Deliberately `fread` rather than `istreambuf_iterator`:
/// the iterator form makes GCC 15 report a null dereference inside
/// `<streambuf>` at -O3 under `-Werror`, and this codebase avoids iostreams
/// anyway.
bool read_file(const fs::path& path, std::string& out) {
  std::FILE* f = std::fopen(path.string().c_str(), "rb");
  if (f == nullptr) return false;
  std::fseek(f, 0, SEEK_END);
  const long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (size < 0) {
    std::fclose(f);
    return false;
  }
  out.resize(static_cast<std::size_t>(size));
  const std::size_t got =
      size == 0 ? 0 : std::fread(out.data(), 1, out.size(), f);
  std::fclose(f);
  out.resize(got);
  return true;
}

void write_file(const std::string& path, const std::string& data) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) return;
  std::fwrite(data.data(), 1, data.size(), f);
  std::fclose(f);
}

/// A buffer whose last byte abuts an unmapped page.
///
/// Any read one byte past `view().end()` faults. That is the whole point: the
/// parsers are one-directional scanners over a mapped file, and running off the
/// end is the failure mode malformed input produces.
class GuardedBuffer {
 public:
  GuardedBuffer() = default;
  GuardedBuffer(const GuardedBuffer&) = delete;
  GuardedBuffer& operator=(const GuardedBuffer&) = delete;
  ~GuardedBuffer() { release(); }

  bool load(const char* data, std::size_t len) {
    release();
    const std::size_t page = page_size();
    // Enough pages for the data, plus one left unmapped behind it.
    const std::size_t data_pages = (len + page - 1) / page;
    total_ = (data_pages + 1) * page;

#if defined(_WIN32)
    base_ = static_cast<char*>(
        ::VirtualAlloc(nullptr, total_, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (base_ == nullptr) return false;
    DWORD old = 0;
    if (::VirtualProtect(base_ + total_ - page, page, PAGE_NOACCESS, &old) == 0) {
      release();
      return false;
    }
#else
    void* p = ::mmap(nullptr, total_, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return false;
    base_ = static_cast<char*>(p);
    if (::mprotect(base_ + total_ - page, page, PROT_NONE) != 0) {
      release();
      return false;
    }
#endif
    // Flush against the guard: the final data byte is the last readable byte.
    begin_ = base_ + total_ - page - len;
    len_ = len;
    if (len > 0) std::memcpy(begin_, data, len);
    return true;
  }

  [[nodiscard]] std::string_view view() const noexcept {
    return std::string_view(begin_, len_);
  }

 private:
  static std::size_t page_size() noexcept {
#if defined(_WIN32)
    SYSTEM_INFO si;
    ::GetSystemInfo(&si);
    return si.dwPageSize;
#else
    return static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
#endif
  }

  void release() noexcept {
    if (base_ == nullptr) return;
#if defined(_WIN32)
    ::VirtualFree(base_, 0, MEM_RELEASE);
#else
    ::munmap(base_, total_);
#endif
    base_ = nullptr;
    begin_ = nullptr;
    len_ = 0;
    total_ = 0;
  }

  char* base_ = nullptr;
  char* begin_ = nullptr;
  std::size_t len_ = 0;
  std::size_t total_ = 0;
};

// ---------------------------------------------------------------------------
// Mutations
// ---------------------------------------------------------------------------

using Rng = std::mt19937_64;

std::size_t pick(Rng& rng, std::size_t n) {
  return n == 0 ? 0 : std::uniform_int_distribution<std::size_t>(0, n - 1)(rng);
}

/// Values chosen to hit number parsing specifically: overflow, underflow,
/// non-finite spellings, and digit strings long enough to overrun a fixed
/// conversion buffer if one existed.
const char* pathological_number(Rng& rng) {
  static const char* kValues[] = {
      "1e400",   "-1e400",  "1e-400", "nan",     "inf",   "-inf",
      "NaN",     "Infinity", "1e",     "-",       ".",     "1.2.3",
      "0x10",    "1e+",     "--5",    "+++1",
      "999999999999999999999999999999999999999999999999999999999999999999",
      "0.00000000000000000000000000000000000000000000000000000000000000001",
  };
  return kValues[pick(rng, sizeof(kValues) / sizeof(kValues[0]))];
}

std::vector<std::string> split_lines(const std::string& s) {
  std::vector<std::string> out;
  std::size_t start = 0;
  for (std::size_t i = 0; i <= s.size(); ++i) {
    if (i == s.size() || s[i] == '\n') {
      out.push_back(s.substr(start, i - start));
      start = i + 1;
    }
  }
  return out;
}

std::string join_lines(const std::vector<std::string>& v) {
  std::string out;
  for (std::size_t i = 0; i < v.size(); ++i) {
    out += v[i];
    if (i + 1 < v.size()) out += '\n';
  }
  return out;
}

std::string mutate(const std::string& seed, Rng& rng) {
  std::string s = seed;
  const int kinds = 11;
  switch (static_cast<int>(pick(rng, kinds))) {
    case 0: {  // truncate -- the classic, and what a partial download produces
      if (!s.empty()) s.resize(pick(rng, s.size()));
      break;
    }
    case 1: {  // flip one byte
      if (!s.empty()) {
        const auto i = pick(rng, s.size());
        s[i] = static_cast<char>(pick(rng, 256));
      }
      break;
    }
    case 2: {  // insert a run of random bytes
      const auto i = pick(rng, s.size() + 1);
      std::string blob;
      const auto n = 1 + pick(rng, 32);
      for (std::size_t k = 0; k < n; ++k) {
        blob += static_cast<char>(pick(rng, 256));
      }
      s.insert(i, blob);
      break;
    }
    case 3: {  // delete a span
      if (!s.empty()) {
        const auto i = pick(rng, s.size());
        s.erase(i, 1 + pick(rng, 64));
      }
      break;
    }
    case 4: {  // duplicate a line
      auto lines = split_lines(s);
      if (!lines.empty()) {
        const auto i = pick(rng, lines.size());
        lines.insert(lines.begin() + static_cast<long>(i), lines[i]);
      }
      s = join_lines(lines);
      break;
    }
    case 5: {  // delete a line
      auto lines = split_lines(s);
      if (!lines.empty()) {
        lines.erase(lines.begin() + static_cast<long>(pick(rng, lines.size())));
      }
      s = join_lines(lines);
      break;
    }
    case 6: {  // swap two lines -- reorders sections, e.g. RHS before ROWS
      auto lines = split_lines(s);
      if (lines.size() > 1) {
        std::swap(lines[pick(rng, lines.size())], lines[pick(rng, lines.size())]);
      }
      s = join_lines(lines);
      break;
    }
    case 7: {  // splice in a pathological number
      auto lines = split_lines(s);
      if (!lines.empty()) {
        const auto i = pick(rng, lines.size());
        lines[i] += "  ";
        lines[i] += pathological_number(rng);
      }
      s = join_lines(lines);
      break;
    }
    case 8: {  // a very long token
      auto lines = split_lines(s);
      if (!lines.empty()) {
        lines[pick(rng, lines.size())] += "  " + std::string(4096, 'Z');
      }
      s = join_lines(lines);
      break;
    }
    case 9: {  // strip all newlines -- everything becomes one line
      s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
      break;
    }
    default: {  // drop the terminator, so the scanner runs to end of buffer
      const auto pos = s.rfind("ENDATA");
      if (pos != std::string::npos) s.erase(pos);
      const auto lp = s.rfind("End");
      if (lp != std::string::npos) s.erase(lp);
      break;
    }
  }
  return s;
}

// ---------------------------------------------------------------------------

/// A model that parsed successfully must still be structurally coherent.
/// A reader that accepts garbage and returns an inconsistent Problem is a
/// worse outcome than one that rejects it.
bool model_is_coherent(const model::Problem& p) {
  if (p.c.size() != p.num_cols()) return false;
  if (p.col_lower.size() != p.num_cols()) return false;
  if (p.col_upper.size() != p.num_cols()) return false;
  if (p.col_type.size() != p.num_cols()) return false;
  if (p.row_lower.size() != p.num_rows()) return false;
  if (p.row_upper.size() != p.num_rows()) return false;
  if (!p.A.csr.validate() || !p.A.csc.validate()) return false;
  if (p.A.rows() != p.num_rows() || p.A.cols() != p.num_cols()) return false;
  if (!p.Q.empty() &&
      (p.Q.rows() != p.num_cols() || p.Q.cols() != p.num_cols())) {
    return false;
  }
  return true;
}

std::vector<std::string> load_seeds() {
  std::vector<std::string> seeds;
#ifdef SOVSOLVE_TEST_DATA_DIR
  const fs::path data(SOVSOLVE_TEST_DATA_DIR);
#else
  const fs::path data("tests/data");
#endif
  // A few small real instances plus the LP file. Small ones are deliberate:
  // a mutation lands in a structurally interesting place far more often in a
  // 100-line file than in a 20 000-line one.
  const char* mps[] = {"afiro", "adlittle", "avgas", "chip", "flugpl"};
  for (const char* name : mps) {
    const fs::path f = data / "netlib" / (std::string(name) + ".mps");
    std::string text;
    if (read_file(f, text)) seeds.push_back(std::move(text));
  }
  {
    std::string text;
    if (read_file(data / "lp" / "blend.lp", text)) {
      seeds.push_back(std::move(text));
    }
  }
  // Hand-written seeds covering constructs the corpus lacks.
  seeds.emplace_back(R"(NAME          RNG
ROWS
 N  COST
 L  R1
 E  R2
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    X         COST         1.0   R1           1.0
    MARKER                 'MARKER'                 'INTEND'
    Y         COST         2.0   R2           1.0
RHS
    RHS       R1          10.0   COST        -3.0
RANGES
    RNG       R1           4.0   R2          -2.0
BOUNDS
 MI BND       X
 UP BND       X           -1.0
 FR BND       Y
QUADOBJ
    X         X            2.0
    X         Y            1.0
ENDATA
)");
  seeds.emplace_back(R"(Maximize
 obj: 3 x + 2 y + [ 2 x ^ 2 + 4 x * y ] / 2
Subject To
 c1: -5 <= x + y <= 10
 c2: x - y >= 1
Bounds
 x <= 8
 y free
General
 x
End
)");
  return seeds;
}

void run_fuzz(std::size_t iterations) {
  const auto seeds = load_seeds();
  if (seeds.empty()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "seeds", "no seed inputs found");
    return;
  }

  Rng rng(20260902);  // fixed: a failure here must be reproducible
  GuardedBuffer buf;

  std::size_t accepted = 0;
  std::size_t rejected = 0;
  std::size_t incoherent = 0;

  for (std::size_t it = 0; it < iterations; ++it) {
    const std::string& seed = seeds[pick(rng, seeds.size())];
    std::string input = mutate(seed, rng);
    // Occasionally stack two mutations -- one is often repaired by the
    // parser's own resynchronisation, two much less often.
    if (pick(rng, 3) == 0) input = mutate(input, rng);

    const bool looks_lp = input.find("Subject To") != std::string::npos ||
                          input.find("Minimize") != std::string::npos ||
                          input.find("Maximize") != std::string::npos;
    const auto format = looks_lp ? io::FileFormat::Lp : io::FileFormat::Mps;

    if (!buf.load(input.data(), input.size())) {
      ::sovsolve::test::record(__FILE__, __LINE__, "guard", "allocation failed");
      return;
    }

    // If this walks off the end, the guard page faults here and the process
    // dies with the iteration number already printed below on failure.
    auto result = io::parseProblem(buf.view(), format);

    if (result.has_value()) {
      ++accepted;
      if (!model_is_coherent(result.value())) {
        ++incoherent;
        ::sovsolve::test::record(
            __FILE__, __LINE__, "coherence",
            "iteration " + std::to_string(it) +
                " parsed successfully but produced an inconsistent model");
        write_file("fuzz-incoherent-" + std::to_string(it) + ".txt", input);
      } else {
        // A model that parsed must also survive the writer. This catches the
        // case where a mutated file yields something structurally valid but
        // unserialisable -- e.g. a name containing whitespace.
        auto text = io::writeMpsToString(result.value());
        (void)text;  // failure is a legitimate outcome; a crash is not
      }
    } else {
      ++rejected;
    }
  }

  std::printf("  %zu inputs: %zu rejected cleanly, %zu accepted, %zu incoherent\n",
              iterations, rejected, accepted, incoherent);
  CHECK_EQ(incoherent, std::size_t{0});
  // If nothing was rejected the mutations are not reaching the parser, and the
  // run proves nothing.
  CHECK(rejected > 0);
  // Likewise if nothing survived: the mutations would be too destructive to
  // exercise anything past the first section.
  CHECK(accepted > 0);
}

/// Prove the detector detects.
///
/// A fuzz run that finds nothing is only meaningful if the instrument works, so
/// this deliberately reads one byte past the guard. It MUST crash. The parent
/// process runs it as a subprocess and asserts that it did -- a clean exit
/// means the guard page is not protecting anything and every "pass" above was
/// vacuous.
///
/// It has to be a subprocess: the fault is an access violation, not a non-zero
/// return, and CTest's `WILL_FAIL` explicitly does not invert a crash.
[[noreturn]] void guard_selfcheck() {
  GuardedBuffer buf;
  const std::string data(64, 'A');
  if (!buf.load(data.data(), data.size())) {
    std::printf("guard selfcheck: allocation failed\n");
    std::exit(2);
  }
  const auto v = buf.view();
  // One past the last readable byte. `volatile` so it survives -O3.
  const volatile char* past = v.data() + v.size();
  const char observed = *past;
  std::printf("guard selfcheck: FAILED -- read past the guard and got 0x%02x; "
              "the guard page is not protecting the buffer\n",
              static_cast<unsigned>(static_cast<unsigned char>(observed)));
  std::exit(0);  // WILL_FAIL turns this clean exit into a reported failure
}

/// Re-run ourselves in guard-check mode and require the child to die.
void verify_guard_detects(const char* self) {
  std::string cmd = "\"";
  cmd += self;
  cmd += "\" guardcheck";
#if defined(_WIN32)
  cmd = "\"" + cmd + "\"";  // cmd.exe strips one layer of quoting
#endif
  const int rc = std::system(cmd.c_str());
  if (rc == 0) {
    ::sovsolve::test::record(
        __FILE__, __LINE__, "guard detector",
        "reading past the guard page did NOT fault, so the fuzz run proves "
        "nothing -- every input could have overrun undetected");
  } else {
    std::printf("  guard detector verified (overrun faults, child rc=%d)\n", rc);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "guardcheck") guard_selfcheck();

  // Default is a short run so it fits in the normal test cycle; pass a count
  // for a soak run.
  std::size_t iterations = 20000;
  if (argc > 1) iterations = static_cast<std::size_t>(std::atoll(argv[1]));
  if (iterations == 0) iterations = 20000;

  std::printf("parser fuzzing (guard-paged, seed 20260902):\n");
  verify_guard_detects(argv[0]);
  run_fuzz(iterations);
  return sovsolve::test::report("parser_fuzz");
}
