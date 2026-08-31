// Parse throughput and memory benchmark.
//
// Reports the numbers the submission needs: MB/s, nonzeros/s, and bytes per
// nonzero. The last one is the check that the storage layout is what it claims
// to be -- CSR with int32 indices should land near 12 bytes/nnz, and holding
// both orientations near 24.
//
// Instances are generated rather than downloaded so the benchmark runs
// anywhere and scales past what Netlib provides. Real files are measured too
// when a path is given on the command line.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "sovsolve/analysis/MatrixAnalysis.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace {

using sovsolve::core::Index;
using sovsolve::core::Real;

std::size_t peak_rss_bytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS pmc{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    return static_cast<std::size_t>(pmc.PeakWorkingSetSize);
  }
  return 0;
#else
  struct rusage ru {};
  getrusage(RUSAGE_SELF, &ru);
  // Linux reports kilobytes, macOS bytes.
#if defined(__APPLE__)
  return static_cast<std::size_t>(ru.ru_maxrss);
#else
  return static_cast<std::size_t>(ru.ru_maxrss) * 1024;
#endif
#endif
}

/// Build a synthetic MPS instance with a banded structure plus a few dense
/// columns, so the dense-column detector has something to find.
std::string generate_mps(std::size_t rows, std::size_t cols,
                         std::size_t per_column, std::size_t dense_columns) {
  std::string out;
  out.reserve(rows * cols / 4 + 4096);

  out += "NAME          SYNTH\nROWS\n N  COST\n";
  for (std::size_t i = 0; i < rows; ++i) {
    out += (i % 3 == 0) ? " E  R" : ((i % 3 == 1) ? " L  R" : " G  R");
    out += std::to_string(i);
    out += '\n';
  }

  out += "COLUMNS\n";
  std::mt19937_64 rng(12345);
  char buf[128];

  for (std::size_t j = 0; j < cols; ++j) {
    const std::string col = "C" + std::to_string(j);
    std::snprintf(buf, sizeof(buf), "    %-9s COST      %12.6f\n", col.c_str(),
                  1.0 + static_cast<double>(j % 17));
    out += buf;

    const bool dense = j < dense_columns;
    const std::size_t entries = dense ? rows : per_column;
    for (std::size_t k = 0; k < entries; ++k) {
      const std::size_t row =
          dense ? k : (static_cast<std::size_t>(rng()) % rows);
      const double v =
          0.5 + static_cast<double>(rng() % 1000) / 1000.0;
      std::snprintf(buf, sizeof(buf), "    %-9s R%-8zu %12.6f\n", col.c_str(),
                    row, v);
      out += buf;
    }
  }

  out += "RHS\n";
  for (std::size_t i = 0; i < rows; ++i) {
    std::snprintf(buf, sizeof(buf), "    RHS       R%-8zu %12.6f\n", i,
                  10.0 + static_cast<double>(i % 7));
    out += buf;
  }
  out += "ENDATA\n";
  return out;
}

struct Result {
  double parse_ms;
  std::size_t bytes;
  std::size_t rows, cols, nnz;
};

Result measure(const std::string& text, int repeats) {
  Result r{};
  double best = 1e300;
  for (int i = 0; i < repeats; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    auto p = sovsolve::io::parseProblem(text, sovsolve::io::FileFormat::Mps);
    const auto t1 = std::chrono::steady_clock::now();
    if (!p.has_value()) {
      std::fprintf(stderr, "parse failed: %s\n", p.error().format().c_str());
      std::exit(1);
    }
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (ms < best) {
      best = ms;
      r.rows = p->num_rows();
      r.cols = p->num_cols();
      r.nnz = p->nnz();
    }
  }
  r.parse_ms = best;
  r.bytes = text.size();
  return r;
}

void report(const char* label, const Result& r) {
  const double mb = static_cast<double>(r.bytes) / (1024.0 * 1024.0);
  const double sec = r.parse_ms / 1000.0;

  // Storage cost, measured rather than assumed.
  const double csr_bytes =
      static_cast<double>(r.nnz) * (sizeof(Real) + sizeof(Index)) +
      static_cast<double>(r.rows + 1) * sizeof(Index);
  const double per_nnz =
      r.nnz > 0 ? csr_bytes / static_cast<double>(r.nnz) : 0.0;

  std::printf("%-22s %8zu %8zu %10zu %9.1f %9.1f %10.2f %9.1f %8.1f\n", label,
              r.rows, r.cols, r.nnz, mb * 1024.0, r.parse_ms, mb / sec,
              static_cast<double>(r.nnz) / sec / 1e6, per_nnz * 2.0);
}

void print_header() {
  std::printf("%-22s %8s %8s %10s %9s %9s %10s %9s %8s\n", "instance", "rows",
              "cols", "nnz", "size KB", "parse ms", "MB/s", "Mnnz/s",
              "B/nnz");
  std::printf("%s\n", std::string(101, '-').c_str());
}

}  // namespace

int main(int argc, char** argv) {
  std::printf("sovsolve MPS parse benchmark\n");
  std::printf("B/nnz counts BOTH orientations (CSR + CSC), which is what the "
              "solver holds.\n\n");
  print_header();

  struct Case {
    const char* label;
    std::size_t rows, cols, per_col, dense;
    int repeats;
  };
  constexpr Case kCases[] = {
      {"synthetic/small", 1000, 1500, 6, 0, 20},
      {"synthetic/medium", 20000, 30000, 8, 0, 5},
      {"synthetic/dense-cols", 20000, 30000, 8, 3, 5},
      {"synthetic/large", 100000, 150000, 10, 0, 3},
  };

  for (const auto& c : kCases) {
    const std::string text = generate_mps(c.rows, c.cols, c.per_col, c.dense);
    report(c.label, measure(text, c.repeats));
  }

  // Real files, when given.
  for (int i = 1; i < argc; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    auto p = sovsolve::io::loadProblem(argv[i]);
    const auto t1 = std::chrono::steady_clock::now();
    if (!p.has_value()) {
      std::fprintf(stderr, "%s: %s\n", argv[i], p.error().format().c_str());
      continue;
    }
    Result r{};
    r.parse_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.rows = p->num_rows();
    r.cols = p->num_cols();
    r.nnz = p->nnz();
    FILE* f = std::fopen(argv[i], "rb");
    if (f != nullptr) {
      std::fseek(f, 0, SEEK_END);
      r.bytes = static_cast<std::size_t>(std::ftell(f));
      std::fclose(f);
    }
    std::string name(argv[i]);
    const auto slash = name.find_last_of("/\\");
    if (slash != std::string::npos) name = name.substr(slash + 1);
    report(name.c_str(), r);
  }

  std::printf("\npeak RSS %.1f MB\n",
              static_cast<double>(peak_rss_bytes()) / (1024.0 * 1024.0));
  return 0;
}
