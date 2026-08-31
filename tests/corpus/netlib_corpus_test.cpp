// Corpus test against real Netlib instances.
//
// The unit tests use hand-written models, which proves the parser handles the
// constructs it was told about. This proves it handles files written by other
// people's tools.
//
// Two layers of checking:
//
//   1. For instances whose dimensions are published with the Netlib
//      collection, the parsed counts are compared against them -- an external
//      oracle, not our own expectations.
//   2. For every `.mps` present in the data directory, structural invariants
//      are verified. That layer grows automatically as more instances are
//      fetched (scripts/fetch_netlib.py) without editing this file.
//
// Only afiro and adlittle are committed, so the test is meaningful with no
// network. The rest are gitignored.

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "sovsolve/analysis/MatrixAnalysis.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
namespace fs = std::filesystem;

namespace {

struct Published {
  const char* stem;
  const char* name;
  std::size_t rows;
  std::size_t cols;
  std::size_t nnz;
  std::size_t discrete;
};

// Published Netlib dimensions. `rows` excludes the objective row and `nnz`
// counts constraint-matrix entries only -- objective coefficients live in `c`,
// not in `A`.
//
// The header inside afiro.mps reads "N=32, M=28, NZ=115": that M counts the
// objective row and that NZ counts objective coefficients. Exactly the
// accounting confusion this table exists to pin down.
constexpr Published kPublished[] = {
    {"afiro",    "AFIRO",       27,   32,    83,   0},
    {"adlittle", ".Z....",      56,   97,   383,   0},
    {"25fv47",   "25FV47",     821, 1571, 10400,   0},
    {"80bau3b",  "80BAU3B",   2262, 9799, 21002,   0},
    {"e226",     "E226",       223,  282,  2578,   0},
    {"etamacro", "ETAMACRO",   400,  688,  2409,   0},
    {"israel",   "ISRAEL",     174,  142,  2269,   0},
    {"shell",    "SHELL",      536, 1775,  3556,   0},
    {"stair",    "STAIR",      356,  467,  3856,   0},
};

std::string data_dir() {
#ifdef SOVSOLVE_TEST_DATA_DIR
  return std::string(SOVSOLVE_TEST_DATA_DIR) + "/netlib";
#else
  return "tests/data/netlib";
#endif
}

/// Layer 1: compare against published counts.
void check_published(const Published& e, std::size_t& checked) {
  const std::string path = data_dir() + "/" + e.stem + ".mps";
  if (!fs::exists(path)) return;  // not fetched; layer 2 still covers presence
  ++checked;

  auto result = io::loadProblem(path);
  if (!result.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, e.stem,
                             result.error().format());
    return;
  }
  const auto& p = result.value();

  CHECK_EQ(p.num_rows(), e.rows);
  CHECK_EQ(p.num_cols(), e.cols);
  CHECK_EQ(p.nnz(), e.nnz);
  CHECK_EQ(p.num_discrete(), e.discrete);
}

/// Layer 2: structural invariants on every instance present.
void check_structure(const fs::path& path, std::size_t& parsed) {
  auto result = io::loadProblem(path.string());
  if (!result.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, path.filename().string().c_str(),
                             result.error().format());
    return;
  }
  ++parsed;
  const auto& p = result.value();

  // The invariants every downstream module relies on.
  CHECK(p.validate());
  CHECK(p.A.csr.validate());   // includes I2: sorted, duplicate-free
  CHECK(p.A.csc.validate());
  CHECK_EQ(p.A.csr.nnz(), p.A.csc.nnz());

  // Names must be complete, or diagnostics mislead.
  CHECK_EQ(p.col_names.size(), p.num_cols());
  CHECK_EQ(p.row_names.size(), p.num_rows());

  // Every bound pair is orderable and every row bound is too.
  bool bounds_ok = true;
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    if (p.col_lower[j] > p.col_upper[j]) bounds_ok = false;
  }
  for (std::size_t i = 0; i < p.num_rows(); ++i) {
    if (p.row_lower[i] > p.row_upper[i]) bounds_ok = false;
  }
  CHECK(bounds_ok);

  // A real objective row was identified, and no NaN or infinity slipped in.
  CHECK(!p.objective_row_name.empty());
  const auto a = analysis::analyze(p.A);
  CHECK(!a.has_invalid_values);
  CHECK_EQ(a.nnz, p.nnz());

  std::printf("  %-12s %5zu x %5zu  nnz %7zu  %-4s  disc %4zu  dense-col %zu\n",
              path.stem().string().c_str(), p.num_rows(), p.num_cols(), p.nnz(),
              p.has_discrete() ? "MILP" : "LP", p.num_discrete(),
              a.dense_columns.size());
}

void test_reparse_is_deterministic() {
  // Parsing the same file twice must give identical structure -- catches
  // uninitialized state and any dependence on allocation addresses.
  const std::string path = data_dir() + "/afiro.mps";
  auto a = io::loadProblem(path);
  auto b = io::loadProblem(path);
  CHECK(a.has_value() && b.has_value());
  if (!a.has_value() || !b.has_value()) return;

  CHECK_EQ(a->nnz(), b->nnz());
  const auto va = a->A.csr.values();
  const auto vb = b->A.csr.values();
  const auto ia = a->A.csr.indices();
  const auto ib = b->A.csr.indices();
  bool identical = true;
  for (std::size_t k = 0; k < a->nnz(); ++k) {
    if (va[k] != vb[k] || ia[k] != ib[k]) identical = false;
  }
  CHECK(identical);
}

void test_missing_file_reports_cleanly() {
  auto result = io::loadProblem(data_dir() + "/no_such_instance.mps");
  CHECK(!result.has_value());
  if (!result.has_value()) {
    CHECK(result.error().code == core::ErrorCode::FileNotFound);
    // The path must appear, or a batch failure is untraceable.
    CHECK(result.error().message.find("no_such_instance") != std::string::npos);
  }
}

}  // namespace

int main() {
  const fs::path dir = data_dir();
  if (!fs::exists(dir)) {
    std::printf("FAIL  netlib_corpus  (data directory %s missing)\n",
                dir.string().c_str());
    return 1;
  }

  std::size_t published_checked = 0;
  for (const auto& e : kPublished) check_published(e, published_checked);

  std::vector<fs::path> files;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.path().extension() == ".mps") files.push_back(entry.path());
  }
  std::sort(files.begin(), files.end());

  std::printf("netlib corpus (%zu instances, %zu with published dimensions):\n",
              files.size(), published_checked);
  std::size_t parsed = 0;
  for (const auto& f : files) check_structure(f, parsed);

  test_reparse_is_deterministic();
  test_missing_file_reports_cleanly();

  // afiro and adlittle are committed, so anything less means the data is gone.
  CHECK(parsed >= 2);
  return sovsolve::test::report("netlib_corpus");
}
