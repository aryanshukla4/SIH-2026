// Round-trip test: parse -> write -> parse -> compare.
//
// This is the strongest check the reader has, because it does not depend on my
// expectations being correct. Hand-written assertions test the reader against
// the same understanding of the format that produced it; a round-trip over the
// real corpus tests it against itself, and any asymmetry between reader and
// writer shows up as a mismatch.
//
// It catches, specifically, the traps where reader and writer must apply the
// SAME convention in opposite directions:
//
//   * the objective constant, which MPS stores NEGATED on the objective row
//   * RANGES, whose four cases must invert exactly
//   * the `UP`-with-negative-value bound quirk
//   * QUADOBJ's triangle, which must not be doubled on the way back
//
// Getting any of those wrong in only one of the two produces a mismatch here;
// getting it wrong in BOTH would pass, which is why the reader also has its own
// hand-checked unit tests against values computed from the format spec.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"
#include "sovsolve/io/Write.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
using core::INF;
using core::is_finite_bound;
using core::Real;
namespace fs = std::filesystem;

namespace {

fs::path data_dir() {
#ifdef SOVSOLVE_TEST_DATA_DIR
  return fs::path(SOVSOLVE_TEST_DATA_DIR);
#else
  return fs::path("tests/data");
#endif
}

/// Compare two models field by field. Indices are directly comparable here:
/// the writer emits rows and columns in their existing order, so a re-read
/// preserves it. That is itself part of the contract being checked.
bool same_model(const char* label, const model::Problem& a,
                const model::Problem& b) {
  const auto fail = [&](const std::string& what) {
    ::sovsolve::test::record(__FILE__, __LINE__, label, what);
    return false;
  };

  if (a.num_rows() != b.num_rows() || a.num_cols() != b.num_cols()) {
    return fail("dimensions " + std::to_string(a.num_rows()) + "x" +
                std::to_string(a.num_cols()) + " -> " +
                std::to_string(b.num_rows()) + "x" +
                std::to_string(b.num_cols()));
  }
  if (a.nnz() != b.nnz()) {
    return fail("nnz " + std::to_string(a.nnz()) + " -> " +
                std::to_string(b.nnz()));
  }
  if (a.sense != b.sense) return fail("objective sense changed");
  if (a.obj_constant != b.obj_constant) {
    return fail("objective constant " + std::to_string(a.obj_constant) +
                " -> " + std::to_string(b.obj_constant) +
                " (the MPS negation trap)");
  }

  for (std::size_t j = 0; j < a.num_cols(); ++j) {
    if (a.c[j] != b.c[j]) {
      return fail("objective coefficient " + std::to_string(j) + " changed");
    }
    if (a.col_lower[j] != b.col_lower[j] || a.col_upper[j] != b.col_upper[j]) {
      return fail("bounds on column " + std::to_string(j) + " changed: [" +
                  std::to_string(a.col_lower[j]) + "," +
                  std::to_string(a.col_upper[j]) + "] -> [" +
                  std::to_string(b.col_lower[j]) + "," +
                  std::to_string(b.col_upper[j]) + "]");
    }
    if (a.col_type[j] != b.col_type[j]) {
      return fail("integrality of column " + std::to_string(j) + " changed");
    }
    if (a.col_names[j] != b.col_names[j]) {
      return fail("name of column " + std::to_string(j) + " changed");
    }
  }

  for (std::size_t i = 0; i < a.num_rows(); ++i) {
    if (a.row_lower[i] != b.row_lower[i] || a.row_upper[i] != b.row_upper[i]) {
      return fail("bounds on row " + std::to_string(i) + " changed: [" +
                  std::to_string(a.row_lower[i]) + "," +
                  std::to_string(a.row_upper[i]) + "] -> [" +
                  std::to_string(b.row_lower[i]) + "," +
                  std::to_string(b.row_upper[i]) + "]");
    }
    if (a.row_names[i] != b.row_names[i]) {
      return fail("name of row " + std::to_string(i) + " changed");
    }
  }

  // Coefficients, compared through CSR. Both sides carry the sorted-index
  // invariant, so a positional walk is valid and any difference in structure
  // shows up as a differing index rather than being silently tolerated.
  {
    const auto oa = a.A.csr.offsets();
    const auto ob = b.A.csr.offsets();
    const auto ia = a.A.csr.indices();
    const auto ib = b.A.csr.indices();
    const auto va = a.A.csr.values();
    const auto vb = b.A.csr.values();
    for (std::size_t i = 0; i <= a.num_rows(); ++i) {
      if (oa[i] != ob[i]) return fail("row offset " + std::to_string(i));
    }
    for (std::size_t k = 0; k < a.nnz(); ++k) {
      if (ia[k] != ib[k]) return fail("column index at nnz " + std::to_string(k));
      if (va[k] != vb[k]) {
        return fail("coefficient at nnz " + std::to_string(k) + ": " +
                    std::to_string(va[k]) + " -> " + std::to_string(vb[k]));
      }
    }
  }

  if (a.has_quadratic() != b.has_quadratic()) return fail("Q presence changed");
  if (a.has_quadratic()) {
    if (a.Q.nnz() != b.Q.nnz()) {
      return fail("Q nnz " + std::to_string(a.Q.nnz()) + " -> " +
                  std::to_string(b.Q.nnz()) + " (the triangle-doubling trap)");
    }
    const auto va = a.Q.csr.values();
    const auto vb = b.Q.csr.values();
    for (std::size_t k = 0; k < a.Q.nnz(); ++k) {
      if (va[k] != vb[k]) return fail("Q value at " + std::to_string(k));
    }
  }
  return true;
}

/// parse -> write -> parse, then compare.
bool roundtrip(const char* label, const model::Problem& original) {
  auto text = io::writeMpsToString(original);
  if (!text.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, label,
                             "write: " + text.error().format());
    return false;
  }
  auto reparsed = io::parseProblem(text.value(), io::FileFormat::Mps);
  if (!reparsed.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, label,
                             "re-parse: " + reparsed.error().format());
    return false;
  }
  return same_model(label, original, reparsed.value());
}

bool roundtrip_text(const char* label, std::string_view mps) {
  auto p = io::parseProblem(mps, io::FileFormat::Mps);
  if (!p.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, label,
                             "parse: " + p.error().format());
    return false;
  }
  return roundtrip(label, p.value());
}

// ---------------------------------------------------------------------------

void test_objective_constant_negation() {
  // The single sharpest MPS trap: an RHS entry on the objective row is the
  // NEGATED constant. Reader and writer must both apply it, or the sign flips
  // once per round-trip.
  CHECK(roundtrip_text("objective constant", R"(NAME          OBJC
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST         3.0   R1           1.0
RHS
    RHS       R1           5.0   COST        -7.0
ENDATA
)"));
  auto p = io::parseProblem(R"(NAME          OBJC
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST         3.0   R1           1.0
RHS
    RHS       R1           5.0   COST        -7.0
ENDATA
)", io::FileFormat::Mps);
  CHECK(p.has_value());
  if (!p.has_value()) return;
  CHECK_NEAR(p->obj_constant, 7.0, 1e-15);
  // And the emitted text must carry -7, not 7.
  auto text = io::writeMpsToString(p.value());
  CHECK(text.has_value());
  if (!text.has_value()) return;
  CHECK(text->find("-7") != std::string::npos);
}

void test_ranges_all_four_cases() {
  // Every RANGES case must invert exactly. These are the rows the reader's
  // table covers, written back out and read again.
  CHECK(roundtrip_text("ranges", R"(NAME          RNG
ROWS
 N  COST
 L  R1
 G  R2
 E  R3
 E  R4
COLUMNS
    X         COST         1.0   R1           1.0
    X         R2           1.0   R3           1.0
    X         R4           1.0
RHS
    RHS       R1          10.0   R2           2.0
    RHS       R3           5.0   R4           8.0
RANGES
    RNG       R1           6.0   R2           8.0
    RNG       R3           3.0   R4          -4.0
ENDATA
)"));
}

void test_ranged_rows_are_normalized_to_L_with_positive_range() {
  // The writer normalizes every ranged row to `L` with a POSITIVE range, and
  // that choice is load-bearing: the reader takes |R| for `L` and `G` rows, so
  // the sign is only significant for `E` rows. As long as ranged rows are
  // written as `L`, a sign error in the range width is unobservable -- which
  // means a round-trip alone cannot catch one. This pins the normalization, so
  // the day someone emits an `E` range instead, the sign starts mattering and
  // this test is what says so.
  //
  // Input has an `E` row with a NEGATIVE range, giving [b+r, b] = [1, 5].
  auto p = io::parseProblem(R"(NAME          RNGNORM
ROWS
 N  COST
 E  R1
COLUMNS
    X         COST         1.0   R1           1.0
RHS
    RHS       R1           5.0
RANGES
    RNG       R1          -4.0
ENDATA
)", io::FileFormat::Mps);
  CHECK(p.has_value());
  if (!p.has_value()) return;
  CHECK_NEAR(p->row_lower[0], 1.0, 1e-15);
  CHECK_NEAR(p->row_upper[0], 5.0, 1e-15);

  auto text = io::writeMpsToString(p.value());
  CHECK(text.has_value());
  if (!text.has_value()) return;

  // Emitted as `L` (not `E`), and the range is written positive.
  CHECK(text->find(" L ") != std::string::npos ||
        text->find("    L ") != std::string::npos);
  CHECK(text->find("RANGES") != std::string::npos);
  CHECK(text->find("-4") == std::string::npos);

  // And the bounds survive the normalization unchanged.
  CHECK(roundtrip("ranged normalization", p.value()));
}

void test_bound_forms() {
  CHECK(roundtrip_text("bounds", R"(NAME          BND
ROWS
 N  COST
 L  R1
COLUMNS
    A         COST         1.0   R1           1.0
    B         COST         1.0   R1           1.0
    C         COST         1.0   R1           1.0
    D         COST         1.0   R1           1.0
    E         COST         1.0   R1           1.0
    F         COST         1.0   R1           1.0
RHS
    RHS       R1         100.0
BOUNDS
 LO BND       A            2.0
 UP BND       B            7.0
 LO BND       C            3.0
 UP BND       C            9.0
 FX BND       D            4.0
 FR BND       E
 MI BND       F
 UP BND       F            5.0
ENDATA
)"));
}

void test_negative_upper_bound_quirk() {
  // `UP` with a negative value and no prior lower bound is the quirk readers
  // disagree on. The writer emits `MI` first whenever the lower bound is
  // infinite, so its own output never depends on which convention a reader
  // takes -- this asserts that the round-trip is stable regardless.
  CHECK(roundtrip_text("negative upper", R"(NAME          NEGUP
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST         1.0   R1           1.0
RHS
    RHS       R1          10.0
BOUNDS
 UP BND       X           -5.0
ENDATA
)"));
}

void test_integrality_markers() {
  CHECK(roundtrip_text("markers", R"(NAME          INTEG
ROWS
 N  COST
 L  R1
COLUMNS
    A         COST         1.0   R1           1.0
    MARKER                 'MARKER'                 'INTORG'
    B         COST         2.0   R1           1.0
    C         COST         3.0   R1           1.0
    MARKER                 'MARKER'                 'INTEND'
    D         COST         4.0   R1           1.0
RHS
    RHS       R1          10.0
BOUNDS
 UP BND       B            8.0
 UP BND       C            8.0
ENDATA
)"));
}

void test_quadratic_triangle_is_not_doubled() {
  // QUADOBJ carries the lower triangle and the reader mirrors it into full
  // symmetric storage. A writer that emits both triangles would double every
  // off-diagonal on the next read, which shows up here as an nnz mismatch.
  CHECK(roundtrip_text("quadobj", R"(NAME          QP
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST         1.0   R1           1.0
    Y         COST        -1.0   R1           1.0
RHS
    RHS       R1          10.0
QUADOBJ
    X         X            2.0
    X         Y            1.0
    Y         Y            3.0
ENDATA
)"));
}

void test_maximize_survives() {
  CHECK(roundtrip_text("maximize", R"(NAME          MAXP
OBJSENSE
    MAX
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST         3.0   R1           1.0
RHS
    RHS       R1          10.0   COST         7.0
ENDATA
)"));
}

void test_corpus_roundtrips() {
  // The real check: every Netlib instance, through the writer and back.
  const fs::path dir = data_dir() / "netlib";
  if (!fs::exists(dir)) {
    std::printf("  (corpus missing, skipped)\n");
    return;
  }
  std::vector<fs::path> files;
  for (const auto& e : fs::directory_iterator(dir)) {
    if (e.path().extension() == ".mps") files.push_back(e.path());
  }
  std::sort(files.begin(), files.end());

  std::size_t ok = 0;
  std::size_t total_bytes = 0;
  for (const auto& path : files) {
    auto loaded = io::loadProblem(path.string());
    CHECK(loaded.has_value());
    if (!loaded.has_value()) continue;

    const std::string label = path.stem().string();
    auto text = io::writeMpsToString(loaded.value());
    if (!text.has_value()) {
      ::sovsolve::test::record(__FILE__, __LINE__, label.c_str(),
                               "write: " + text.error().format());
      continue;
    }
    total_bytes += text->size();
    if (roundtrip(label.c_str(), loaded.value())) ++ok;
  }
  std::printf("  %zu of %zu instances round-tripped exactly (%zu KB written)\n",
              ok, files.size(), total_bytes / 1024);
  CHECK_EQ(ok, files.size());
}

void test_lp_source_roundtrips_through_mps() {
  // A model read from LP must survive the MPS writer too -- the two readers
  // have to agree on the Problem they produce, not just on their own formats.
  const fs::path path = data_dir() / "lp" / "blend.lp";
  if (!fs::exists(path)) return;
  auto loaded = io::loadProblem(path.string());
  CHECK(loaded.has_value());
  if (!loaded.has_value()) return;
  CHECK(roundtrip("lp source", loaded.value()));
}

}  // namespace

int main() {
  test_objective_constant_negation();
  test_ranges_all_four_cases();
  test_ranged_rows_are_normalized_to_L_with_positive_range();
  test_bound_forms();
  test_negative_upper_bound_quirk();
  test_integrality_markers();
  test_quadratic_triangle_is_not_doubled();
  test_maximize_survives();
  test_lp_source_roundtrips_through_mps();
  std::printf("round-trip over the Netlib corpus:\n");
  test_corpus_roundtrips();
  return sovsolve::test::report("roundtrip");
}
