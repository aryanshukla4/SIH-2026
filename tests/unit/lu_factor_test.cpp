// Module 23, Stage 1: sparse LU of the basis, FTRAN/BTRAN, product-form update.
//
// Every solve here is checked against the DEFINITION of the basis matrix --
// `apply_basis` and `apply_basis_transpose` below multiply by `B` and `B'`
// directly out of `AugmentedMatrix`, with no reference to the factorization --
// rather than against a re-derivation of the same formulas the implementation
// uses. A sign or index-space error shared between implementation and test
// would otherwise pass.
//
// The two index spaces (`ftran` takes row space to slot space, `btran` slot
// space to row space; see LuFactor.hpp) are exactly where such an error would
// live, so the helpers are written to make the spaces explicit in their names.

#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "sovsolve/core/Span.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/simplex/Basis.hpp"
#include "sovsolve/solver/simplex/LuFactor.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
using core::Real;
using solver::simplex::AugmentedMatrix;
using solver::simplex::Basis;
using solver::simplex::LuFactorization;
using solver::simplex::make_logical_basis;
using solver::simplex::VarStatus;

namespace {

constexpr Real kPivotTolerance = 0.1;

model::CanonicalProblem canonical_from_mps(std::string_view text) {
  auto parsed = io::parseProblem(text, io::FileFormat::Mps);
  if (!parsed.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "parse", parsed.error().format());
    return {};
  }
  auto canon = model::canonicalize(parsed.value());
  if (!canon.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "canonicalize",
                             canon.error().format());
    return {};
  }
  return std::move(canon).value().problem;
}

/// `out = B * d`. `d` is slot space, `out` is row space.
std::vector<Real> apply_basis(const AugmentedMatrix& matrix, const Basis& basis,
                              const std::vector<Real>& d) {
  std::vector<Real> out(matrix.num_rows(), 0.0);
  for (std::size_t r = 0; r < basis.basic.size(); ++r) {
    const Real coeff = d[r];
    if (coeff == 0.0) continue;
    matrix.for_each_in_column(static_cast<std::size_t>(basis.basic[r]),
                              [&](std::size_t i, Real value) { out[i] += value * coeff; });
  }
  return out;
}

/// `out = B' * rho`. `rho` is row space, `out` is slot space.
std::vector<Real> apply_basis_transpose(const AugmentedMatrix& matrix, const Basis& basis,
                                        const std::vector<Real>& rho) {
  std::vector<Real> out(basis.basic.size(), 0.0);
  for (std::size_t r = 0; r < basis.basic.size(); ++r) {
    Real sum = 0.0;
    matrix.for_each_in_column(static_cast<std::size_t>(basis.basic[r]),
                              [&](std::size_t i, Real value) { sum += value * rho[i]; });
    out[r] = sum;
  }
  return out;
}

std::vector<Real> random_vector(std::size_t n, std::mt19937& rng) {
  std::uniform_real_distribution<Real> dist(-1.0, 1.0);
  std::vector<Real> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = dist(rng);
  return v;
}

Real max_abs_diff(const std::vector<Real>& a, const std::vector<Real>& b) {
  Real worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::fmax(worst, std::fabs(a[i] - b[i]));
  }
  return worst;
}

/// Round-trip both solves against the basis definition.
void check_solves(const char* label, const AugmentedMatrix& matrix, const Basis& basis,
                  const LuFactorization& lu, std::mt19937& rng, Real tolerance) {
  const std::size_t m = matrix.num_rows();

  // ftran: B * (B^-1 v) == v.
  std::vector<Real> rhs = random_vector(m, rng);
  std::vector<Real> solved = rhs;
  lu.ftran(core::HostSpan<Real>(solved.data(), solved.size()));
  const std::vector<Real> reconstructed = apply_basis(matrix, basis, solved);
  const Real ftran_error = max_abs_diff(reconstructed, rhs);
  ++::sovsolve::test::checks_run();
  if (!(ftran_error <= tolerance)) {
    ::sovsolve::test::record(__FILE__, __LINE__, "ftran round-trip",
                             std::string(label) + ": max |B*ftran(v) - v| = " +
                                 std::to_string(ftran_error));
  }

  // btran: B' * (B^-T v) == v.
  rhs = random_vector(m, rng);
  solved = rhs;
  lu.btran(core::HostSpan<Real>(solved.data(), solved.size()));
  const std::vector<Real> reconstructed_t = apply_basis_transpose(matrix, basis, solved);
  const Real btran_error = max_abs_diff(reconstructed_t, rhs);
  ++::sovsolve::test::checks_run();
  if (!(btran_error <= tolerance)) {
    ::sovsolve::test::record(__FILE__, __LINE__, "btran round-trip",
                             std::string(label) + ": max |B'*btran(v) - v| = " +
                                 std::to_string(btran_error));
  }
}

// --------------------------------------------------------------------------

/// The starting basis is the identity, so both solves must be the identity
/// map. Anything else means the logical columns are not being read as unit
/// vectors, which would be silently survivable on a square test but wrong
/// everywhere.
void test_logical_basis_is_identity() {
  const auto problem = canonical_from_mps(R"(NAME          TINY
ROWS
 N  COST
 E  R1
 L  R2
COLUMNS
    X1        COST      1.0        R1        1.0
    X1        R2        1.0
    X2        COST      1.0        R1        1.0
    X2        R2        2.0
RHS
    RHS       R1        10.0       R2        14.0
BOUNDS
 UP BND       X1        8.0
 UP BND       X2        8.0
ENDATA
)");
  if (problem.num_rows() == 0) return;

  const AugmentedMatrix matrix(problem);
  CHECK_EQ(matrix.num_rows(), std::size_t{2});
  CHECK_EQ(matrix.num_structural(), std::size_t{2});
  CHECK_EQ(matrix.num_total(), std::size_t{4});

  // Equality row's logical is fixed at zero; the inequality row's is the
  // canonical slack, non-negative and unbounded above.
  CHECK_EQ(problem.num_equality, std::size_t{1});
  CHECK_NEAR(matrix.upper(matrix.logical_of_row(0)), 0.0, 1e-15);
  CHECK(core::is_pos_infinite(matrix.upper(matrix.logical_of_row(1))));
  CHECK_NEAR(matrix.lower(matrix.logical_of_row(1)), 0.0, 1e-15);

  const Basis basis = make_logical_basis(matrix);
  CHECK(basis.validate());
  CHECK_EQ(basis.basic.size(), std::size_t{2});
  CHECK(basis.status[matrix.logical_of_row(0)] == VarStatus::Basic);

  LuFactorization lu;
  const auto status = lu.factorize(matrix, basis, kPivotTolerance);
  CHECK(status.ok());
  if (!status.ok()) return;
  CHECK(lu.valid());
  CHECK_EQ(lu.num_updates(), std::size_t{0});

  std::vector<Real> v{3.0, -5.0};
  lu.ftran(core::HostSpan<Real>(v.data(), v.size()));
  CHECK_NEAR(v[0], 3.0, 1e-15);
  CHECK_NEAR(v[1], -5.0, 1e-15);

  std::vector<Real> w{2.0, 7.0};
  lu.btran(core::HostSpan<Real>(w.data(), w.size()));
  CHECK_NEAR(w[0], 2.0, 1e-15);
  CHECK_NEAR(w[1], 7.0, 1e-15);
}

/// A 3x3 basis over structural columns whose inverse is known by hand, so the
/// index spaces are pinned to specific numbers rather than only to a
/// self-consistent round trip.
void test_known_small_basis() {
  const auto problem = canonical_from_mps(R"(NAME          SMALL
ROWS
 N  COST
 E  R1
 E  R2
 E  R3
COLUMNS
    X1        COST      1.0        R1        2.0
    X1        R2        1.0
    X2        COST      1.0        R2        3.0
    X2        R3        1.0
    X3        COST      1.0        R1        1.0
    X3        R3        4.0
RHS
    RHS       R1        1.0        R2        2.0
    RHS       R3        3.0
ENDATA
)");
  if (problem.num_rows() != 3) {
    ::sovsolve::test::record(__FILE__, __LINE__, "small basis shape",
                             "expected 3 rows, got " +
                                 std::to_string(problem.num_rows()));
    return;
  }

  const AugmentedMatrix matrix(problem);
  Basis basis = make_logical_basis(matrix);
  // Make all three structural columns basic, displacing every logical.
  for (std::size_t r = 0; r < 3; ++r) {
    basis.status[static_cast<std::size_t>(basis.basic[r])] = VarStatus::AtLower;
    basis.basic[r] = static_cast<core::Index>(r);
    basis.status[r] = VarStatus::Basic;
  }
  CHECK(basis.validate());

  LuFactorization lu;
  const auto status = lu.factorize(matrix, basis, kPivotTolerance);
  CHECK(status.ok());
  if (!status.ok()) return;

  //      [2 0 1]
  //  B = [1 3 0]       B * (1, 1, 1)' = (3, 4, 5)'
  //      [0 1 4]
  std::vector<Real> v{3.0, 4.0, 5.0};
  lu.ftran(core::HostSpan<Real>(v.data(), v.size()));
  CHECK_NEAR(v[0], 1.0, 1e-12);
  CHECK_NEAR(v[1], 1.0, 1e-12);
  CHECK_NEAR(v[2], 1.0, 1e-12);

  //  B' * (1, 1, 1)' = (3, 4, 5)' as well only by coincidence of this matrix;
  //  compute it explicitly: columns of B' are rows of B, so
  //  B' * (1,1,1)' = (2+0+1, 1+3+0, 0+1+4)' = (3, 4, 5)'.
  std::vector<Real> w{3.0, 4.0, 5.0};
  lu.btran(core::HostSpan<Real>(w.data(), w.size()));
  CHECK_NEAR(w[0], 1.0, 1e-12);
  CHECK_NEAR(w[1], 1.0, 1e-12);
  CHECK_NEAR(w[2], 1.0, 1e-12);
}

/// A basis with a duplicated column is singular. It must be REPORTED, not
/// silently factorized into garbage -- a bound-tightened branch-and-bound node
/// can produce one, so the caller has to be able to tell the difference.
void test_singular_basis_is_reported() {
  const auto problem = canonical_from_mps(R"(NAME          SING
ROWS
 N  COST
 E  R1
 E  R2
COLUMNS
    X1        COST      1.0        R1        1.0
    X1        R2        2.0
    X2        COST      1.0        R1        2.0
    X2        R2        4.0
RHS
    RHS       R1        1.0        R2        2.0
ENDATA
)");
  if (problem.num_rows() != 2) return;

  const AugmentedMatrix matrix(problem);
  Basis basis = make_logical_basis(matrix);
  // X2 is exactly 2*X1, so {X1, X2} has rank 1.
  for (std::size_t r = 0; r < 2; ++r) {
    basis.status[static_cast<std::size_t>(basis.basic[r])] = VarStatus::AtLower;
    basis.basic[r] = static_cast<core::Index>(r);
    basis.status[r] = VarStatus::Basic;
  }
  CHECK(basis.validate());

  LuFactorization lu;
  const auto status = lu.factorize(matrix, basis, kPivotTolerance);
  CHECK(!status.ok());
  if (!status.ok()) {
    CHECK(status.error().code == core::ErrorCode::NumericalError);
  }
  CHECK(!lu.valid());
}

/// Drive real pivots on a real instance: build a mixed structural/logical
/// basis by entering columns one at a time, updating the factorization with a
/// product-form eta each time, and re-checking both solves against the basis
/// definition after every pivot.
///
/// This is the test that would catch an eta applied in the wrong order --
/// forward in `btran` or reverse in `ftran` -- which a single pivot cannot
/// distinguish.
void test_pivots_on_real_instance() {
  auto loaded = io::loadProblem(std::string(SOVSOLVE_TEST_DATA_DIR) + "/netlib/afiro.mps");
  if (!loaded.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "load afiro.mps",
                             loaded.error().format());
    return;
  }
  auto canon = model::canonicalize(loaded.value());
  if (!canon.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "canonicalize afiro",
                             canon.error().format());
    return;
  }
  const model::CanonicalProblem problem = std::move(canon).value().problem;

  const AugmentedMatrix matrix(problem);
  const std::size_t m = matrix.num_rows();
  CHECK(m > 0);
  if (m == 0) return;

  Basis basis = make_logical_basis(matrix);
  LuFactorization lu;
  auto status = lu.factorize(matrix, basis, kPivotTolerance);
  CHECK(status.ok());
  if (!status.ok()) return;

  std::mt19937 rng(20260913u);
  check_solves("afiro, logical basis", matrix, basis, lu, rng, 1e-9);

  std::size_t pivots = 0;
  std::vector<Real> column(m, 0.0);
  for (std::size_t j = 0; j < matrix.num_structural() && pivots < 12; ++j) {
    // alpha = B^-1 * Ahat_j, in slot space.
    for (std::size_t i = 0; i < m; ++i) column[i] = 0.0;
    matrix.for_each_in_column(j, [&](std::size_t i, Real value) { column[i] = value; });
    lu.ftran(core::HostSpan<Real>(column.data(), column.size()));

    // Leave whichever logical still in the basis has the largest pivot: any
    // choice keeps the basis nonsingular, and the largest is the stable one.
    std::size_t leaving = m;
    Real best = 1e-6;
    for (std::size_t r = 0; r < m; ++r) {
      if (!matrix.is_logical(static_cast<std::size_t>(basis.basic[r]))) continue;
      if (std::fabs(column[r]) > best) {
        best = std::fabs(column[r]);
        leaving = r;
      }
    }
    if (leaving == m) continue;

    status = lu.update(leaving, core::HostSpan<const Real>(column.data(), column.size()));
    CHECK(status.ok());
    if (!status.ok()) return;

    basis.status[static_cast<std::size_t>(basis.basic[leaving])] = VarStatus::AtLower;
    basis.basic[leaving] = static_cast<core::Index>(j);
    basis.status[j] = VarStatus::Basic;
    ++pivots;

    CHECK(basis.validate());
    CHECK_EQ(lu.num_updates(), pivots);
    check_solves("afiro, after pivot", matrix, basis, lu, rng, 1e-8);
  }

  CHECK(pivots > 0);

  // A fresh factorization of the pivoted basis must agree with the updated
  // one. If it does not, the eta file has drifted from the basis it claims to
  // represent -- the failure mode a refactorization interval exists to bound.
  LuFactorization fresh;
  status = fresh.factorize(matrix, basis, kPivotTolerance);
  CHECK(status.ok());
  if (!status.ok()) return;
  CHECK_EQ(fresh.num_updates(), std::size_t{0});

  std::vector<Real> rhs = random_vector(m, rng);
  std::vector<Real> a = rhs;
  std::vector<Real> b = rhs;
  lu.ftran(core::HostSpan<Real>(a.data(), a.size()));
  fresh.ftran(core::HostSpan<Real>(b.data(), b.size()));
  const Real ftran_gap = max_abs_diff(a, b);
  ++::sovsolve::test::checks_run();
  if (!(ftran_gap <= 1e-8)) {
    ::sovsolve::test::record(__FILE__, __LINE__, "updated vs fresh ftran",
                             "max difference " + std::to_string(ftran_gap));
  }

  rhs = random_vector(m, rng);
  a = rhs;
  b = rhs;
  lu.btran(core::HostSpan<Real>(a.data(), a.size()));
  fresh.btran(core::HostSpan<Real>(b.data(), b.size()));
  const Real btran_gap = max_abs_diff(a, b);
  ++::sovsolve::test::checks_run();
  if (!(btran_gap <= 1e-8)) {
    ::sovsolve::test::record(__FILE__, __LINE__, "updated vs fresh btran",
                             "max difference " + std::to_string(btran_gap));
  }
}

/// Markowitz ordering has to survive a basis whose natural order is the worst
/// possible one. An arrowhead matrix pivoted in index order fills in
/// completely; pivoted on its singletons first it produces no fill at all.
void test_arrowhead_ordering() {
  std::string mps = "NAME          ARROW\nROWS\n N  COST\n";
  const std::size_t n = 40;
  for (std::size_t i = 0; i < n; ++i) mps += " E  R" + std::to_string(i) + "\n";
  mps += "COLUMNS\n";
  // Column 0 is dense; column i>0 touches row 0 and row i.
  mps += "    X0        COST      1.0";
  for (std::size_t i = 0; i < n; ++i) {
    mps += "\n    X0        R" + std::to_string(i) + "        1.0";
  }
  mps += "\n";
  for (std::size_t j = 1; j < n; ++j) {
    const std::string x = "X" + std::to_string(j);
    mps += "    " + x + "        COST      1.0\n";
    mps += "    " + x + "        R0        1.0\n";
    mps += "    " + x + "        R" + std::to_string(j) + "        2.0\n";
  }
  mps += "RHS\n";
  for (std::size_t i = 0; i < n; ++i) {
    mps += "    RHS       R" + std::to_string(i) + "        1.0\n";
  }
  mps += "ENDATA\n";

  const auto problem = canonical_from_mps(mps);
  if (problem.num_rows() != n) {
    ::sovsolve::test::record(__FILE__, __LINE__, "arrowhead shape",
                             "expected " + std::to_string(n) + " rows, got " +
                                 std::to_string(problem.num_rows()));
    return;
  }

  const AugmentedMatrix matrix(problem);
  Basis basis = make_logical_basis(matrix);
  for (std::size_t r = 0; r < n; ++r) {
    basis.status[static_cast<std::size_t>(basis.basic[r])] = VarStatus::AtLower;
    basis.basic[r] = static_cast<core::Index>(r);
    basis.status[r] = VarStatus::Basic;
  }
  CHECK(basis.validate());

  LuFactorization lu;
  const auto status = lu.factorize(matrix, basis, kPivotTolerance);
  CHECK(status.ok());
  if (!status.ok()) return;

  std::mt19937 rng(7u);
  check_solves("arrowhead", matrix, basis, lu, rng, 1e-9);

  // The matrix has 3n - 2 nonzeros. A fill-oblivious order would make the
  // factors O(n^2); the Markowitz order keeps them proportional to the input.
  // The bound is deliberately loose -- this asserts "ordering happened at all",
  // not a specific pivot sequence.
  const std::size_t input_nnz = 3 * n - 2;
  ++::sovsolve::test::checks_run();
  if (!(lu.factor_nnz() <= 4 * input_nnz)) {
    ::sovsolve::test::record(__FILE__, __LINE__, "arrowhead fill",
                             "factor_nnz = " + std::to_string(lu.factor_nnz()) +
                                 " against input nnz " + std::to_string(input_nnz));
  }
}

}  // namespace

int main() {
  test_logical_basis_is_identity();
  test_known_small_basis();
  test_singular_basis_is_reported();
  test_pivots_on_real_instance();
  test_arrowhead_ordering();
  return sovsolve::test::report("lu_factor");
}
