#include "sovsolve/solver/MilpPresolve.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "sovsolve/core/SparseBuilder.hpp"

namespace sovsolve::solver {

namespace {

std::int64_t gcd_i64(std::int64_t a, std::int64_t b) noexcept {
  while (b != 0) {
    const auto t = b;
    b = a % b;
    a = t;
  }
  return a;
}

}  // namespace

core::Status tighten_integer_rows(model::Problem& problem) {
  using core::Real;
  using core::VarType;

  const auto& csr = problem.A.csr;
  const std::size_t m = problem.num_rows();

  for (std::size_t i = 0; i < m; ++i) {
    bool exploitable = true;
    std::int64_t g = 0;

    for (auto k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      const auto j = static_cast<std::size_t>(csr.indices()[k]);
      if (problem.col_type[j] == VarType::Continuous ||
          problem.col_type[j] == VarType::SemiContinuous) {
        exploitable = false;
        break;
      }
      const Real value = csr.values()[k];
      const Real rounded = std::round(value);
      // A fractional coefficient breaks the "always a multiple of g"
      // property the whole reduction depends on -- leave the row untouched
      // rather than tighten against an assumption that does not hold here.
      if (std::fabs(value - rounded) > 1e-9) {
        exploitable = false;
        break;
      }
      const auto coeff = static_cast<std::int64_t>(rounded);
      if (coeff == 0) continue;
      const std::int64_t mag = coeff < 0 ? -coeff : coeff;
      g = g == 0 ? mag : gcd_i64(g, mag);
    }

    if (!exploitable || g <= 1) continue;  // no exploitable common factor
    const Real gd = static_cast<Real>(g);

    if (core::is_finite_bound(problem.row_upper[i])) {
      problem.row_upper[i] = gd * std::floor(problem.row_upper[i] / gd + 1e-9);
    }
    if (core::is_finite_bound(problem.row_lower[i])) {
      problem.row_lower[i] = gd * std::ceil(problem.row_lower[i] / gd - 1e-9);
    }

    if (problem.row_lower[i] > problem.row_upper[i] + 1e-6) {
      return core::make_error(
          core::ErrorCode::PrimalInfeasible,
          "milp presolve: row " + std::to_string(i) +
              " has only integer columns with gcd " + std::to_string(g) +
              " -- no integer combination can satisfy its bounds");
    }
  }

  return core::Status::Ok();
}

void eliminate_equality_row_absorbing_singletons(
    model::Problem& problem, std::vector<AbsorbingColumnElimination>& eliminations) {
  using core::Real;
  using core::VarType;

  const auto& csr = problem.A.csr;
  const auto& csc = problem.A.csc;
  const std::size_t m = problem.num_rows();
  const std::size_t n = problem.num_cols();
  constexpr std::size_t kNoSkip = static_cast<std::size_t>(-1);

  std::vector<std::size_t> skip_col_for_row(m, kNoSkip);
  std::vector<Real> new_row_lower(m);
  std::vector<Real> new_row_upper(m);
  for (std::size_t i = 0; i < m; ++i) {
    new_row_lower[i] = problem.row_lower[i];
    new_row_upper[i] = problem.row_upper[i];
  }

  bool any = false;
  for (std::size_t i = 0; i < m; ++i) {
    const Real lo_i = problem.row_lower[i];
    const Real hi_i = problem.row_upper[i];
    // Only a genuine equality -- a ranged or one-sided row with an
    // absorbing column is a separate, rarer case not attempted here (see
    // the doc comment).
    if (!core::is_finite_bound(lo_i) || !core::is_finite_bound(hi_i) || lo_i != hi_i) continue;
    const Real b_i = hi_i;

    for (auto k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      const auto j = static_cast<std::size_t>(csr.indices()[k]);
      if (problem.col_type[j] != VarType::Continuous &&
          problem.col_type[j] != VarType::SemiContinuous) {
        continue;
      }
      const Real lo_j = problem.col_lower[j];
      const Real hi_j = problem.col_upper[j];
      const bool lo_finite = core::is_finite_bound(lo_j);
      const bool hi_finite = core::is_finite_bound(hi_j);
      if (lo_finite == hi_finite) continue;  // boxed or free -- not this rule's case
      if (csc.slice_nnz(j) != 1) continue;   // not a singleton -- constrained elsewhere too

      // x_j has no other constraint anywhere, so it can absorb any value in
      // its own one-sided range with no effect elsewhere -- the row is
      // equivalent to a derived ONE-SIDED bound purely on the REMAINING
      // columns, with x_j's own contribution range folded in. Four sign/
      // bound-side combinations collapse to one rule: whichever side of
      // R = (a_ij*x_j + R) - a_ij*x_j the finite bound pushes toward is the
      // side that becomes derivable; the other becomes unconstrained (-INF
      // or +INF), since x_j can grow without limit on its open side.
      const Real a_ij = csr.values()[k];
      const Real bound_val = lo_finite ? lo_j : hi_j;
      const Real contribution = a_ij * bound_val;
      const bool tighten_upper = (a_ij > 0.0 && lo_finite) || (a_ij < 0.0 && hi_finite);

      if (tighten_upper) {
        new_row_upper[i] = b_i - contribution;
        new_row_lower[i] = -core::INF;
      } else {
        new_row_lower[i] = b_i - contribution;
        new_row_upper[i] = core::INF;
      }

      // x_j is not an independently-chosen variable -- the row FORCES
      // `x_j = (b_i - R) / a_ij`, so its cost `c_j*x_j` is actually a
      // function of R, not a free term. Left un-folded, the existing
      // empty-column presolve rule would fix x_j at whichever bound
      // minimizes ITS OWN isolated cost, completely ignoring what the row
      // actually requires -- silently decoupling the deviation penalty
      // from the real constraint (confirmed: this was a real bug caught by
      // comparing markshare_4_0's reported objective, 0.0, against HiGHS's
      // true optimum, 1.0). Substituting `x_j = (b_i-R)/a_ij` into `c_j*x_j`
      // gives `c_j*b_i/a_ij - (c_j/a_ij)*R`, folded below the same way
      // RemoveFreeSingleton/RemoveFixedVariable already fold a substituted
      // column's cost elsewhere in this codebase. Once folded, c_j itself
      // is zeroed, so the now-empty column is fixed arbitrarily by the
      // EXISTING zero-cost empty-column rule -- correct, because its cost
      // has already been fully accounted for here.
      const Real fold_factor = problem.c[j] / a_ij;
      problem.obj_constant += fold_factor * b_i;
      for (auto k2 = csr.slice_begin(i); k2 < csr.slice_end(i); ++k2) {
        const auto col2 = static_cast<std::size_t>(csr.indices()[k2]);
        if (col2 == j) continue;
        problem.c[col2] -= fold_factor * csr.values()[k2];
      }
      problem.c[j] = 0.0;

      eliminations.push_back({i, j, a_ij, b_i});
      skip_col_for_row[i] = j;
      any = true;
      break;  // only one absorbing column removed per row
    }
  }

  if (!any) return;

  // Rebuild A skipping exactly the one marked entry per row. x_j itself is
  // NOT removed as a column -- with its only entry gone, it becomes a
  // genuinely empty column, and the EXISTING Presolver (Module 4,
  // Presolver.cpp) already fixes an empty column with a nonzero cost at
  // whichever bound improves the objective (or proves Unbounded), reusing
  // RemoveFixedVariable's already-tested recovery -- no new recovery math
  // needed here.
  core::SparseBuilder builder(m, n);
  for (std::size_t i = 0; i < m; ++i) {
    for (auto k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      const auto j = static_cast<std::size_t>(csr.indices()[k]);
      if (skip_col_for_row[i] == j) continue;
      builder.count(static_cast<core::Index>(i), static_cast<core::Index>(j));
    }
  }
  // Cannot fail: this rebuild only ever REMOVES entries from a matrix that
  // already fit within Index limits with strictly more of them. The check
  // is still made explicit rather than discarded, per this codebase's
  // policy on [[nodiscard]] Status returns.
  if (!builder.allocate().ok()) return;
  for (std::size_t i = 0; i < m; ++i) {
    for (auto k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      const auto j = static_cast<std::size_t>(csr.indices()[k]);
      if (skip_col_for_row[i] == j) continue;
      builder.insert(static_cast<core::Index>(i), static_cast<core::Index>(j), csr.values()[k]);
    }
  }
  problem.A = builder.finish();

  for (std::size_t i = 0; i < m; ++i) {
    problem.row_lower[i] = new_row_lower[i];
    problem.row_upper[i] = new_row_upper[i];
  }
}

}  // namespace sovsolve::solver
