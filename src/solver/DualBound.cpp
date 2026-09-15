#include "sovsolve/solver/DualBound.hpp"

#include <cmath>
#include <limits>

namespace sovsolve::solver {

namespace {

using core::is_finite_bound;

}  // namespace

core::Status compute_dual_bound(const model::CanonicalProblem& problem,
                                const core::RealVector& y, const core::RealVector& z,
                                const core::RealVector& v, DualBound& out) {
  const std::size_t n = problem.num_cols();
  const std::size_t m = problem.num_rows();
  const std::size_t m_e = problem.num_equality;

  if (y.size() != m || z.size() != n || v.size() != n) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "compute_dual_bound: dual vectors do not match the problem");
  }

  out = DualBound{};

  // Step 1: project onto the dual's SIGN constraints, before anything else.
  //
  // `z, v >= 0` and `y_I <= 0` are part of dual feasibility, and a first-order
  // method violates them slightly. Clamping here rather than after means the
  // residual below is computed against the point we will actually use, so the
  // repair covers the clamping too. Doing it the other way round leaves the
  // clamp uncorrected and the "valid" bound is then not valid.
  core::RealVector yc(m);
  for (std::size_t i = 0; i < m; ++i) {
    yc[i] = (i < m_e) ? y[i] : std::fmin(y[i], 0.0);
  }
  core::RealVector zc(n);
  core::RealVector vc(n);
  for (std::size_t j = 0; j < n; ++j) {
    zc[j] = std::fmax(z[j], 0.0);
    vc[j] = std::fmax(v[j], 0.0);
  }

  // Step 2: the dual equality residual, `r = c - A'y - z + v`. Read through the
  // CSC so each output entry is one contiguous sweep, as `pdlp::HostMatVec`
  // does for the same product.
  const auto& csc = problem.A.csc;
  Real residual_inf = 0.0;
  Real bound = 0.0;
  Real penalty = 0.0;

  for (std::size_t i = 0; i < m; ++i) bound += problem.b[i] * yc[i];

  for (std::size_t j = 0; j < n; ++j) {
    Real aty = 0.0;
    for (std::size_t k = csc.slice_begin(j); k < csc.slice_end(j); ++k) {
      aty += csc.values()[k] * yc[static_cast<std::size_t>(csc.indices()[k])];
    }
    const Real r = problem.c[j] - aty - zc[j] + vc[j];
    residual_inf = std::fmax(residual_inf, std::fabs(r));

    // Step 3: [SW] section 2's shift. `r+` goes onto the lower-bound dual and
    // `r-` onto the upper-bound dual, which restores `A'y + z - v = c` exactly
    // while keeping both nonnegative.
    const Real r_plus = r > 0.0 ? r : 0.0;
    const Real r_minus = r < 0.0 ? -r : 0.0;
    const Real z_final = zc[j] + r_plus;
    const Real v_final = vc[j] + r_minus;

    // Step 4: evaluate `l'z - u'v`, and notice where it cannot be evaluated.
    //
    // A dual variable for a bound that does not exist is not repairable: a free
    // column requires the reduced cost to be exactly zero, and no shift can
    // achieve that, because the shift is the thing that had to be nonzero. This
    // is the case [SW] wrote project-and-shift for.
    if (z_final > 0.0) {
      if (!is_finite_bound(problem.col_lower[j])) {
        ++out.unbounded_columns;
      } else {
        bound += problem.col_lower[j] * z_final;
        penalty += problem.col_lower[j] * r_plus;
      }
    }
    if (v_final > 0.0) {
      if (!is_finite_bound(problem.col_upper[j])) {
        ++out.unbounded_columns;
      } else {
        bound -= problem.col_upper[j] * v_final;
        penalty -= problem.col_upper[j] * r_minus;
      }
    }
  }

  out.residual_inf = residual_inf;
  out.correction_penalty = penalty;
  out.finite = (out.unbounded_columns == 0);
  out.bound = out.finite ? bound : -std::numeric_limits<Real>::infinity();
  return core::Status::Ok();
}

}  // namespace sovsolve::solver
