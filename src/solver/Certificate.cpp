#include "sovsolve/solver/Certificate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace sovsolve::solver {

Certificate certify(const model::Problem& problem, const model::Solution& solution) {
  using core::is_finite_bound;
  using core::Real;
  Certificate cert;
  const std::size_t m = problem.num_rows();
  const std::size_t n = problem.num_cols();
  if (solution.x.size() != n || solution.y.size() != m) return cert;
  cert.valid = true;

  const Real sigma = problem.sense == core::ObjSense::Maximize ? -1.0 : 1.0;
  const auto& csr = problem.A.csr;
  std::vector<Real> aty(n, 0.0);
  Real g = 0.0;
  for (std::size_t i = 0; i < m; ++i) {
    Real act = 0.0;
    Real y = solution.y[i];
    if (y > 0.0 && !is_finite_bound(problem.row_lower[i])) y = 0.0;
    if (y < 0.0 && !is_finite_bound(problem.row_upper[i])) y = 0.0;
    if (y > 0.0) g += y * problem.row_lower[i];
    if (y < 0.0) g += y * problem.row_upper[i];
    for (auto k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      const auto j = static_cast<std::size_t>(csr.indices()[k]);
      act += csr.values()[k] * solution.x[j];
      aty[j] += csr.values()[k] * y;
    }
    const Real lo = problem.row_lower[i], hi = problem.row_upper[i];
    if (is_finite_bound(lo)) cert.primal = std::max(cert.primal, (lo - act) / (1.0 + std::fabs(lo)));
    if (is_finite_bound(hi)) cert.primal = std::max(cert.primal, (act - hi) / (1.0 + std::fabs(hi)));
  }

  Real primal_obj = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    const Real d = sigma * problem.c[j] - aty[j];
    Real lambda = 0.0;
    if (d > 0.0 && is_finite_bound(problem.col_lower[j])) lambda = d;
    if (d < 0.0 && is_finite_bound(problem.col_upper[j])) lambda = d;
    if (lambda > 0.0) g += lambda * problem.col_lower[j];
    if (lambda < 0.0) g += lambda * problem.col_upper[j];
    cert.dual = std::max(cert.dual, std::fabs(d - lambda) / (1.0 + std::fabs(problem.c[j])));
    primal_obj += sigma * problem.c[j] * solution.x[j];
  }
  cert.gap = std::fabs(primal_obj - g) / (1.0 + std::fabs(primal_obj) + std::fabs(g));
  cert.dual_bound = sigma * g + problem.obj_constant;
  return cert;
}

}  // namespace sovsolve::solver
