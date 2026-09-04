#include "sovsolve/solver/Scaler.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace sovsolve::solver {

namespace {

/// One geometric-mean pass over `major` slices of `matrix` (rows of A_csr
/// when scaling rows, columns of A_csc when scaling columns), each entry
/// weighted by `other_scale[minor_index]` -- the scale already computed for
/// the OTHER dimension, so this pass reads its current value rather than a
/// stale snapshot from before that dimension was touched.
template <typename Matrix>
void geometric_mean_pass(const Matrix& matrix, const core::RealVector& other_scale,
                          core::RealVector& out_scale) {
  for (std::size_t major = 0; major < out_scale.size(); ++major) {
    Real min_val = 0.0;
    Real max_val = 0.0;
    bool any = false;
    for (std::size_t k = matrix.slice_begin(major); k < matrix.slice_end(major); ++k) {
      const auto minor = static_cast<std::size_t>(matrix.indices()[k]);
      const Real mag = std::fabs(matrix.values()[k]) * other_scale[minor];
      if (mag <= 0.0) continue;
      if (!any) {
        min_val = mag;
        max_val = mag;
        any = true;
      } else {
        min_val = std::min(min_val, mag);
        max_val = std::max(max_val, mag);
      }
    }
    out_scale[major] = any ? 1.0 / std::sqrt(min_val * max_val) : 1.0;
  }
}

}  // namespace

Status scale(CanonicalProblem& problem, const Options& /*options*/,
             TransformStack& transforms) {
  const std::size_t m = problem.num_rows();
  const std::size_t n = problem.num_cols();
  if (m == 0 || n == 0) return Status::Ok();

  core::RealVector row_scale(m, 1.0);
  core::RealVector col_scale(n, 1.0);

  constexpr int kPasses = 2;
  for (int pass = 0; pass < kPasses; ++pass) {
    geometric_mean_pass(problem.A.csr, col_scale, row_scale);
    geometric_mean_pass(problem.A.csc, row_scale, col_scale);
  }

  // -- apply to A, both orientations kept consistent -------------------------
  {
    auto& csr = problem.A.csr;
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
        const auto j = static_cast<std::size_t>(csr.indices()[k]);
        csr.values()[k] *= row_scale[i] * col_scale[j];
      }
    }
    auto& csc = problem.A.csc;
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t k = csc.slice_begin(j); k < csc.slice_end(j); ++k) {
        const auto i = static_cast<std::size_t>(csc.indices()[k]);
        csc.values()[k] *= row_scale[i] * col_scale[j];
      }
    }
  }

  // -- apply to Q: both indices are columns -----------------------------------
  if (!problem.Q.empty()) {
    auto& qcsr = problem.Q.csr;
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t k = qcsr.slice_begin(j); k < qcsr.slice_end(j); ++k) {
        const auto kk = static_cast<std::size_t>(qcsr.indices()[k]);
        qcsr.values()[k] *= col_scale[j] * col_scale[kk];
      }
    }
    auto& qcsc = problem.Q.csc;
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t k = qcsc.slice_begin(j); k < qcsc.slice_end(j); ++k) {
        const auto i = static_cast<std::size_t>(qcsc.indices()[k]);
        qcsc.values()[k] *= col_scale[i] * col_scale[j];
      }
    }
  }

  for (std::size_t i = 0; i < m; ++i) problem.b[i] *= row_scale[i];
  for (std::size_t j = 0; j < n; ++j) problem.c[j] *= col_scale[j];

  for (std::size_t j = 0; j < n; ++j) {
    if (core::is_finite_bound(problem.col_lower[j])) problem.col_lower[j] /= col_scale[j];
    if (core::is_finite_bound(problem.col_upper[j])) problem.col_upper[j] /= col_scale[j];
  }

  for (std::size_t i = 0; i < m; ++i) {
    transforms.push(TransformRecord{model::TransformKind::RowScaling,
                                    static_cast<core::Index>(i), TransformRecord::kIndexNone,
                                    row_scale[i], 0.0});
  }
  for (std::size_t j = 0; j < n; ++j) {
    transforms.push(TransformRecord{model::TransformKind::ColumnScaling,
                                    static_cast<core::Index>(j), TransformRecord::kIndexNone,
                                    col_scale[j], 0.0});
  }

  return Status::Ok();
}

}  // namespace sovsolve::solver
