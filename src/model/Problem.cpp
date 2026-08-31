#include "sovsolve/model/Problem.hpp"

#include <cstddef>

#include "sovsolve/core/Types.hpp"

namespace sovsolve::model {

bool Problem::validate() const noexcept {
  const std::size_t m = A.rows();
  const std::size_t n = A.cols();

  // Both orientations must describe the same matrix.
  if (A.csr.rows() != A.csc.rows() || A.csr.cols() != A.csc.cols()) return false;
  if (A.csr.nnz() != A.csc.nnz()) return false;
  if (!A.csr.validate() || !A.csc.validate()) return false;

  // Parallel array lengths.
  if (c.size() != n) return false;
  if (row_lower.size() != m || row_upper.size() != m) return false;
  if (col_lower.size() != n || col_upper.size() != n) return false;
  if (col_type.size() != n) return false;

  // Names, when present, must be complete rather than partial -- a half-filled
  // arena produces misleading diagnostics, which is worse than none.
  if (!row_names.empty() && row_names.size() != m) return false;
  if (!col_names.empty() && col_names.size() != n) return false;

  // Bound consistency. Note this is `>` and not `>=`: a fixed variable with
  // lower == upper is legal and common.
  for (std::size_t i = 0; i < m; ++i) {
    if (row_lower[i] > row_upper[i]) return false;
  }
  for (std::size_t j = 0; j < n; ++j) {
    if (col_lower[j] > col_upper[j]) return false;
  }

  if (!Q.empty()) {
    if (Q.rows() != n || Q.cols() != n) return false;
    if (!Q.csr.validate() || !Q.csc.validate()) return false;

    // Q is stored expanded to full symmetric, so the CSR and CSC views hold
    // the same entries in transposed order. Comparing the two orientations
    // checks symmetry without a second traversal of either.
    const auto r_off = Q.csr.offsets();
    const auto r_idx = Q.csr.indices();
    const auto r_val = Q.csr.values();
    const auto c_off = Q.csc.offsets();
    const auto c_idx = Q.csc.indices();
    const auto c_val = Q.csc.values();
    for (std::size_t i = 0; i < n; ++i) {
      if (r_off[i] != c_off[i]) return false;
      const auto b = static_cast<std::size_t>(r_off[i]);
      const auto e = static_cast<std::size_t>(r_off[i + 1]);
      for (std::size_t k = b; k < e; ++k) {
        if (r_idx[k] != c_idx[k]) return false;
        if (r_val[k] != c_val[k]) return false;
      }
    }
  }

  return true;
}

}  // namespace sovsolve::model
