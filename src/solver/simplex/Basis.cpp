#include "sovsolve/solver/simplex/Basis.hpp"

#include <cstddef>

namespace sovsolve::solver::simplex {

bool Basis::validate() const {
  const std::size_t total = status.size();
  const std::size_t rows = basic.size();
  if (total < rows) return false;

  std::size_t basic_count = 0;
  for (const VarStatus st : status) {
    if (st == VarStatus::Basic) ++basic_count;
  }
  if (basic_count != rows) return false;

  std::vector<bool> seen(total, false);
  for (const Index w : basic) {
    if (w < 0) return false;
    const auto index = static_cast<std::size_t>(w);
    if (index >= total) return false;
    if (status[index] != VarStatus::Basic) return false;
    if (seen[index]) return false;
    seen[index] = true;
  }
  return true;
}

Basis make_logical_basis(const AugmentedMatrix& matrix) {
  const std::size_t rows = matrix.num_rows();
  const std::size_t structural = matrix.num_structural();
  const std::size_t total = matrix.num_total();

  Basis basis;
  basis.status.assign(total, VarStatus::AtLower);
  basis.basic.resize(rows);

  for (std::size_t j = 0; j < structural; ++j) {
    const Real lo = matrix.lower(j);
    const Real up = matrix.upper(j);
    const bool lo_finite = core::is_finite_bound(lo);
    const bool up_finite = core::is_finite_bound(up);

    if (!lo_finite && !up_finite) {
      basis.status[j] = VarStatus::Free;
    } else if (lo_finite && up_finite && lo == up) {
      // The canonicalizer substitutes fixed columns out, so this should not
      // occur -- but presolve runs after it and this class is also used on
      // bound-tightened branch-and-bound nodes, where l == u is exactly what
      // branching a binary variable produces.
      basis.status[j] = VarStatus::Fixed;
    } else if (!lo_finite) {
      basis.status[j] = VarStatus::AtUpper;
    } else if (!up_finite) {
      basis.status[j] = VarStatus::AtLower;
    } else {
      // Both bounds finite: pick the dual-feasible one. `d_j == c_j` here
      // because `y == 0` under an all-logical basis, so `c_j >= 0` wants the
      // lower bound (`d_j >= 0`) and `c_j < 0` the upper (`d_j <= 0`).
      basis.status[j] = matrix.cost(j) >= 0.0 ? VarStatus::AtLower : VarStatus::AtUpper;
    }
  }

  for (std::size_t i = 0; i < rows; ++i) {
    const std::size_t w = matrix.logical_of_row(i);
    basis.status[w] = VarStatus::Basic;
    basis.basic[i] = static_cast<Index>(w);
  }
  return basis;
}

Real nonbasic_value(const AugmentedMatrix& matrix, std::size_t w,
                    VarStatus status) noexcept {
  switch (status) {
    case VarStatus::AtLower:
    case VarStatus::Fixed:
      return matrix.lower(w);
    case VarStatus::AtUpper:
      return matrix.upper(w);
    case VarStatus::Free:
      return 0.0;
    case VarStatus::Basic:
      break;
  }
  return 0.0;
}

}  // namespace sovsolve::solver::simplex
