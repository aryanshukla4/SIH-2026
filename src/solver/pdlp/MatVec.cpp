#include "sovsolve/solver/pdlp/MatVec.hpp"

namespace sovsolve::solver::pdlp {

void HostMatVec::multiply(core::HostSpan<const Real> x, core::HostSpan<Real> out) {
  const auto& csr = problem_->A.csr;
  const std::size_t m = problem_->num_rows();
  for (std::size_t i = 0; i < m; ++i) {
    Real acc = 0.0;
    for (std::size_t k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      acc += csr.values()[k] * x[static_cast<std::size_t>(csr.indices()[k])];
    }
    out[i] = acc;
  }
  ++products_;
}

void HostMatVec::multiply_transpose(core::HostSpan<const Real> y,
                                    core::HostSpan<Real> out) {
  // `K'y` component `j` is the dot of column `j` with `y`, so this reads the
  // CSC and accumulates per output entry -- not a scatter over the CSR, which
  // would write the same entry from many rows and serialize badly on a GPU.
  const auto& csc = problem_->A.csc;
  const std::size_t n = problem_->num_cols();
  for (std::size_t j = 0; j < n; ++j) {
    Real acc = 0.0;
    for (std::size_t k = csc.slice_begin(j); k < csc.slice_end(j); ++k) {
      acc += csc.values()[k] * y[static_cast<std::size_t>(csc.indices()[k])];
    }
    out[j] = acc;
  }
  ++products_;
}

}  // namespace sovsolve::solver::pdlp
