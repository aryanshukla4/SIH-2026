#include "sovsolve/solver/Logging.hpp"

namespace sovsolve::solver {

void log_iteration(const IterationRecord& record, const LogOptions& options,
                    std::FILE* out) {
  if (options.level == LogOptions::Level::Silent) return;

  std::fprintf(out,
               "iter %3zu  obj % .8e  mu % .3e  mu_aff % .3e  sigma %.3f  rp %.3e  "
               "rd %.3e  ap %.3f  ad %.3f\n",
               record.iteration, record.objective, record.mu, record.mu_aff, record.sigma,
               record.primal_residual_inf, record.dual_residual_inf, record.alpha_primal,
               record.alpha_dual);

  if (options.level == LogOptions::Level::Debug) {
    std::fprintf(out,
                 "    factor %.3fms solve %.3fms refine %.3fms (passes=%zu) "
                 "theta_floor=%zu\n",
                 record.factorization_time * 1000.0, record.solve_time * 1000.0,
                 record.refinement_time * 1000.0, record.refinement_passes,
                 record.theta_floor_activations);
  }
}

void log_presolve_summary(std::size_t rows_before, std::size_t cols_before,
                          std::size_t nnz_before, std::size_t rows_after,
                          std::size_t cols_after, std::size_t nnz_after,
                          const LogOptions& options, std::FILE* out) {
  if (options.level == LogOptions::Level::Silent) return;

  std::fprintf(out,
               "presolve: rows %zu(-%zu)  cols %zu(-%zu)  nnz %zu(-%zu)\n",
               rows_after, rows_before - rows_after, cols_after, cols_before - cols_after,
               nnz_after, nnz_before - nnz_after);
}

}  // namespace sovsolve::solver
