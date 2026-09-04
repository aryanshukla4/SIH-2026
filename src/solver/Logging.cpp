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

}  // namespace sovsolve::solver
