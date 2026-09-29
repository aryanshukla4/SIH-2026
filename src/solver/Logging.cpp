#include "sovsolve/solver/Logging.hpp"

#include "sovsolve/solver/ConcurrentSolve.hpp"

namespace sovsolve::solver {

namespace {

/// Local to this file on purpose: Module 20 is presentation, so the spelling
/// of a status belongs here rather than in core, where it would invite every
/// other module to format its own output.
[[nodiscard]] const char* spell(core::SolverStatus s) {
  switch (s) {
    case core::SolverStatus::NotConverged: return "NotConverged";
    case core::SolverStatus::Optimal: return "Optimal";
    case core::SolverStatus::Infeasible: return "Infeasible";
    case core::SolverStatus::Unbounded: return "Unbounded";
    case core::SolverStatus::Nonconvex: return "Nonconvex";
    case core::SolverStatus::MaxIterations: return "MaxIterations";
    case core::SolverStatus::TimeLimit: return "TimeLimit";
    case core::SolverStatus::NumericalError: return "NumericalError";
  }
  return "unknown";
}

}  // namespace

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

void log_lp_presolve(const LpPresolveStats& s, const LogOptions& options, std::FILE* out) {
  if (options.level == LogOptions::Level::Silent) return;
  std::fprintf(out,
               "lp presolve: rows %zu -> %zu  cols %zu -> %zu  nnz %zu -> %zu  (%zu rounds)\n",
               s.rows_before, s.rows_after, s.cols_before, s.cols_after, s.nnz_before,
               s.nnz_after, s.rounds);
  std::fprintf(out,
               "  rows: empty %zu redundant %zu singleton %zu forcing %zu | cols: fixed %zu "
               "empty %zu free-singleton %zu implied-free %zu doubleton %zu dominated %zu "
               "parallel %zu\n",
               s.empty_rows, s.redundant_rows, s.singleton_rows, s.forcing_rows,
               s.fixed_columns, s.empty_columns, s.free_singletons, s.implied_free_singletons,
               s.doubletons, s.dominated_columns, s.parallel_columns);
}

void log_concurrent_race(const ConcurrentReport& report, const LogOptions& options,
                         std::FILE* out) {
  if (options.level == LogOptions::Level::Silent) return;

  std::fprintf(out, "concurrent: %zu engines on %zu threads\n",
               report.entries.size(), report.threads);
  for (const ConcurrentEntry& e : report.entries) {
    std::fprintf(out, "  %-16s %-14s %10.3fs  iters=%-8zu %s\n", e.name.c_str(),
                 e.failed ? "failed" : spell(e.status), e.seconds,
                 e.iterations, e.won ? "<-- winner" : "");
  }
  if (report.crossover_ran) {
    std::fprintf(out, "  %-16s %-14s %10.3fs  iters=%-8zu %s\n", "crossover",
                 spell(report.crossover_status), report.crossover_seconds,
                 report.crossover_iterations,
                 report.crossover_used ? "<-- vertex returned" : "(winner's answer kept)");
  }
}

}  // namespace sovsolve::solver
