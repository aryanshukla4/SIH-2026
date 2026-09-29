// Per-iteration record and the container that accumulates it.
//
// docs/spec/module.txt Module 19: theta_floor_activations and refinement_passes are
// "the evidence for any numerical-robustness claim in the submission" -- this
// is a benchmarking/grading artifact, not incidental logging, so it needs
// CSV/JSON export rather than printf.

#ifndef SOVSOLVE_SOLVER_DIAGNOSTICS_HPP
#define SOVSOLVE_SOLVER_DIAGNOSTICS_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "sovsolve/core/Types.hpp"

namespace sovsolve::solver {

using core::Real;
using core::SolverStatus;

/// Everything measured during one IPM iteration.
struct IterationRecord {
  std::size_t iteration = 0;
  Real objective = 0.0;

  Real primal_residual_inf = 0.0;
  Real dual_residual_inf = 0.0;
  Real xz_complementarity = 0.0;
  Real uv_complementarity = 0.0;
  Real sy_complementarity = 0.0;
  Real aggregate_complementarity = 0.0;

  Real mu = 0.0;
  Real mu_aff = 0.0;
  Real sigma = 0.0;

  Real alpha_primal = 0.0;
  Real alpha_dual = 0.0;

  double symbolic_time = 0.0;
  double factorization_time = 0.0;
  double solve_time = 0.0;
  double refinement_time = 0.0;
  double gpu_transfer_time = 0.0;

  std::size_t regularization_events = 0;
  std::size_t theta_floor_activations = 0;
  std::size_t refinement_passes = 0;

  double iteration_time = 0.0;
  SolverStatus status = SolverStatus::NotConverged;
};

/// Iteration history plus export. Independent of Logging (Logging.hpp) --
/// this is the data record, Logging is presentation, and docs/spec/module.txt Module 20
/// requires the two stay decoupled.
class Diagnostics {
 public:
  void record(const IterationRecord& r) { history_.push_back(r); }

  [[nodiscard]] const std::vector<IterationRecord>& history() const noexcept {
    return history_;
  }
  [[nodiscard]] std::size_t size() const noexcept { return history_.size(); }
  [[nodiscard]] bool empty() const noexcept { return history_.empty(); }

  /// Benchmarking ~98 Netlib instances plus a MIPLIB subset cannot be read
  /// off printf output -- these exist for that, not for interactive use.
  [[nodiscard]] std::string to_csv() const;
  [[nodiscard]] std::string to_json() const;

 private:
  std::vector<IterationRecord> history_;
};

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_DIAGNOSTICS_HPP
