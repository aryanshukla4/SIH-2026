#include "sovsolve/solver/LpSolve.hpp"

#include <chrono>

#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/Presolver.hpp"
#include "sovsolve/solver/Scaler.hpp"
#include "sovsolve/solver/SolutionReconstructor.hpp"
#include "sovsolve/solver/pdlp/Pdlp.hpp"
#include "sovsolve/solver/HomogeneousSolve.hpp"
#include "sovsolve/solver/pdlp/PdlpSolution.hpp"
#include "sovsolve/solver/simplex/SolveSimplex.hpp"
#include "sovsolve/solver/simplex/SimplexSolution.hpp"

namespace sovsolve::solver {

using core::Expected;
using model::Options;
using model::Problem;
using model::Solution;

namespace {

/// A verdict about the model, reported as an answer rather than as a failure.
///
/// `canonicalize()` and `presolve()` both detect infeasibility and
/// unboundedness by inspection and report them as `ErrorCode::PrimalInfeasible`
/// / `ErrorCode::Unbounded` -- correctly, since at that layer there is no
/// `Solution` to put a status on. At THIS layer there is, and a caller asking
/// "is this model feasible" should not have to distinguish "the answer is no"
/// from "the solver broke" by inspecting an error code. Everything else stays
/// an error.
Solution verdict(const Problem& problem, core::SolverStatus status) {
  Solution solution;
  solution.status = status;
  solution.x.resize(problem.num_cols());
  solution.x.assign(0.0);
  solution.s.resize(problem.num_rows());
  solution.s.assign(0.0);
  solution.y.resize(problem.num_rows());
  solution.y.assign(0.0);
  solution.z.resize(problem.num_cols());
  solution.z.assign(0.0);
  solution.v.resize(problem.num_cols());
  solution.v.assign(0.0);
  return solution;
}

[[nodiscard]] bool is_verdict(core::ErrorCode code) {
  return code == core::ErrorCode::PrimalInfeasible || code == core::ErrorCode::Unbounded;
}

[[nodiscard]] core::SolverStatus verdict_status(core::ErrorCode code) {
  return code == core::ErrorCode::PrimalInfeasible ? core::SolverStatus::Infeasible
                                                   : core::SolverStatus::Unbounded;
}

}  // namespace

Expected<Solution> solve_lp(const Problem& problem, const Options& options,
                            const MatVecProvider& matvec_provider) {
  const auto start = std::chrono::steady_clock::now();

  if (problem.has_quadratic()) {
    return core::make_error(core::ErrorCode::UnsupportedFeature,
                            "solve_lp: the simplex and PDLP engines solve linear "
                            "programs; this model has a quadratic objective");
  }

  auto canon = model::canonicalize(problem, options);
  if (!canon.has_value()) {
    if (is_verdict(canon.error().code)) {
      return verdict(problem, verdict_status(canon.error().code));
    }
    return canon.error();
  }

  core::Status status = presolve(canon->problem, options, canon->transforms);
  if (!status.ok()) {
    if (is_verdict(status.error().code)) {
      return verdict(problem, verdict_status(status.error().code));
    }
    return status.error();
  }

  status = scale(canon->problem, options, canon->transforms);
  if (!status.ok()) return status.error();

  // Four engines, one pipeline. Everything above and below this block is
  // shared; only the middle differs, and each engine is responsible for
  // producing the same five canonical-space vectors.
  Solution canonical;
  if (options.simplex.method == model::Method::Pdlp) {
    // The injected backend, when there is one. PDLP's only contact with the
    // matrix is through this, so the GPU path differs from the host path in
    // exactly one object and nothing else.
    pdlp::MatVec* matvec = nullptr;
    if (matvec_provider) {
      auto supplied = matvec_provider(canon->problem);
      if (!supplied.has_value()) return supplied.error();
      matvec = *supplied;
    }
    auto result = matvec ? pdlp::solve_pdlp(canon->problem, options, *matvec)
                         : pdlp::solve_pdlp(canon->problem, options);
    if (!result.has_value()) return result.error();
    canonical = pdlp::to_canonical_solution(canon->problem, *result);
  } else if (options.simplex.method == model::Method::Hsd) {
    auto result = solve_hsd(canon->problem, options);
    if (!result.has_value()) return result.error();
    canonical = to_canonical_solution(canon->problem, *result);
  } else {
    auto result = simplex::solve_simplex(canon->problem, options);
    if (!result.has_value()) return result.error();
    canonical = simplex::to_canonical_solution(canon->problem, *result);
  }

  auto recovered =
      reconstruct_solution(problem, canon->problem, canon->transforms, canonical);
  if (!recovered.has_value()) return recovered.error();

  const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
  recovered->solve_time_seconds = elapsed.count();
  return recovered;
}

}  // namespace sovsolve::solver
