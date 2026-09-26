#include "sovsolve/solver/LpSolve.hpp"

#include <chrono>

#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/ConcurrentSolve.hpp"
#include "sovsolve/solver/Logging.hpp"
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
                            const PdlpBackendProvider& backend_provider) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();

  // Stage timings. Wall time on this thread, recorded so that "presolve is a
  // bottleneck" is a measurement rather than a belief -- nothing in this
  // project timed the stages before, so the fraction each one costs was
  // simply unknown.
  auto mark = Clock::now();
  const auto lap = [&mark]() {
    const auto now = Clock::now();
    const double seconds = std::chrono::duration<double>(now - mark).count();
    mark = now;
    return seconds;
  };
  double canonicalize_seconds = 0.0, presolve_seconds = 0.0;
  double scale_seconds = 0.0, engine_seconds = 0.0;

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
  canonicalize_seconds = lap();

  core::Status status = presolve(canon->problem, options, canon->transforms);
  if (!status.ok()) {
    if (is_verdict(status.error().code)) {
      return verdict(problem, verdict_status(status.error().code));
    }
    return status.error();
  }
  presolve_seconds = lap();

  status = scale(canon->problem, options, canon->transforms);
  if (!status.ok()) return status.error();
  scale_seconds = lap();

  // The preconditioner's composite factors, `R` and `S` in `A~ = R A S`, so
  // PDLP can measure its termination criteria on the ORIGINAL problem
  // (cuPDLPx section 4; PdlpOptions::terminate_on_original). Read back from
  // the transform stack rather than returned by `scale()` so that nothing
  // about scaling's interface changes for the engines that do not need it.
  // Multiplied, not assigned, in case scaling was ever applied twice. Owned
  // HERE, for the whole call; the options below only borrow them.
  std::vector<Real> row_scale(canon->problem.num_rows(), 1.0);
  std::vector<Real> col_scale(canon->problem.num_cols(), 1.0);
  for (const auto& rec : canon->transforms.records()) {
    if (rec.kind == model::TransformKind::RowScaling) {
      const auto i = static_cast<std::size_t>(rec.primary);
      if (i < row_scale.size()) row_scale[i] *= rec.value;
    } else if (rec.kind == model::TransformKind::ColumnScaling) {
      const auto j = static_cast<std::size_t>(rec.primary);
      if (j < col_scale.size()) col_scale[j] *= rec.value;
    }
  }
  model::Options scaled_options = options;
  scaled_options.pdlp.original_row_scale = {row_scale.data(), row_scale.size()};
  scaled_options.pdlp.original_col_scale = {col_scale.data(), col_scale.size()};

  // Four engines, one pipeline. Everything above and below this block is
  // shared; only the middle differs, and each engine is responsible for
  // producing the same five canonical-space vectors.
  Solution canonical;
  if (options.simplex.method == model::Method::Concurrent) {
    // Module 30: race several engines and keep the first verdict. The GPU
    // PDLP backend is deliberately NOT forwarded -- one device shared by
    // three racing threads would serialize them on the same hardware, which
    // is the opposite of the point.
    ConcurrentReport report;
    auto result = solve_concurrent(canon->problem, scaled_options, {}, &report);
    log_concurrent_race(report, options.log);
    if (!result.has_value()) return result.error();
    canonical = std::move(*result);
  } else if (options.simplex.method == model::Method::Pdlp ||
             options.simplex.method == model::Method::PdlpX) {
    // The injected backend, when there is one. PDLP's only contact with the
    // matrix is through this, so the GPU path differs from the host path in
    // exactly one object and nothing else.
    PdlpBackends backends;
    if (backend_provider) {
      auto supplied = backend_provider(canon->problem);
      if (!supplied.has_value()) return supplied.error();
      backends = *supplied;
    }
    // Module 31. `PdlpX` is one engine with one flag set, not a second
    // engine: a copy of the options here keeps the caller's `Options` a
    // `const&` all the way down and keeps the two schemes sharing every line
    // of setup, termination and postprocessing.
    model::Options pdlp_options = scaled_options;
    pdlp_options.pdlp.halpern = options.simplex.method == model::Method::PdlpX;
    const model::Options& active = pdlp_options;
    auto result =
        backends.matvec && backends.iteration
            ? pdlp::solve_pdlp(canon->problem, active, *backends.matvec,
                               *backends.iteration)
        : backends.matvec ? pdlp::solve_pdlp(canon->problem, active, *backends.matvec)
                          : pdlp::solve_pdlp(canon->problem, active);
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

  engine_seconds = lap();

  auto recovered =
      reconstruct_solution(problem, canon->problem, canon->transforms, canonical);
  if (!recovered.has_value()) return recovered.error();

  const std::chrono::duration<double> elapsed = Clock::now() - start;
  recovered->solve_time_seconds = elapsed.count();
  recovered->canonicalize_seconds = canonicalize_seconds;
  recovered->presolve_seconds = presolve_seconds;
  recovered->scale_seconds = scale_seconds;
  recovered->engine_seconds = engine_seconds;
  return recovered;
}

}  // namespace sovsolve::solver
