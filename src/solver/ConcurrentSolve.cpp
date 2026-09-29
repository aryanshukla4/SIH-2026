#include "sovsolve/solver/ConcurrentSolve.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <algorithm>
#include <optional>
#include <thread>

#include "sovsolve/core/Cancel.hpp"
#include "sovsolve/solver/Crossover.hpp"
#include "sovsolve/solver/HomogeneousSolve.hpp"
#include "sovsolve/solver/pdlp/Pdlp.hpp"
#include "sovsolve/solver/pdlp/PdlpSolution.hpp"
#include "sovsolve/solver/simplex/Basis.hpp"
#include "sovsolve/solver/simplex/PrimalSimplex.hpp"
#include "sovsolve/solver/simplex/SimplexSolution.hpp"
#include "sovsolve/solver/simplex/SolveSimplex.hpp"

namespace sovsolve::solver {

namespace {

using core::Real;
using core::SolverStatus;
using model::Method;
using model::Options;
using model::Solution;

constexpr std::size_t kNoWinner = std::numeric_limits<std::size_t>::max();

/// Largest relative objective difference between the crossover's vertex and
/// the winner's answer at which the vertex is accepted. OURS: ten times the
/// 1e-8 relative stopping tolerance of cuPDLPx and HSD, so any vertex the
/// winner's own accuracy cannot tell apart from its answer is taken, and a
/// simplex vertex that has drifted further (measured 1.6e-6 to 2e-6 on
/// Netlib pilot and pilot87) is not.
constexpr Real kCrossoverAgreement = 1e-7;

[[nodiscard]] const char* method_name(Method m) {
  switch (m) {
    case Method::DualSimplex: return "dual-simplex";
    case Method::PrimalSimplex: return "primal-simplex";
    case Method::Pdlp: return "pdlp";
    case Method::PdlpX: return "pdlpx";
    case Method::Hsd: return "hsd";
    case Method::InteriorPoint: return "ipm";
    case Method::Concurrent: return "concurrent";
  }
  return "unknown";
}

/// A verdict is a statement about the MODEL, and any engine reaching one ends
/// the race. `MaxIterations`, `TimeLimit`, `NumericalError` and
/// `NotConverged` are statements about the ENGINE -- another entrant may
/// still succeed, so they do not.
[[nodiscard]] bool is_definitive(SolverStatus s) {
  return s == SolverStatus::Optimal || s == SolverStatus::Infeasible ||
         s == SolverStatus::Unbounded;
}

/// One entrant, start to finish, in canonical space.
[[nodiscard]] core::Expected<Solution> run_one(const model::CanonicalProblem& problem,
                                               const Options& base, Method method,
                                               const core::CancelToken& token) {
  Options o = base;
  o.cancel = &token;
  o.simplex.method = method;

  if (method == Method::Pdlp || method == Method::PdlpX) {
    o.pdlp.halpern = method == Method::PdlpX;
    auto r = pdlp::solve_pdlp(problem, o);
    if (!r.has_value()) return r.error();
    return pdlp::to_canonical_solution(problem, *r);
  }
  if (method == Method::Hsd) {
    auto r = solve_hsd(problem, o);
    if (!r.has_value()) return r.error();
    return to_canonical_solution(problem, *r);
  }
  auto r = simplex::solve_simplex(problem, o);
  if (!r.has_value()) return r.error();
  return simplex::to_canonical_solution(problem, *r);
}

/// An engine whose Optimal is a vertex. The others stop at a tolerance.
[[nodiscard]] bool ends_at_vertex(Method m) {
  return m == Method::DualSimplex || m == Method::PrimalSimplex;
}

}  // namespace

std::size_t concurrent_thread_budget(const Options& options) {
  if (options.concurrent.max_threads != 0) return options.concurrent.max_threads;
  // `hardware_concurrency` is allowed to return 0 when it cannot tell. Two is
  // the useful floor: one engine is not a race, and every machine this will
  // realistically run on has at least two cores.
  const unsigned hw = std::thread::hardware_concurrency();
  return hw == 0 ? 2u : static_cast<std::size_t>(hw);
}

std::vector<Method> default_concurrent_methods(const Options& options) {
  std::vector<Method> all = default_concurrent_methods();
  if (options.concurrent.include_gpu_interior_point) {
    all.push_back(Method::InteriorPoint);
  }
  // Truncate to what the machine can actually run at once. Racing more
  // engines than there are cores does not hedge, it time-slices: every
  // entrant runs slower and the winner finishes later than it would have
  // alone. The order in `default_concurrent_methods()` is a priority order
  // for exactly this reason.
  const std::size_t budget = concurrent_thread_budget(options);
  if (all.size() > budget) all.resize(std::max<std::size_t>(budget, 1));
  return all;
}

std::vector<Method> default_concurrent_methods() {
  // Four engines from four families, which is the point: racing two variants
  // of the same algorithm buys almost nothing, because they tend to be fast
  // and slow on the same models.
  //
  //   dual simplex    the usual winner on small and medium models
  //   primal simplex  wins where the dual stalls, e.g. after a bound change
  //   cuPDLPx         reflected-Halpern first-order method; factors NOTHING
  //                   and is the one that keeps
  //                   going on the largest and sparsest models, and stands
  //                   in for the barrier slot in Gurobi's line-up
  //   HSD             homogeneous self-dual interior point. Earns its thread
  //                   on INFEASIBLE and UNBOUNDED models: the embedding
  //                   produces those verdicts by construction, which is the
  //                   case a simplex can take longest to settle.
  //
  // The GPU interior-point engine is NOT here. It sits behind the one-way
  // solver -> gpu edge (src/solver/CMakeLists.txt) and is not reachable from
  // this layer; it would also rarely win, since FP64 factorization on a
  // consumer Ampere card runs at 1/64 rate (docs/ARCHITECTURE-REVIEW.md 3.5).
  //
  // PRIORITY ORDER, not a fixed set: the caller truncates this to the number
  // of threads the machine actually has, so a two-core box races the first
  // two and a large workstation races them all. Nothing here assumes a
  // particular core count.
  return {Method::DualSimplex, Method::PdlpX, Method::PrimalSimplex, Method::Hsd};
}

core::Expected<Solution> solve_concurrent(const model::CanonicalProblem& problem,
                                          const Options& options,
                                          const std::vector<Method>& methods,
                                          ConcurrentReport* report) {
  std::vector<Method> line_up =
      methods.empty() ? default_concurrent_methods(options) : methods;
  if (line_up.empty()) {
    return core::make_error(core::ErrorCode::NotImplemented,
                            "solve_concurrent: no engines to race");
  }

  const auto start = std::chrono::steady_clock::now();
  const auto since_start = [&start]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  };

  core::CancelToken token;
  const std::size_t n = line_up.size();

  std::vector<core::Expected<Solution>> outcomes;
  outcomes.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    outcomes.emplace_back(core::make_error(core::ErrorCode::NotImplemented, "not run"));
  }
  std::vector<double> seconds(n, 0.0);
  std::vector<SolverStatus> statuses(n, SolverStatus::NotConverged);

  // The first engine to reach a verdict claims the win. `compare_exchange`
  // makes "first" mean first in time rather than first in the line-up, which
  // is the whole point -- and it is why the winner is not deterministic when
  // two engines finish within the same instant.
  std::atomic<std::size_t> winner{kNoWinner};

  // Only the LAST entrant runs on this thread. Spawning a thread for it too
  // would leave the caller's core idle while it waits.
  std::vector<std::thread> workers;
  workers.reserve(n - 1);

  const auto body = [&](std::size_t i) {
    auto result = run_one(problem, options, line_up[i], token);
    seconds[i] = since_start();
    if (result.has_value()) {
      statuses[i] = result->status;
      if (is_definitive(result->status)) {
        std::size_t expected = kNoWinner;
        if (winner.compare_exchange_strong(expected, i, std::memory_order_acq_rel)) {
          token.cancel();  // tell everyone else to stop
        }
      }
    }
    outcomes[i] = std::move(result);
  };

  for (std::size_t i = 0; i + 1 < n; ++i) workers.emplace_back(body, i);
  body(n - 1);
  for (std::thread& t : workers) t.join();

  std::size_t chosen = winner.load(std::memory_order_acquire);
  if (chosen == kNoWinner) {
    // Nobody reached a verdict -- every engine hit a limit or failed. Report
    // the first that at least produced a solution object, so the caller sees
    // MaxIterations/TimeLimit rather than a bare error.
    for (std::size_t i = 0; i < n; ++i) {
      if (outcomes[i].has_value()) {
        chosen = i;
        break;
      }
    }
  }

  // Crossover (ConcurrentOptions::crossover): a win by an engine that stops at
  // a tolerance is finished at a vertex, so the answer does not depend on
  // which thread was quicker. Budgeted, and only ever an improvement: if the
  // simplex does not reach Optimal in time, the winner's answer stands.
  bool crossover_ran = false;
  bool crossover_used = false;
  SolverStatus crossover_status = SolverStatus::NotConverged;
  std::size_t crossover_iterations = 0;
  double crossover_seconds = 0.0;
  CrossoverStats crossover_stats;
  double crossover_cleanup_seconds = 0.0;
  Real crossover_disagreement = 0.0;
  std::optional<Solution> crossed;
  const bool won = winner.load(std::memory_order_acquire) != kNoWinner;
  if (options.concurrent.crossover && won && !ends_at_vertex(line_up[chosen]) &&
      outcomes[chosen]->status == SolverStatus::Optimal && problem.num_rows() > 0) {
    const double race_seconds = since_start();
    const double budget =
        std::max(options.concurrent.crossover_min_seconds,
                 options.concurrent.crossover_time_factor * race_seconds);
    const double allowed =
        std::min(budget, std::max(0.0, options.limits.time_limit_seconds - race_seconds));
    crossover_ran = allowed > 0.0;

    // Crash, primal and dual push (Crossover.hpp): a vertex basis.
    auto pushed = crossover_ran ? crossover_basis(problem, *outcomes[chosen],
                                                  options.simplex.pivot_tolerance, allowed,
                                                  &crossover_stats)
                                : core::Expected<simplex::Basis>(core::make_error(
                                      core::ErrorCode::NumericalError, "no time left"));
    // What is left of the budget. The simplex reads a time limit of 0 as "no
    // limit", so an exhausted budget must skip the cleanup, not pass 0.
    const double remaining = allowed - (since_start() - race_seconds);
    if (pushed.has_value() && remaining > 0.0) {
      Options o = options;
      o.simplex.method = options.concurrent.crossover_method;
      o.limits.time_limit_seconds = remaining;
      const double cleanup_start = since_start();
      auto r = o.simplex.method == Method::PrimalSimplex
                   ? simplex::solve_primal_simplex(problem, o, &*pushed)
                   : simplex::solve_simplex(problem, o, &*pushed);
      crossover_cleanup_seconds = since_start() - cleanup_start;
      if (r.has_value()) {
        crossover_status = r->status;
        crossover_iterations = r->iterations;
        if (r->status == SolverStatus::Optimal) {
          // The crossover may refine the winner's answer, never change it:
          // the two objectives must agree to within what the winner's own
          // stopping test allows. On the worst-conditioned models (Netlib
          // pilot, pilot87) the simplex's vertex lands 1.6e-6 to 2e-6 from
          // the true optimum while the interior point's answer is within
          // 1e-8 -- there the winner is the better answer and is kept.
          // Both objectives through the same canonical conversion, so an
          // objective offset counts the same on each side.
          Solution vertex = simplex::to_canonical_solution(problem, *r);
          const Real w_obj = outcomes[chosen]->objective;
          const Real disagreement =
              std::abs(vertex.objective - w_obj) / (1.0 + std::abs(w_obj));
          crossover_disagreement = disagreement;
          if (disagreement <= kCrossoverAgreement) {
            vertex.iterations = outcomes[chosen]->iterations + r->iterations;
            crossed = std::move(vertex);
            crossover_used = true;
          }
        }
      }
    }
    crossover_seconds = since_start() - race_seconds;
  }

  if (report != nullptr) {
    report->crossover_ran = crossover_ran;
    report->crossover_used = crossover_used;
    report->crossover_status = crossover_status;
    report->crossover_iterations = crossover_iterations;
    report->crossover_seconds = crossover_seconds;
    report->crossover_superbasic = crossover_stats.superbasic;
    report->crossover_push_pivots = crossover_stats.pivots;
    report->crossover_dual_superbasic = crossover_stats.dual_superbasic;
    report->crossover_dual_pivots = crossover_stats.dual_pivots;
    report->crossover_crash_seconds = crossover_stats.crash_seconds;
    report->crossover_repairs = crossover_stats.repairs;
    report->crossover_disagreement = crossover_disagreement;
    report->crossover_primal_push_seconds = crossover_stats.primal_push_seconds;
    report->crossover_dual_push_seconds = crossover_stats.dual_push_seconds;
    report->crossover_cleanup_seconds = crossover_cleanup_seconds;
    report->entries.clear();
    report->entries.reserve(n);
    report->threads = n;
    for (std::size_t i = 0; i < n; ++i) {
      ConcurrentEntry e;
      e.method = line_up[i];
      e.name = method_name(line_up[i]);
      e.seconds = seconds[i];
      e.failed = !outcomes[i].has_value();
      if (!e.failed) {
        e.status = outcomes[i]->status;
        e.objective = outcomes[i]->objective;
        e.iterations = outcomes[i]->iterations;
      }
      e.won = (i == chosen);
      report->entries.push_back(std::move(e));
    }
    report->winner = chosen == kNoWinner ? n : chosen;
  }

  if (chosen == kNoWinner) return outcomes[0].error();  // every entrant errored
  if (crossed.has_value()) return std::move(*crossed);
  return std::move(outcomes[chosen]);
}

}  // namespace sovsolve::solver
