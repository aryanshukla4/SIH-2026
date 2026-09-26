#include "sovsolve/solver/ConcurrentSolve.hpp"

#include <atomic>
#include <chrono>
#include <limits>
#include <algorithm>
#include <thread>

#include "sovsolve/core/Cancel.hpp"
#include "sovsolve/solver/HomogeneousSolve.hpp"
#include "sovsolve/solver/pdlp/Pdlp.hpp"
#include "sovsolve/solver/pdlp/PdlpSolution.hpp"
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
  //   PDLP            first-order, factors NOTHING -- the one that keeps
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
  return {Method::DualSimplex, Method::Pdlp, Method::PrimalSimplex, Method::Hsd};
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

  if (report != nullptr) {
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
  return std::move(outcomes[chosen]);
}

}  // namespace sovsolve::solver
