// Module 30: the concurrent optimizer -- several LP engines, one model, the
// first finisher wins.
//
// ---------------------------------------------------------------------------
// Why racing rather than predicting
// ---------------------------------------------------------------------------
//
// PS 26119 requires "multi-core parallelization". The obvious reading -- make
// one engine use every core -- does not survive contact with the algorithms:
// a simplex pivot depends on the basis the previous pivot produced, so the
// iteration is a sequential chain. That is not a limitation of this
// implementation; it is why essentially every solver's simplex is
// single-threaded.
//
// What multiple cores CAN do for an LP is run different algorithms at once.
// This is what Gurobi ships as the DEFAULT for LP models -- its concurrent
// optimizer devotes one thread to dual simplex, several to barrier and one to
// primal simplex, and returns the basis from whichever finishes first
// (Gurobi Optimizer Reference Manual, "Concurrent Optimization"). This module
// is the same idea over this project's own engines.
//
// The tempting alternative is to PREDICT the best engine from features of the
// instance -- the "algorithm selection" literature, from Rice's 1976 framing
// to Hutter, Xu, Hoos and Leyton-Brown's "Algorithm runtime prediction"
// (Artificial Intelligence 206:79-111, 2014), and Xu et al.'s Hydra-MIP for
// the MIP case. Those methods work, and they need hundreds of training
// instances per engine to fit a runtime model. This project's LP corpus is 19
// Netlib instances. A predictor fitted on that would be noise, and a
// mispredicting selector is worse than no selector -- whereas racing cannot
// be wrong, only wasteful. So: race now, and record per-engine timings while
// racing, which is exactly the training data a selector would later need.
//
// ---------------------------------------------------------------------------
// What this does and does not promise
// ---------------------------------------------------------------------------
//
// It does NOT make the solver faster than its best engine on a given model --
// on a single instance the winner is the engine that would have won anyway,
// minus a little memory-bandwidth contention. What it removes is the risk of
// CHOOSING WRONG, which on a mixed corpus is the larger effect: cuPDLPx wins on
// very large sparse LPs, the dual simplex on medium ones, and nothing about
// the model tells you which in advance.
//
// DETERMINISM. The objective value is deterministic; WHICH optimal basis
// comes back is not, because it depends on which thread finished first. A
// degenerate LP has many optimal vertices and two engines may legitimately
// land on different ones with the same objective. Gurobi documents the same
// tradeoff and offers a deterministic concurrent mode that is "significantly
// slower". This module therefore stays OPT-IN (`--method=concurrent`), so
// every existing test and the MILP search -- which compares solution vectors,
// not just objectives -- keep the single-engine behaviour they were
// validated against.

#ifndef SOVSOLVE_SOLVER_CONCURRENT_SOLVE_HPP
#define SOVSOLVE_SOLVER_CONCURRENT_SOLVE_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Solution.hpp"

namespace sovsolve::solver {

/// What one racing engine did. Kept for every entrant, winner or not: this is
/// the per-engine runtime record that an algorithm-selection model would be
/// trained on, and the reason the racing path is also a data collector.
struct ConcurrentEntry {
  model::Method method = model::Method::DualSimplex;
  std::string name;
  core::SolverStatus status = core::SolverStatus::NotConverged;
  core::Real objective = 0.0;
  std::size_t iterations = 0;
  /// Wall time from the start of the race to this engine stopping, whether it
  /// stopped by converging or by being cancelled.
  double seconds = 0.0;
  bool won = false;
  /// Set when the engine returned an error rather than a verdict.
  bool failed = false;
};

struct ConcurrentReport {
  std::vector<ConcurrentEntry> entries;
  /// Index into `entries`, or `entries.size()` when nothing decided.
  std::size_t winner = 0;
  /// Threads actually started.
  std::size_t threads = 0;
};

/// The engines this build can race, in PRIORITY order.
[[nodiscard]] std::vector<model::Method> default_concurrent_methods();

/// The same list, adjusted for the machine and the options: truncated to the
/// thread budget, and extended with the GPU interior-point engine when
/// `options.concurrent.include_gpu_interior_point` asks for it.
[[nodiscard]] std::vector<model::Method> default_concurrent_methods(
    const model::Options& options);

/// Threads the race may use: `options.concurrent.max_threads`, or the
/// hardware's own count when that is 0.
[[nodiscard]] std::size_t concurrent_thread_budget(const model::Options& options);

/// Solve `problem` in CANONICAL space by racing `methods`, returning the
/// canonical-space solution of the first engine to reach a definitive
/// verdict. `report`, when non-null, receives one entry per engine.
///
/// The problem is shared by const reference across every thread: the engines
/// only read it (verified -- no `const_cast` anywhere in them, and no mutable
/// statics), so no copy is made and a large model costs no extra memory.
[[nodiscard]] core::Expected<model::Solution> solve_concurrent(
    const model::CanonicalProblem& problem, const model::Options& options,
    const std::vector<model::Method>& methods, ConcurrentReport* report = nullptr);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_CONCURRENT_SOLVE_HPP
