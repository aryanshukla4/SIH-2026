// Module 30: the concurrent optimizer.
//
// THE ORACLE IS THE SINGLE-ENGINE ANSWER. Racing engines must not change what
// the solver says -- only how quickly it says it. So every claim here is a
// comparison against running the same model on one engine at a time: same
// verdict, same objective, and for a verdict (infeasible, unbounded) the same
// verdict from the race as from the engine that can prove it.
//
// What is deliberately NOT asserted is which engine wins, or that the primal
// point matches. A degenerate LP has many optimal vertices, engines land on
// different ones, and which finishes first depends on the scheduler. The
// objective is the invariant; the vertex is not. ConcurrentSolve.hpp says the
// same thing, and it is why the concurrent method is opt-in.

#include <cmath>
#include <string>
#include <vector>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Problem.hpp"
#include "sovsolve/solver/ConcurrentSolve.hpp"
#include "sovsolve/solver/LpSolve.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
using core::Real;
using core::SolverStatus;
using model::Method;

namespace {

bool parse(const std::string& text, model::Problem& out) {
  auto parsed = io::parseProblem(text, io::FileFormat::Lp);
  if (!parsed.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "parse", parsed.error().format());
    return false;
  }
  out = std::move(parsed.value());
  return true;
}

model::Options options_for(Method m) {
  model::Options o;
  o.log.level = model::LogOptions::Level::Silent;
  o.simplex.method = m;
  return o;
}

/// A handful of small LPs with different shapes: a plain bounded optimum, a
/// degenerate one, an equality-constrained one, and one whose optimum sits at
/// a bound rather than at a constraint intersection.
std::vector<std::string> models() {
  return {
      "Maximize\n obj: 3 x + 2 y\n"
      "Subject To\n c1: x + y <= 4\n c2: x + 3 y <= 6\n"
      "Bounds\n 0 <= x <= 3\n 0 <= y <= 3\nEnd\n",

      "Minimize\n obj: 2 a + 3 b + 4 c\n"
      "Subject To\n r1: a + b + c >= 10\n r2: a - b = 0\n r3: b + 2 c >= 6\n"
      "Bounds\n 0 <= a <= 20\n 0 <= b <= 20\n 0 <= c <= 20\nEnd\n",

      // Degenerate: three constraints meeting at one vertex.
      "Maximize\n obj: x + y\n"
      "Subject To\n d1: x + y <= 2\n d2: x <= 1\n d3: y <= 1\n"
      "Bounds\n 0 <= x <= 5\n 0 <= y <= 5\nEnd\n",

      "Minimize\n obj: - p - q\n"
      "Subject To\n e1: p + 2 q <= 8\n e2: 3 p + q <= 9\n"
      "Bounds\n 0 <= p <= 10\n 0 <= q <= 10\nEnd\n",
  };
}

/// Claim 1: the race returns the same verdict and the same objective as each
/// engine that can solve the model on its own.
void test_race_matches_every_single_engine() {
  std::size_t compared = 0;
  for (const std::string& text : models()) {
    model::Problem problem;
    if (!parse(text, problem)) continue;

    auto raced = solver::solve_lp(problem, options_for(Method::Concurrent));
    CHECK(raced.has_value());
    if (!raced.has_value()) continue;
    CHECK(raced->status == SolverStatus::Optimal);

    for (const Method m : {Method::DualSimplex, Method::PrimalSimplex, Method::Hsd}) {
      auto single = solver::solve_lp(problem, options_for(m));
      CHECK(single.has_value());
      if (!single.has_value()) continue;
      if (single->status != SolverStatus::Optimal) continue;  // that engine's own limit
      CHECK(raced->status == single->status);
      CHECK_NEAR(raced->objective, single->objective,
                 1e-6 * (1.0 + std::fabs(single->objective)));
      ++compared;
    }
  }
  CHECK(compared > 0);
}

/// Claim 2: repeated races agree with each other. The WINNER may differ run
/// to run -- that is the scheduler, and it is allowed -- but the answer may
/// not. This is the property that would break first if a cancelled engine
/// were ever allowed to report `Optimal`.
void test_repeated_races_agree() {
  for (const std::string& text : models()) {
    model::Problem problem;
    if (!parse(text, problem)) continue;

    Real first = 0.0;
    bool have = false;
    for (int trial = 0; trial < 8; ++trial) {
      auto r = solver::solve_lp(problem, options_for(Method::Concurrent));
      CHECK(r.has_value());
      if (!r.has_value()) continue;
      CHECK(r->status == SolverStatus::Optimal);
      if (!have) {
        first = r->objective;
        have = true;
        continue;
      }
      CHECK_NEAR(r->objective, first, 1e-9 * (1.0 + std::fabs(first)));
    }
    CHECK(have);
  }
}

/// Claim 3: the thread budget picks the line-up, and one thread is still a
/// correct solve. This is the path a single-core deployment takes.
void test_thread_budget_controls_the_line_up() {
  model::Options o;
  o.log.level = model::LogOptions::Level::Silent;

  o.concurrent.max_threads = 1;
  CHECK_EQ(solver::default_concurrent_methods(o).size(), std::size_t{1});
  CHECK_EQ(solver::concurrent_thread_budget(o), std::size_t{1});

  o.concurrent.max_threads = 2;
  CHECK_EQ(solver::default_concurrent_methods(o).size(), std::size_t{2});

  // 0 means "ask the hardware", which must never produce an empty line-up.
  o.concurrent.max_threads = 0;
  CHECK(!solver::default_concurrent_methods(o).empty());
  CHECK(solver::concurrent_thread_budget(o) >= 1);

  // A budget far above the number of engines is capped by the engines, not
  // padded with duplicates.
  o.concurrent.max_threads = 64;
  CHECK_EQ(solver::default_concurrent_methods(o).size(),
           solver::default_concurrent_methods().size());

  model::Problem problem;
  if (!parse(models().front(), problem)) return;
  model::Options single = options_for(Method::Concurrent);
  single.concurrent.max_threads = 1;
  auto one = solver::solve_lp(problem, single);
  auto many = solver::solve_lp(problem, options_for(Method::Concurrent));
  CHECK(one.has_value() && many.has_value());
  if (one.has_value() && many.has_value()) {
    CHECK(one->status == many->status);
    CHECK_NEAR(one->objective, many->objective, 1e-9 * (1.0 + std::fabs(many->objective)));
  }
}

/// Claim 4: a verdict about the MODEL still comes back. An infeasible model
/// has no optimum for anyone to win with, so this is the case where the
/// "first definitive status wins" rule has to fire on something other than
/// `Optimal` -- and where a race that simply waited for an optimum would hang.
void test_infeasible_model_still_gets_a_verdict() {
  const std::string text =
      "Minimize\n obj: x\n"
      "Subject To\n c1: x >= 5\n c2: x <= 2\n"
      "Bounds\n 0 <= x <= 10\nEnd\n";
  model::Problem problem;
  if (!parse(text, problem)) return;
  auto r = solver::solve_lp(problem, options_for(Method::Concurrent));
  CHECK(r.has_value());
  if (r.has_value()) CHECK(r->status == SolverStatus::Infeasible);
}

/// Claim 5: the report accounts for every entrant, not just the winner --
/// which is what makes the racing path double as the per-engine timing record
/// an algorithm-selection model would be trained on.
void test_report_covers_every_entrant() {
  model::Problem problem;
  if (!parse(models().front(), problem)) return;
  auto canon = model::canonicalize(problem);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;

  model::Options o = options_for(Method::Concurrent);
  solver::ConcurrentReport report;
  auto r = solver::solve_concurrent(canon->problem, o, {}, &report);
  CHECK(r.has_value());
  CHECK_EQ(report.entries.size(), report.threads);
  CHECK(!report.entries.empty());
  CHECK(report.winner < report.entries.size());

  std::size_t winners = 0;
  for (const solver::ConcurrentEntry& e : report.entries) {
    CHECK(!e.name.empty());
    CHECK(e.seconds >= 0.0);
    if (e.won) ++winners;
  }
  CHECK_EQ(winners, std::size_t{1});
  CHECK(report.entries[report.winner].won);
}

/// Claim 6: with crossover on, a win by an engine that stops at a tolerance
/// is finished at a vertex -- the objective then matches the simplex's to
/// rounding, not merely to the engine's 1e-8. A line-up of ONE tolerance
/// engine makes that engine win every time, so this does not depend on
/// thread timing. With crossover off nothing runs after the race.
void test_crossover_finishes_a_tolerance_win_at_a_vertex() {
  for (const std::string& text : models()) {
    model::Problem problem;
    if (!parse(text, problem)) continue;
    auto canon = model::canonicalize(problem);
    CHECK(canon.has_value());
    if (!canon.has_value()) continue;

    model::Options exact_options = options_for(Method::Concurrent);
    auto exact = solver::solve_concurrent(canon->problem, exact_options,
                                          {Method::DualSimplex});
    CHECK(exact.has_value());
    if (!exact.has_value() || exact->status != SolverStatus::Optimal) continue;

    for (const Method tolerance_engine : {Method::PdlpX, Method::Hsd}) {
      model::Options on = options_for(Method::Concurrent);
      on.concurrent.crossover = true;
      solver::ConcurrentReport report;
      auto crossed = solver::solve_concurrent(canon->problem, on, {tolerance_engine}, &report);
      CHECK(crossed.has_value());
      if (!crossed.has_value()) continue;
      CHECK(crossed->status == SolverStatus::Optimal);
      CHECK(report.crossover_ran);
      CHECK(report.crossover_used);
      CHECK_NEAR(crossed->objective, exact->objective,
                 1e-12 * (1.0 + std::fabs(exact->objective)));

      model::Options off = options_for(Method::Concurrent);
      off.concurrent.crossover = false;
      solver::ConcurrentReport off_report;
      auto plain = solver::solve_concurrent(canon->problem, off, {tolerance_engine}, &off_report);
      CHECK(plain.has_value());
      CHECK(!off_report.crossover_ran);
    }
  }
}

}  // namespace

int main() {
  test_race_matches_every_single_engine();
  test_repeated_races_agree();
  test_thread_budget_controls_the_line_up();
  test_infeasible_model_still_gets_a_verdict();
  test_report_covers_every_entrant();
  test_crossover_finishes_a_tolerance_win_at_a_vertex();
  return ::sovsolve::test::report("concurrent_test");
}
