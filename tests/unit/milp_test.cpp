// Module 28: branch-and-bound on the warm-started dual simplex.
//
// THE ORACLE IS ENUMERATION. Small random pure-integer programs, whose every
// integer point can be listed and checked, are solved under ALL THREE
// branching rules; each must reproduce the enumerated optimum exactly, or
// report Infeasible exactly when no point is feasible. That tests the thing a
// branching rule must never do -- change the answer -- independently of the
// thing it is for, which is changing the node count.
//
// The rest pins behaviour the enumeration cannot reach: maximization and the
// sign of best_bound, a model infeasible only for integers, the
// strong-branching counters, and the Module 22 fixtures ported to this engine.

#include <cmath>
#include <cstdio>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Problem.hpp"
#include "sovsolve/solver/MilpSolve.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
using core::Real;
using core::SolverStatus;
using model::BranchingRule;
using model::NodeSelection;

namespace {

constexpr BranchingRule kRules[] = {BranchingRule::MostFractional,
                                    BranchingRule::Pseudocost,
                                    BranchingRule::Reliability};

constexpr NodeSelection kSelections[] = {NodeSelection::BestFirst,
                                         NodeSelection::Interleaved};

model::Options options_for(BranchingRule rule,
                           NodeSelection selection = NodeSelection::Interleaved) {
  model::Options o;
  o.log.level = model::LogOptions::Level::Silent;
  o.milp.branching = rule;
  o.milp.node_selection = selection;
  return o;
}

bool parse(const std::string& text, model::Problem& out) {
  auto parsed = io::parseProblem(text, io::FileFormat::Lp);
  if (!parsed.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "parse", parsed.error().format());
    return false;
  }
  out = std::move(parsed.value());
  return true;
}

// -------------------------------------------------------------------------
// The enumeration oracle
// -------------------------------------------------------------------------

struct RandomIp {
  bool maximize = false;
  std::vector<int> c;
  std::vector<std::vector<int>> a;  ///< rows
  std::vector<int> rhs;
  std::vector<char> sense;          ///< 'L' <=, 'G' >=, 'E' =
  std::vector<int> lo, hi;
};

RandomIp make_random_ip(unsigned seed, int n, int m) {
  auto next = [&seed]() {
    seed = seed * 1664525u + 1013904223u;
    return static_cast<int>((seed >> 8) % 10007u);
  };
  RandomIp ip;
  ip.maximize = next() % 2 == 0;
  for (int j = 0; j < n; ++j) {
    ip.c.push_back(next() % 19 - 9);
    const int lo = next() % 3 - 1;  // -1, 0 or 1
    ip.lo.push_back(lo);
    ip.hi.push_back(lo + 2 + next() % 3);  // width 2..4
  }
  for (int i = 0; i < m; ++i) {
    std::vector<int> row;
    for (int j = 0; j < n; ++j) row.push_back(next() % 11 - 5);
    ip.a.push_back(row);
    ip.rhs.push_back(next() % 21 - 4);
    const int s = next() % 6;
    ip.sense.push_back(s < 3 ? 'L' : (s < 5 ? 'G' : 'E'));
  }
  return ip;
}

std::string to_lp(const RandomIp& ip) {
  auto term = [](int coef, int j) {
    return std::string(coef < 0 ? " - " : " + ") + std::to_string(coef < 0 ? -coef : coef) +
           " x" + std::to_string(j);
  };
  std::string t = ip.maximize ? "Maximize\n obj:" : "Minimize\n obj:";
  for (std::size_t j = 0; j < ip.c.size(); ++j) t += term(ip.c[j], static_cast<int>(j));
  t += "\nSubject To\n";
  for (std::size_t i = 0; i < ip.a.size(); ++i) {
    t += " r" + std::to_string(i) + ":";
    for (std::size_t j = 0; j < ip.a[i].size(); ++j) t += term(ip.a[i][j], static_cast<int>(j));
    t += ip.sense[i] == 'L' ? " <= " : (ip.sense[i] == 'G' ? " >= " : " = ");
    t += std::to_string(ip.rhs[i]) + "\n";
  }
  t += "Bounds\n";
  for (std::size_t j = 0; j < ip.c.size(); ++j) {
    t += " " + std::to_string(ip.lo[j]) + " <= x" + std::to_string(j) + " <= " +
         std::to_string(ip.hi[j]) + "\n";
  }
  t += "General\n";
  for (std::size_t j = 0; j < ip.c.size(); ++j) t += " x" + std::to_string(j) + "\n";
  t += "End\n";
  return t;
}

/// Every integer point in the box, checked against every row. Returns false
/// when none is feasible.
bool enumerate(const RandomIp& ip, Real& best) {
  const std::size_t n = ip.c.size();
  std::vector<int> x(ip.lo.begin(), ip.lo.end());
  bool found = false;
  best = ip.maximize ? -std::numeric_limits<Real>::infinity()
                     : std::numeric_limits<Real>::infinity();
  for (;;) {
    bool feasible = true;
    for (std::size_t i = 0; i < ip.a.size() && feasible; ++i) {
      int lhs = 0;
      for (std::size_t j = 0; j < n; ++j) lhs += ip.a[i][j] * x[j];
      if (ip.sense[i] == 'L') feasible = lhs <= ip.rhs[i];
      else if (ip.sense[i] == 'G') feasible = lhs >= ip.rhs[i];
      else feasible = lhs == ip.rhs[i];
    }
    if (feasible) {
      Real obj = 0.0;
      for (std::size_t j = 0; j < n; ++j) obj += ip.c[j] * x[j];
      if (!found || (ip.maximize ? obj > best : obj < best)) best = obj;
      found = true;
    }
    // Odometer increment over the box.
    std::size_t k = 0;
    while (k < n && x[k] == ip.hi[k]) {
      x[k] = ip.lo[k];
      ++k;
    }
    if (k == n) break;
    ++x[k];
  }
  return found;
}

void test_every_rule_matches_enumeration() {
  int feasible_cases = 0;
  int infeasible_cases = 0;
  for (unsigned seed = 1; seed <= 60; ++seed) {
    const RandomIp ip = make_random_ip(seed * 7919u, 5, 3);
    Real expected = 0.0;
    const bool has_solution = enumerate(ip, expected);
    (has_solution ? feasible_cases : infeasible_cases)++;

    model::Problem problem;
    if (!parse(to_lp(ip), problem)) continue;
    for (BranchingRule rule : kRules) {
      for (NodeSelection selection : kSelections) {
        auto result = solver::solve_milp(problem, options_for(rule, selection));
        CHECK(result.has_value());
        if (!result.has_value()) continue;
        if (has_solution && result->status != SolverStatus::Optimal) {
          std::fprintf(stderr, "DEBUG seed=%u rule=%d sel=%d status=%d obj=%g expected=%g\n%s\n",
                       seed, static_cast<int>(rule), static_cast<int>(selection),
                       static_cast<int>(result->status), result->objective, expected,
                       to_lp(ip).c_str());
        }
        if (has_solution) {
          CHECK(result->status == SolverStatus::Optimal);
          CHECK_NEAR(result->objective, expected, 1e-6);
          // Optimal means the bound meets the objective.
          CHECK_NEAR(result->best_bound, result->objective, 1e-6);
        } else {
          CHECK(result->status == SolverStatus::Infeasible);
        }
      }
    }
  }
  // Both outcomes must actually be exercised, or half the oracle is idle.
  CHECK(feasible_cases >= 15);
  CHECK(infeasible_cases >= 5);
}

// -------------------------------------------------------------------------
// Behaviour the enumeration does not reach
// -------------------------------------------------------------------------

/// Module 22's tiny knapsack, on this engine: min -5a - 4b s.t. 6a + 5b <= 10.
void test_tiny_knapsack() {
  model::Problem p;
  if (!parse(R"(Minimize
 obj: -5 a - 4 b
Subject To
 cap: 6 a + 5 b <= 10
Binary
 a
 b
End
)",
             p)) {
    return;
  }
  for (BranchingRule rule : kRules) {
    auto r = solver::solve_milp(p, options_for(rule));
    CHECK(r.has_value());
    if (!r.has_value()) continue;
    CHECK(r->status == SolverStatus::Optimal);
    CHECK_NEAR(r->objective, -5.0, 1e-9);
    CHECK_NEAR(r->best_bound, -5.0, 1e-9);
  }
}

/// Feasible as an LP, empty for integers -- the relaxation alone cannot see it.
void test_integer_infeasible() {
  model::Problem p;
  if (!parse(R"(Minimize
 obj: x + y
Subject To
 r: 2 x + 2 y = 3
Bounds
 0 <= x <= 5
 0 <= y <= 5
General
 x
 y
End
)",
             p)) {
    return;
  }
  for (BranchingRule rule : kRules) {
    auto r = solver::solve_milp(p, options_for(rule));
    CHECK(r.has_value());
    if (r.has_value()) CHECK(r->status == SolverStatus::Infeasible);
  }
}

/// A maximization: the canonical search minimizes the negation, so the
/// reported objective AND best_bound must come back in the original sense.
void test_maximization_sign() {
  model::Problem p;
  if (!parse(R"(Maximize
 obj: 3 x + 2 y + 4 z
Subject To
 a: x + y + 2 z <= 4
 b: 2 x + z <= 5
Bounds
 0 <= x <= 3
 0 <= y <= 3
 0 <= z <= 3
General
 x
 y
 z
End
)",
             p)) {
    return;
  }
  // By enumeration: x=2, y=0, z=1 gives 10; x=2, y=2, z=0 gives 10; nothing
  // exceeds it.
  for (BranchingRule rule : kRules) {
    auto r = solver::solve_milp(p, options_for(rule));
    CHECK(r.has_value());
    if (!r.has_value()) continue;
    CHECK(r->status == SolverStatus::Optimal);
    CHECK_NEAR(r->objective, 10.0, 1e-9);
    CHECK_NEAR(r->best_bound, 10.0, 1e-9);
  }
}

/// Continuous columns alongside integer ones: only the integers are branched.
void test_mixed_integer() {
  model::Problem p;
  if (!parse(R"(Minimize
 obj: - x - 2 y + 0.5 w
Subject To
 a: x + y + w >= 1.5
 b: 3 x + 4 y <= 9.5
Bounds
 0 <= x <= 4
 0 <= y <= 4
 0 <= w <= 10
General
 x
 y
End
)",
             p)) {
    return;
  }
  // Best integer pair under 3x + 4y <= 9.5 maximizing x + 2y: (0,2) -> 4,
  // (1,1) -> 3, (3,0) -> 3, (1,1)... so x=0, y=2, and w=0 already satisfies
  // row a. Objective -4.
  for (BranchingRule rule : kRules) {
    auto r = solver::solve_milp(p, options_for(rule));
    CHECK(r.has_value());
    if (!r.has_value()) continue;
    CHECK(r->status == SolverStatus::Optimal);
    CHECK_NEAR(r->objective, -4.0, 1e-7);
  }
}

/// Strong branching runs under Reliability and ONLY there.
void test_strong_branching_is_confined_to_reliability() {
  const RandomIp ip = make_random_ip(424242u, 8, 4);
  model::Problem p;
  if (!parse(to_lp(ip), p)) return;
  for (BranchingRule rule : kRules) {
    solver::MilpStatistics stats;
    auto r = solver::solve_milp(p, options_for(rule), &stats);
    CHECK(r.has_value());
    if (!r.has_value()) continue;
    CHECK_EQ(stats.unreliable_nodes, std::size_t{0});
    if (rule == BranchingRule::Reliability) {
      if (stats.nodes > 1) CHECK(stats.strong_branching_probes > 0);
    } else {
      CHECK_EQ(stats.strong_branching_probes, std::size_t{0});
    }
  }
}

/// A continuous model goes straight to the LP engine.
void test_continuous_model_delegates() {
  model::Problem p;
  if (!parse(R"(Minimize
 obj: - x - y
Subject To
 a: x + 2 y <= 4
 b: 3 x + y <= 6
End
)",
             p)) {
    return;
  }
  auto r = solver::solve_milp(p, options_for(BranchingRule::Reliability));
  CHECK(r.has_value());
  if (!r.has_value()) return;
  CHECK(r->status == SolverStatus::Optimal);
  CHECK_NEAR(r->objective, -2.8, 1e-9);
}

/// Node selection on trees deep enough for it to matter. The 5-variable
/// oracle above rarely goes past depth 3, where [CIP] section 6.3's plunge
/// limit (0.5 dmax steps) is one step at most. Nine variables of width up to
/// 4 give trees where plunges run several steps, incumbents arrive mid-search,
/// and the gap-based abort and out-of-order pruning are exercised -- all of
/// which must leave the ANSWER untouched. Still checked by enumeration.
void test_node_selection_on_deeper_trees() {
  std::size_t plunge_steps = 0;
  std::size_t checked = 0;
  for (unsigned seed = 1; seed <= 12; ++seed) {
    const RandomIp ip = make_random_ip(seed * 104729u, 9, 4);
    Real expected = 0.0;
    const bool has_solution = enumerate(ip, expected);
    model::Problem problem;
    if (!parse(to_lp(ip), problem)) continue;
    for (NodeSelection selection : kSelections) {
      solver::MilpStatistics stats;
      auto result = solver::solve_milp(
          problem, options_for(BranchingRule::Reliability, selection), &stats);
      CHECK(result.has_value());
      if (!result.has_value()) continue;
      ++checked;
      if (has_solution) {
        CHECK(result->status == SolverStatus::Optimal);
        CHECK_NEAR(result->objective, expected, 1e-6);
        CHECK_NEAR(result->best_bound, result->objective, 1e-6);
      } else {
        CHECK(result->status == SolverStatus::Infeasible);
      }
      if (selection == NodeSelection::BestFirst) CHECK_EQ(stats.plunge_steps, std::size_t{0});
      if (selection == NodeSelection::Interleaved) plunge_steps += stats.plunge_steps;
    }
  }
  CHECK_EQ(checked, std::size_t{24});
  // Plunging must actually happen somewhere, or this test checks nothing new.
  CHECK(plunge_steps > 0);
}


/// [CIP] section 9.1.2 on the smallest case it applies to. The LP optimum
/// puts a 3-coefficient column at 4/3 (objective 1.33); every column has a
/// down-lock (the >= row) and no up-lock, so simple rounding rounds UP, to an
/// integer point of objective 2, which is optimal (no single column reaches
/// 4). The coefficients have gcd 1 so the MILP row tightening cannot make the
/// LP integral first. With a one-node limit that point is the only incumbent
/// the run can have -- and without heuristics the run has none.
void test_simple_rounding_at_the_root() {
  model::Problem p;
  if (!parse(R"(Minimize
 obj: x1 + x2 + x3
Subject To
 cover: 2 x1 + 3 x2 + 3 x3 >= 4
Bounds
 0 <= x1 <= 5
 0 <= x2 <= 5
 0 <= x3 <= 5
General
 x1
 x2
 x3
End
)",
             p)) {
    return;
  }
  for (bool heuristics : {true, false}) {
    model::Options o = options_for(BranchingRule::Reliability);
    o.milp.heuristics = heuristics;
    o.milp.node_limit = 1;
    // Cuts could make the root LP integral, leaving nothing to round.
    o.milp.root_cuts = false;
    solver::MilpStatistics stats;
    auto r = solver::solve_milp(p, o, &stats);
    CHECK(r.has_value());
    if (!r.has_value()) continue;
    if (heuristics) {
      CHECK_EQ(stats.rounding_solutions, std::size_t{1});
      CHECK(stats.incumbents >= 1);
      CHECK_NEAR(r->objective, 2.0, 1e-9);
    } else {
      CHECK_EQ(stats.incumbents, std::size_t{0});
      CHECK_EQ(stats.dives, std::size_t{0});
    }
  }
}

/// Heuristics may only ADD incumbents, so the answer must not move: the
/// 9-variable oracle, heuristics on and off, both exact. And they must
/// actually run and find points somewhere -- otherwise "the answer did not
/// change" is vacuous. Every point they offer is checked against the rows
/// before it may prune (MilpSolve.cpp offer_point); an unchecked infeasible
/// point with a good objective would prune the true optimum and show up here
/// as a wrong answer.
void test_heuristics_never_change_the_answer() {
  std::size_t dives = 0;
  std::size_t found = 0;
  for (unsigned seed = 1; seed <= 20; ++seed) {
    const RandomIp ip = make_random_ip(seed * 15485863u, 9, 4);
    Real expected = 0.0;
    const bool has_solution = enumerate(ip, expected);
    model::Problem problem;
    if (!parse(to_lp(ip), problem)) continue;
    for (bool heuristics : {true, false}) {
      model::Options o = options_for(BranchingRule::Reliability);
      o.milp.heuristics = heuristics;
      solver::MilpStatistics stats;
      auto result = solver::solve_milp(problem, o, &stats);
      CHECK(result.has_value());
      if (!result.has_value()) continue;
      if (has_solution) {
        CHECK(result->status == SolverStatus::Optimal);
        CHECK_NEAR(result->objective, expected, 1e-6);
        CHECK_NEAR(result->best_bound, result->objective, 1e-6);
      } else {
        CHECK(result->status == SolverStatus::Infeasible);
      }
      if (heuristics) {
        dives += stats.dives;
        found += stats.dive_solutions + stats.rounding_solutions;
      } else {
        CHECK_EQ(stats.dives, std::size_t{0});
        CHECK_EQ(stats.dive_solutions + stats.rounding_solutions, std::size_t{0});
      }
    }
  }
  CHECK(dives > 0);
  CHECK(found > 0);
}

/// The root heuristics -- feasibility pump and RENS -- on their own: dives
/// are given no budget, so any heuristic point comes from these two or from
/// simple rounding. Answers stay exact under enumeration (both only ADD
/// incumbents, and every point is checked before it may prune), and both
/// must actually run and, somewhere across the seeds, produce a point.
/// The first run of this test caught the pump scoring its candidates against
/// its OWN objective (the costs were swapped in for the whole pump instead of
/// only for each LP solve): wrong optima, fractional objective values.
void test_pump_and_rens_never_change_the_answer() {
  std::size_t pump_rounds = 0;
  std::size_t rens_nodes = 0;
  std::size_t pump_found = 0;
  std::size_t rens_found = 0;
  for (unsigned seed = 1; seed <= 20; ++seed) {
    const RandomIp ip = make_random_ip(seed * 32452843u, 9, 4);
    Real expected = 0.0;
    const bool has_solution = enumerate(ip, expected);
    model::Problem problem;
    if (!parse(to_lp(ip), problem)) continue;
    model::Options o = options_for(BranchingRule::Reliability);
    o.milp.dive_quota = 0.0;
    o.milp.dive_allowance = 0;
    solver::MilpStatistics stats;
    auto result = solver::solve_milp(problem, o, &stats);
    CHECK(result.has_value());
    if (!result.has_value()) continue;
    if (has_solution) {
      CHECK(result->status == SolverStatus::Optimal);
      CHECK_NEAR(result->objective, expected, 1e-6);
      CHECK_NEAR(result->best_bound, result->objective, 1e-6);
    } else {
      CHECK(result->status == SolverStatus::Infeasible);
    }
    CHECK_EQ(stats.dives, std::size_t{0});
    pump_rounds += stats.pump_rounds;
    rens_nodes += stats.rens_nodes;
    pump_found += stats.pump_solutions;
    rens_found += stats.rens_solutions;
  }
  CHECK(pump_rounds > 0);
  CHECK(rens_nodes > 0);
  // Each separately: a RENS whose points are all rejected (say, mapped back
  // to canonical space with the wrong scale) must not hide behind the pump.
  // Measured with the code as written: pump 11 points, RENS 3.
  CHECK(pump_found > 0);
  CHECK(rens_found > 0);
}

/// [CIP] chapter 7 under the oracle. Propagation removes points from node
/// domains, so a wrong deduction -- a sign slip in a residual, a bound
/// rounded the wrong way, a cutoff one unit too tight -- deletes feasible
/// (possibly optimal) points and shows up as a wrong optimum or a false
/// Infeasible. Both settings must match enumeration, and with it on, every
/// mechanism must actually fire somewhere: local tightenings, nodes proven
/// empty before their LP, and global reduced-cost tightenings.
void test_propagation_never_changes_the_answer() {
  std::size_t tightenings = 0;
  std::size_t cutoffs = 0;
  std::size_t redcost = 0;
  for (unsigned seed = 1; seed <= 30; ++seed) {
    const RandomIp ip = make_random_ip(seed * 49979687u, 9, 4);
    Real expected = 0.0;
    const bool has_solution = enumerate(ip, expected);
    model::Problem problem;
    if (!parse(to_lp(ip), problem)) continue;
    for (bool propagation : {true, false}) {
      model::Options o = options_for(BranchingRule::Reliability);
      o.milp.propagation = propagation;
      solver::MilpStatistics stats;
      auto result = solver::solve_milp(problem, o, &stats);
      CHECK(result.has_value());
      if (!result.has_value()) continue;
      if (has_solution) {
        CHECK(result->status == SolverStatus::Optimal);
        CHECK_NEAR(result->objective, expected, 1e-6);
        CHECK_NEAR(result->best_bound, result->objective, 1e-6);
      } else {
        CHECK(result->status == SolverStatus::Infeasible);
      }
      if (propagation) {
        tightenings += stats.propagation_tightenings;
        cutoffs += stats.propagation_cutoffs;
        redcost += stats.redcost_tightenings;
      } else {
        CHECK_EQ(stats.propagation_tightenings + stats.propagation_cutoffs +
                     stats.redcost_tightenings,
                 std::size_t{0});
      }
    }
  }
  CHECK(tightenings > 0);
  CHECK(cutoffs > 0);
  CHECK(redcost > 0);
}

/// An INFINITE activity contribution ([CIP] 7.1: "accumulated in separate
/// counters"). In `x - y <= 2` with y >= 0 unbounded above, the row's minimum
/// activity is -inf, so it implies NOTHING about x. A propagator that dropped
/// the infinite term would read the finite part alone and conclude x <= 2,
/// cutting off the optimum x = 10, y = 8 (objective -10 + 0.8 = -9.2). The
/// enumeration oracle cannot see this: all its columns are bounded.
void test_propagation_respects_infinite_bounds() {
  model::Problem p;
  if (!parse(R"(Minimize
 obj: - x + 0.1 y
Subject To
 r: x - y <= 2
Bounds
 0 <= x <= 10
 y >= 0
General
 x
End
)",
             p)) {
    return;
  }
  for (bool propagation : {true, false}) {
    model::Options o = options_for(BranchingRule::Reliability);
    o.milp.propagation = propagation;
    auto r = solver::solve_milp(p, o);
    CHECK(r.has_value());
    if (!r.has_value()) continue;
    CHECK(r->status == SolverStatus::Optimal);
    CHECK_NEAR(r->objective, -9.2, 1e-7);
  }

  // Infinite bounds are stored as 1e20 (Types.hpp), not IEEE infinity. A
  // propagator that summed y's -1e20 as a NUMBER would get a minimum activity
  // of exactly -1e20 -- the finite part, w's -5, lost to rounding -- and then
  // y's residual as 0 instead of -5, deducing y <= 2 where the truth is y <= 7.
  // The optimum is x = 0, w = -5, y = 7: objective -7, not -2. This was a real
  // bug, found when the mutant above survived and the rows were printed.
  model::Problem q;
  if (!parse(R"(Minimize
 obj: - y
Subject To
 r: x + w + y <= 2
Bounds
 0 <= x <= 10
 -5 <= w <= 0
 -inf <= y <= 10
General
 x
End
)",
             q)) {
    return;
  }
  for (bool propagation : {true, false}) {
    model::Options o = options_for(BranchingRule::Reliability);
    o.milp.propagation = propagation;
    auto r = solver::solve_milp(q, o);
    CHECK(r.has_value());
    if (!r.has_value()) continue;
    CHECK(r->status == SolverStatus::Optimal);
    CHECK_NEAR(r->objective, -7.0, 1e-7);
  }
}
}  // namespace

int main() {
  test_every_rule_matches_enumeration();
  test_node_selection_on_deeper_trees();
  test_simple_rounding_at_the_root();
  test_heuristics_never_change_the_answer();
  test_pump_and_rens_never_change_the_answer();
  test_propagation_never_changes_the_answer();
  test_propagation_respects_infinite_bounds();
  test_tiny_knapsack();
  test_integer_infeasible();
  test_maximization_sign();
  test_mixed_integer();
  test_strong_branching_is_confined_to_reliability();
  test_continuous_model_delegates();
  return ::sovsolve::test::report("milp_test");
}
