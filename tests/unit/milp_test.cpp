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
#include "sovsolve/solver/LpSolve.hpp"
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

/// For tests of one search component (plunging, a heuristic): the root cuts
/// of stage 8b often solve these small programs outright, and then the
/// component under test never gets to run. Answers are unaffected either way.
model::Options without_cuts(model::Options o) {
  o.milp.gomory_cuts = false;
  o.milp.cmir_cuts = false;
  return o;
}

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

/// Every feasible integer point of `ip` with its objective, for checking a
/// claim about the feasible set directly rather than through the optimum.
std::vector<std::pair<std::vector<int>, Real>> feasible_points(const RandomIp& ip) {
  std::vector<std::pair<std::vector<int>, Real>> out;
  const std::size_t n = ip.c.size();
  std::vector<int> x(ip.lo.begin(), ip.lo.end());
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
      out.emplace_back(x, obj);
    }
    std::size_t k = 0;
    while (k < n && x[k] == ip.hi[k]) {
      x[k] = ip.lo[k];
      ++k;
    }
    if (k == n) break;
    ++x[k];
  }
  return out;
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
          problem, without_cuts(options_for(BranchingRule::Reliability, selection)), &stats);
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
    o = without_cuts(o);
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
    o = without_cuts(o);
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
  std::size_t local_redcost = 0;
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
        local_redcost += stats.local_redcost_tightenings;
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
  CHECK(local_redcost > 0);  // [CIP] 8.8 at the nodes
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

/// Stage 8b's root cuts under the oracle. A cut that removes an integer
/// point -- a wrong MIR coefficient, a sign slip substituting a bound or a
/// slack back, a scaling that is not exact -- can delete the optimum, and
/// shows up here as a wrong answer or a false Infeasible. Cuts on and off
/// must both match enumeration; with them on, both separators must produce
/// cuts somewhere, cuts must enter the LP, and the root bound must never get
/// WORSE -- adding valid rows to a minimization can only raise its LP value.
void test_root_cuts_never_change_the_answer() {
  std::size_t gomory = 0;
  std::size_t cmir = 0;
  std::size_t added = 0;
  std::size_t improved = 0;
  for (unsigned seed = 1; seed <= 40; ++seed) {
    const RandomIp ip = make_random_ip(seed * 86028121u, 9, 4);
    Real expected = 0.0;
    const bool has_solution = enumerate(ip, expected);
    model::Problem problem;
    if (!parse(to_lp(ip), problem)) continue;
    for (bool cuts : {true, false}) {
      model::Options o = options_for(BranchingRule::Reliability);
      o.milp.gomory_cuts = cuts;
      o.milp.cmir_cuts = cuts;
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
      if (cuts) {
        gomory += stats.gomory_cuts;
        cmir += stats.cmir_cuts;
        added += stats.cuts_added;
        if (stats.cut_rounds > 0) {
          CHECK(stats.root_bound_after_cuts >=
                stats.root_bound_before_cuts - 1e-7 * (1.0 + std::fabs(stats.root_bound_before_cuts)));
          if (stats.root_bound_after_cuts > stats.root_bound_before_cuts + 1e-7) ++improved;
        }
      } else {
        CHECK_EQ(stats.cuts_added, std::size_t{0});
      }
    }
  }
  CHECK(gomory > 0);
  CHECK(cmir > 0);
  CHECK(added > 0);
  CHECK(improved > 0);
}

/// A MIXED-integer oracle, for what the pure-integer one cannot reach:
/// continuous structural columns, which c-MIR aggregates away and whose
/// bounds it substitutes -- a lower, an upper, or both finite. Every integer
/// assignment in the box is fixed in turn and the continuous remainder solved
/// as an LP (solve_lp, whose own suites stand behind it); the best of those is
/// the optimum. Cuts on and off must both reproduce it.
void test_root_cuts_on_mixed_programs() {
  std::size_t checked = 0;
  std::size_t cmir = 0;
  for (unsigned seed = 1; seed <= 25; ++seed) {
    unsigned st = seed * 2654435761u;
    auto next = [&st]() {
      st = st * 1664525u + 1013904223u;
      return static_cast<int>((st >> 8) % 10007u);
    };
    auto nonzero = [&next]() {
      const int v = next() % 17 - 8;
      return v == 0 ? 3 : v;
    };
    // x0..x2 integer, y0 in [0,5], y1 in [-3,4], y2 >= 0.
    std::string t = next() % 2 == 0 ? "Maximize\n obj:" : "Minimize\n obj:";
    const char* names[6] = {"x0", "x1", "x2", "y0", "y1", "y2"};
    for (int j = 0; j < 6; ++j) {
      const int c = nonzero();
      t += std::string(c < 0 ? " - " : " + ") + std::to_string(c < 0 ? -c : c) + " " + names[j];
    }
    t += "\nSubject To\n";
    for (int i = 0; i < 3; ++i) {
      t += " r" + std::to_string(i) + ":";
      for (int j = 0; j < 6; ++j) {
        const int a = next() % 11 - 5;
        const std::string mag = j < 3 ? std::to_string(a < 0 ? -a : a)
                                      : std::to_string(a < 0 ? -a : a) + ".5";
        t += std::string(a < 0 ? " - " : " + ") + mag + " " + names[j];
      }
      const int kind = next() % 3;
      t += kind == 0 ? " <= " : (kind == 1 ? " >= " : " = ");
      t += std::to_string(next() % 15 - 3) + "\n";
    }
    t += "Bounds\n";
    for (int j = 0; j < 3; ++j) {
      const int lo = next() % 3 - 1;
      t += " " + std::to_string(lo) + " <= " + names[j] + " <= " +
           std::to_string(lo + 2 + next() % 3) + "\n";
    }
    t += " 0 <= y0 <= 5\n -3 <= y1 <= 4\n y2 >= 0\nGeneral\n x0\n x1\n x2\nEnd\n";

    model::Problem problem;
    if (!parse(t, problem)) continue;

    // The oracle: every integer point, the LP over the rest.
    std::vector<std::size_t> ints;
    for (std::size_t j = 0; j < problem.num_cols(); ++j) {
      if (problem.col_type[j] != core::VarType::Continuous) ints.push_back(j);
    }
    if (ints.size() != 3) continue;
    const bool maximize = problem.sense == core::ObjSense::Maximize;
    bool found = false;
    bool unbounded = false;
    Real best = 0.0;
    std::vector<int> v(3);
    for (std::size_t k = 0; k < 3; ++k) v[k] = static_cast<int>(problem.col_lower[ints[k]]);
    for (;;) {
      model::Problem fixed = problem.clone();
      for (std::size_t k = 0; k < 3; ++k) {
        fixed.col_lower[ints[k]] = v[k];
        fixed.col_upper[ints[k]] = v[k];
      }
      model::Options lo;
      lo.log.level = model::LogOptions::Level::Silent;
      lo.simplex.method = model::Method::DualSimplex;
      auto r = solver::solve_lp(fixed, lo);
      if (r.has_value() && r->status == SolverStatus::Unbounded) unbounded = true;
      if (r.has_value() && r->status == SolverStatus::Optimal) {
        if (!found || (maximize ? r->objective > best : r->objective < best)) best = r->objective;
        found = true;
      }
      std::size_t k = 0;
      while (k < 3 && v[k] == static_cast<int>(problem.col_upper[ints[k]])) {
        v[k] = static_cast<int>(problem.col_lower[ints[k]]);
        ++k;
      }
      if (k == 3) break;
      ++v[k];
    }
    if (unbounded) continue;

    for (bool cuts : {true, false}) {
      model::Options o = options_for(BranchingRule::Reliability);
      o.milp.gomory_cuts = cuts;
      o.milp.cmir_cuts = cuts;
      solver::MilpStatistics stats;
      auto result = solver::solve_milp(problem, o, &stats);
      CHECK(result.has_value());
      if (!result.has_value()) continue;
      ++checked;
      if (found) {
        CHECK(result->status == SolverStatus::Optimal);
        CHECK_NEAR(result->objective, best, 1e-6 * (1.0 + std::fabs(best)));
      } else {
        CHECK(result->status == SolverStatus::Infeasible);
      }
      if (cuts) cmir += stats.cmir_cuts;
    }
  }
  CHECK(checked >= 30);
  CHECK(cmir > 0);
}

/// [CIP] chapter 11 under the oracle. A conflict constraint is a claim that
/// some combination of bounds contains no solution better than the incumbent;
/// an unsound one -- a reason read from the wrong side of a row, a literal
/// negated the wrong way, an infeasibility proof relaxed one bound too far --
/// deletes feasible points, and shows up here as a wrong optimum or a false
/// Infeasible. On and off must both match enumeration, and with it on,
/// conflicts must be analyzed, stored and actually used, both from
/// propagation and from infeasible LPs. Cuts are off so the trees are deep
/// enough to fail in.
void test_conflict_analysis_never_changes_the_answer() {
  std::size_t analyzed = 0;
  std::size_t constraints = 0;
  std::size_t deductions = 0;
  std::size_t logged = 0;
  std::size_t logged_sets = 0;
  for (unsigned seed = 1; seed <= 40; ++seed) {
    // Binary programs, 16 columns and 6 rows: 65536 points to enumerate, and
    // trees large enough that a learned conflict gets used again elsewhere.
    RandomIp ip = make_random_ip(seed * 39916801u, 16, 6);
    for (std::size_t j = 0; j < ip.lo.size(); ++j) {
      ip.lo[j] = 0;
      ip.hi[j] = 1;
    }
    Real expected = 0.0;
    const bool has_solution = enumerate(ip, expected);
    model::Problem problem;
    if (!parse(to_lp(ip), problem)) continue;
    // Problem column -> RandomIp variable, by name ("x<j>").
    std::vector<std::size_t> var_of(problem.num_cols(), 0);
    for (std::size_t c = 0; c < problem.num_cols(); ++c) {
      var_of[c] = static_cast<std::size_t>(std::stoul(std::string(problem.col_names[c].substr(1))));
    }
    const auto points = feasible_points(ip);
    for (bool conflicts : {true, false}) {
      model::Options o = without_cuts(options_for(BranchingRule::Reliability));
      o.milp.conflict_analysis = conflicts;
      solver::MilpStatistics stats;
      std::vector<solver::ConflictRecord> log;
      stats.conflict_log = &log;
      auto result = solver::solve_milp(problem, o, &stats);
      // THE SOUNDNESS CHECK, per conflict: a bound disjunction may exclude no
      // feasible point -- or, if its proof used the objective cutoff, no
      // feasible point strictly better than the incumbent it was derived
      // under. An invalid conflict fails here even when it happens never to
      // prune the optimum.
      for (const auto& rec : log) {
        for (const auto& [x, obj] : points) {
          if (rec.uses_cutoff) {
            const bool better = ip.maximize ? obj > rec.incumbent + 1e-6
                                            : obj < rec.incumbent - 1e-6;
            if (!better) continue;
          }
          bool satisfied = false;
          for (const auto& lit : rec.literals) {
            const Real v = x[var_of[lit.col]];
            if (lit.upper ? v <= lit.bound + 1e-6 : v >= lit.bound - 1e-6) {
              satisfied = true;
              break;
            }
          }
          CHECK(satisfied);
          if (!satisfied) {
            std::fprintf(stderr, "BADCONFLICT seed=%u cutoff=%d inc=%g obj=%g lits:", seed,
                         rec.uses_cutoff ? 1 : 0, rec.incumbent, obj);
            for (const auto& lit : rec.literals) {
              std::fprintf(stderr, " x%zu%s%g(val %d)", var_of[lit.col], lit.upper ? "<=" : ">=",
                           lit.bound, x[var_of[lit.col]]);
            }
            std::fprintf(stderr, "\n");
            break;
          }
        }
      }
      ++logged_sets;
      logged += log.size();
      CHECK(result.has_value());
      if (!result.has_value()) continue;
      if (has_solution) {
        CHECK(result->status == SolverStatus::Optimal);
        CHECK_NEAR(result->objective, expected, 1e-6);
        CHECK_NEAR(result->best_bound, result->objective, 1e-6);
      } else {
        CHECK(result->status == SolverStatus::Infeasible);
      }
      if (conflicts) {
        analyzed += stats.conflicts_analyzed;
        constraints += stats.conflict_constraints;
        deductions += stats.conflict_deductions + stats.conflict_cutoffs;
      } else {
        CHECK_EQ(stats.conflict_constraints, std::size_t{0});
      }
    }
  }
  CHECK(analyzed > 0);
  CHECK(constraints > 0);
  CHECK(deductions > 0);
  CHECK(logged > 0);  // the per-conflict check above actually checked something
  CHECK(logged_sets > 0);
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
  test_root_cuts_never_change_the_answer();
  test_root_cuts_on_mixed_programs();
  test_conflict_analysis_never_changes_the_answer();
  test_tiny_knapsack();
  test_integer_infeasible();
  test_maximization_sign();
  test_mixed_integer();
  test_strong_branching_is_confined_to_reliability();
  test_continuous_model_delegates();
  return ::sovsolve::test::report("milp_test");
}
