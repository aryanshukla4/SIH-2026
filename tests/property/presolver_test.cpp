// Presolver (Module 4) property tests.
//
// Each rule is checked independently, against hand-built tiny problems whose
// expected outcome is worked out from the spec (docs/spec/module.txt section 4,
// Presolver.hpp's doc comment) rather than read out of the implementation --
// the same discipline tests/property/canonical_test.cpp already uses.

#include <cmath>
#include <cstddef>

#include "sovsolve/core/SparseBuilder.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Problem.hpp"
#include "sovsolve/solver/Presolver.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
using core::INF;
using core::Real;

namespace {

/// Seeds a TransformStack with identity KeepColumn/MapRow records -- the
/// shape canonicalize() always produces when nothing was substituted out --
/// so `presolve()`'s original-index translation has real records to read,
/// the same way it would after a genuine canonicalize() call.
model::TransformStack identity_stack(std::size_t rows, std::size_t cols) {
  model::TransformStack stack;
  stack.original_rows = rows;
  stack.original_cols = cols;
  for (std::size_t i = 0; i < rows; ++i) {
    stack.push({model::TransformKind::MapRow, static_cast<core::Index>(i),
                static_cast<core::Index>(i), 0.0, 0.0});
  }
  for (std::size_t j = 0; j < cols; ++j) {
    stack.push({model::TransformKind::KeepColumn, static_cast<core::Index>(j),
                static_cast<core::Index>(j), 0.0, 0.0});
  }
  return stack;
}

void test_empty_row_consistent_is_removed() {
  // Row 0: x0 + 2*x1 = 5 (coefficients deliberately DIFFERENT -- equal
  // coefficients plus equal cost would make x0/x1 duplicate columns, which
  // Phase 2 now merges, cascading well past what this test means to cover).
  // Row 1: empty, b=0 -- vacuously satisfied, must be dropped without
  // touching row 0 or either column.
  core::SparseBuilder builder(2, 2);
  builder.count(0, 0);
  builder.count(0, 1);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 1.0);
  builder.insert(0, 1, 2.0);

  model::CanonicalProblem problem;
  problem.A = builder.finish();
  problem.c = core::RealVector(2, 1.0);
  problem.b = core::RealVector(2, 0.0);
  problem.b[0] = 5.0;
  problem.col_lower = core::RealVector(2, 0.0);
  problem.col_upper = core::RealVector(2, 10.0);
  problem.num_equality = 2;

  auto transforms = identity_stack(2, 2);
  model::Options options;
  CHECK(solver::presolve(problem, options, transforms).ok());

  CHECK_EQ(problem.num_rows(), std::size_t{1});
  CHECK_EQ(problem.num_cols(), std::size_t{2});
  CHECK_NEAR(problem.b[0], 5.0, 1e-12);
}

void test_empty_row_inconsistent_is_infeasible() {
  core::SparseBuilder builder(2, 2);
  builder.count(0, 0);
  builder.count(0, 1);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 1.0);
  builder.insert(0, 1, 1.0);

  model::CanonicalProblem problem;
  problem.A = builder.finish();
  problem.c = core::RealVector(2, 1.0);
  problem.b = core::RealVector(2, 0.0);
  problem.b[0] = 5.0;
  problem.b[1] = 3.0;  // row 1 is empty but claims activity == 3: infeasible
  problem.col_lower = core::RealVector(2, 0.0);
  problem.col_upper = core::RealVector(2, 10.0);
  problem.num_equality = 2;

  auto transforms = identity_stack(2, 2);
  model::Options options;
  const auto status = solver::presolve(problem, options, transforms);
  CHECK(!status.ok());
  CHECK(status.error().code == core::ErrorCode::PrimalInfeasible);
}

void test_empty_inequality_row_with_positive_slack_is_feasible() {
  // Row 0: x0 + x1 = 5 (equality). Row 1: empty, INEQUALITY, canonical
  // b=3 -- means `0 + s_1 = 3, s_1 >= 0`, satisfied by s_1=3. Must be
  // removed WITHOUT reporting infeasible -- a positive leftover RHS on an
  // inequality row is unused slack, not a violation (unlike the equality
  // case, where any nonzero b is infeasible). This is the case the
  // pre-fix |b|>tol check got wrong. Coefficients deliberately differ (see
  // test_empty_row_consistent_is_removed) so x0/x1 aren't duplicate columns.
  core::SparseBuilder builder(2, 2);
  builder.count(0, 0);
  builder.count(0, 1);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 1.0);
  builder.insert(0, 1, 2.0);

  model::CanonicalProblem problem;
  problem.A = builder.finish();
  problem.c = core::RealVector(2, 1.0);
  problem.b = core::RealVector(2, 0.0);
  problem.b[0] = 5.0;
  problem.b[1] = 3.0;  // row 1: inequality, empty, positive slack -- feasible
  problem.col_lower = core::RealVector(2, 0.0);
  problem.col_upper = core::RealVector(2, 10.0);
  problem.num_equality = 1;  // row 0 equality, row 1 inequality

  auto transforms = identity_stack(2, 2);
  model::Options options;
  CHECK(solver::presolve(problem, options, transforms).ok());

  CHECK_EQ(problem.num_rows(), std::size_t{1});
  CHECK_EQ(problem.num_cols(), std::size_t{2});
}

void test_empty_inequality_row_with_negative_rhs_is_infeasible() {
  // Same shape, but row 1's canonical b is NEGATIVE: `0 + s_1 = -3, s_1>=0`
  // has no solution -- genuinely infeasible.
  core::SparseBuilder builder(2, 2);
  builder.count(0, 0);
  builder.count(0, 1);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 1.0);
  builder.insert(0, 1, 1.0);

  model::CanonicalProblem problem;
  problem.A = builder.finish();
  problem.c = core::RealVector(2, 1.0);
  problem.b = core::RealVector(2, 0.0);
  problem.b[0] = 5.0;
  problem.b[1] = -3.0;
  problem.col_lower = core::RealVector(2, 0.0);
  problem.col_upper = core::RealVector(2, 10.0);
  problem.num_equality = 1;

  auto transforms = identity_stack(2, 2);
  model::Options options;
  const auto status = solver::presolve(problem, options, transforms);
  CHECK(!status.ok());
  CHECK(status.error().code == core::ErrorCode::PrimalInfeasible);
}

void test_singleton_row_tightens_without_fixing() {
  // Row 0 (equality): x0 + x1 = 5 -- not a singleton, stays untouched.
  // Row 1 (inequality, singleton): x1 <= 8 -- tighter than x1's current
  // upper bound of 10, but not tight enough to collapse to a fix. Row 1
  // must survive; only x1's upper bound changes.
  core::SparseBuilder builder(2, 2);
  builder.count(0, 0);
  builder.count(0, 1);
  builder.count(1, 1);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 1.0);
  builder.insert(0, 1, 1.0);
  builder.insert(1, 1, 1.0);

  model::CanonicalProblem problem;
  problem.A = builder.finish();
  problem.c = core::RealVector(2, 1.0);
  problem.b = core::RealVector(2);
  problem.b[0] = 5.0;
  problem.b[1] = 8.0;
  problem.col_lower = core::RealVector(2, 0.0);
  problem.col_upper = core::RealVector(2, 10.0);
  problem.num_equality = 1;  // row 0 equality, row 1 inequality

  auto transforms = identity_stack(2, 2);
  model::Options options;
  CHECK(solver::presolve(problem, options, transforms).ok());

  CHECK_EQ(problem.num_rows(), std::size_t{2});  // row 1 survives, just tightened
  CHECK_EQ(problem.num_cols(), std::size_t{2});
  CHECK_NEAR(problem.col_upper[1], 8.0, 1e-12);
  CHECK_NEAR(problem.col_lower[1], 0.0, 1e-12);
}

void test_singleton_row_collapses_to_fix() {
  // Row 0 (equality): x0 + x1 + 2*x2 = 5 -- deliberately THREE columns, not
  // two, so that after x1 is fixed and removed, row 0 still has two entries
  // left (x0, x2) rather than becoming a singleton itself -- keeping this
  // test's cascade to exactly one fix (a row 0 that collapses further, once
  // x1's removal leaves it with only one column, is real and correct
  // behavior, just a different, separately-covered scenario). x2's
  // coefficient is deliberately 2.0, not 1.0 -- equal to x0's would make them
  // duplicate columns (same pattern, same cost), which Phase 2 now merges,
  // cascading this test well past its own intent.
  //
  // Row 1 (inequality, singleton): x1 <= 3, and x1's own lower bound is
  // ALREADY 3 -- the implied upper bound (3) meets the existing lower bound
  // (3) exactly, collapsing to a fix. x1=3 must fold into row 0
  // (b[0] -= 1*3 = 2) and the objective (obj_offset += c[1]*3 = 3); row 1
  // then reads empty with a positive leftover RHS (3) and is removed by the
  // empty-row block on the next internal pass, all within this one
  // presolve() call.
  core::SparseBuilder builder(2, 3);
  builder.count(0, 0);
  builder.count(0, 1);
  builder.count(0, 2);
  builder.count(1, 1);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 1.0);
  builder.insert(0, 1, 1.0);
  builder.insert(0, 2, 2.0);
  builder.insert(1, 1, 1.0);

  model::CanonicalProblem problem;
  problem.A = builder.finish();
  problem.c = core::RealVector(3, 1.0);
  problem.b = core::RealVector(2);
  problem.b[0] = 5.0;
  problem.b[1] = 3.0;
  problem.col_lower = core::RealVector(3, 0.0);
  problem.col_lower[1] = 3.0;
  problem.col_upper = core::RealVector(3, 10.0);
  problem.num_equality = 1;

  auto transforms = identity_stack(2, 3);
  model::Options options;
  CHECK(solver::presolve(problem, options, transforms).ok());

  CHECK_EQ(problem.num_rows(), std::size_t{1});  // row 1 gone too, via the empty-row cascade
  CHECK_EQ(problem.num_cols(), std::size_t{2});  // x0, x2 survive; x1 fixed and removed
  CHECK_NEAR(problem.b[0], 2.0, 1e-12);       // 5 - 1*3
  CHECK_NEAR(problem.obj_offset, 3.0, 1e-12);  // c[1]*3

  bool found = false;
  for (const auto& rec : transforms.records()) {
    if (rec.kind == model::TransformKind::RemoveFixedVariable && rec.primary == 1) {
      found = true;
      CHECK_NEAR(rec.value, 3.0, 1e-12);
    }
  }
  CHECK(found);
}

void test_singleton_row_fix_folds_its_own_row() {
  // Row 0 (equality, singleton): x1 = 3 -- THIS row (not some other row) is
  // the one whose own implied value triggers the fix, so it is the row this
  // test is actually targeting: presolve() must fold x1's fixed value back
  // into row 0's OWN rhs, not just every other row that mentions column 1.
  //
  // A real bug (found via the `egout` Netlib instance, whose row 4 is exactly
  // this shape) skipped row i in the fold loop, reasoning that "row i caused
  // the fix, so its residual must already be zero" -- false: b[0] stays at
  // its ORIGINAL value (3) unless explicitly folded, so once column 1 is
  // removed, row 0 reads as an empty row with a nonzero equality rhs and is
  // wrongly reported PrimalInfeasible on the very next presolve pass, even
  // though x1=3 satisfies row 0 exactly.
  //
  // Row 1 (inequality) also carries column 1, alongside x0 and x2, so the
  // test also confirms the fold still reaches every OTHER row as before.
  // x2's coefficient there is deliberately 2.0, not 1.0 -- equal to x0's
  // would make them duplicate columns (same pattern, same cost), which
  // Phase 2 now merges, cascading this test well past its own intent.
  //
  // Row 2 (inequality, generous, never binding) gives x0 and x2 a SECOND
  // entry each -- without it they are singleton columns in an inequality
  // row with matching cost/coefficient signs, which Phase 3 now fixes
  // outright (a real, separate, correct reduction, just not what this test
  // means to isolate).
  core::SparseBuilder builder(3, 3);
  builder.count(0, 1);
  builder.count(1, 0);
  builder.count(1, 1);
  builder.count(1, 2);
  builder.count(2, 0);
  builder.count(2, 2);
  CHECK(builder.allocate().ok());
  builder.insert(0, 1, 1.0);
  builder.insert(1, 0, 1.0);
  builder.insert(1, 1, 1.0);
  builder.insert(1, 2, 2.0);
  builder.insert(2, 0, 1.0);
  builder.insert(2, 2, 1.0);

  model::CanonicalProblem problem;
  problem.A = builder.finish();
  problem.c = core::RealVector(3, 1.0);
  problem.b = core::RealVector(3);
  problem.b[0] = 3.0;
  problem.b[1] = 20.0;
  problem.b[2] = 100.0;
  problem.col_lower = core::RealVector(3, 0.0);
  problem.col_upper = core::RealVector(3, 10.0);
  problem.num_equality = 1;

  auto transforms = identity_stack(3, 3);
  model::Options options;
  CHECK(solver::presolve(problem, options, transforms).ok());

  // Row 0 folds to b=0 (3 - 1*3), reads empty and feasible, and is removed;
  // rows 1 and 2 (reindexed to 0 and 1) survive.
  CHECK_EQ(problem.num_rows(), std::size_t{2});
  CHECK_EQ(problem.num_cols(), std::size_t{2});  // x0, x2 survive; x1 fixed and removed
  CHECK_NEAR(problem.b[0], 17.0, 1e-12);          // 20 - 1*3, row 1's own fold
  CHECK_NEAR(problem.b[1], 100.0, 1e-12);         // row 2 untouched by x1's fold
  CHECK_NEAR(problem.obj_offset, 3.0, 1e-12);     // c[1]*3

  bool found = false;
  for (const auto& rec : transforms.records()) {
    if (rec.kind == model::TransformKind::RemoveFixedVariable && rec.primary == 1) {
      found = true;
      CHECK_NEAR(rec.value, 3.0, 1e-12);
    }
  }
  CHECK(found);
}

void test_singleton_row_implies_infeasible() {
  // Row 1 (inequality, singleton): x1 <= 3, but x1's own lower bound is
  // already 5 -- the implied upper bound (3) is below the existing lower
  // bound (5): infeasible.
  core::SparseBuilder builder(2, 2);
  builder.count(0, 0);
  builder.count(0, 1);
  builder.count(1, 1);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 1.0);
  builder.insert(0, 1, 1.0);
  builder.insert(1, 1, 1.0);

  model::CanonicalProblem problem;
  problem.A = builder.finish();
  problem.c = core::RealVector(2, 1.0);
  problem.b = core::RealVector(2);
  problem.b[0] = 5.0;
  problem.b[1] = 3.0;
  problem.col_lower = core::RealVector(2);
  problem.col_lower[0] = 0.0;
  problem.col_lower[1] = 5.0;
  problem.col_upper = core::RealVector(2, 10.0);
  problem.num_equality = 1;

  auto transforms = identity_stack(2, 2);
  model::Options options;
  const auto status = solver::presolve(problem, options, transforms);
  CHECK(!status.ok());
  CHECK(status.error().code == core::ErrorCode::PrimalInfeasible);
}

void test_empty_column_zero_cost_is_fixed() {
  // Column 1 has zero cost and no entries in A -- inert, must be fixed (at
  // its lower bound, since one exists) and removed. Row 0 deliberately
  // references TWO other columns (0 and 2), not one -- keeping row 0 a
  // non-singleton so this test exercises the empty-column rule in
  // isolation, without also tripping the singleton-row rule (which would,
  // correctly, go on to fix column 0 too if row 0 had only one entry left --
  // a real interaction covered by its own tests, not this one's concern).
  // Column 2's coefficient is deliberately 2.0, not 1.0 -- equal to column
  // 0's would make them duplicate columns (same pattern, same cost), which
  // Phase 2 now merges, cascading this test well past its own intent.
  core::SparseBuilder builder(1, 3);
  builder.count(0, 0);
  builder.count(0, 2);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 1.0);
  builder.insert(0, 2, 2.0);

  model::CanonicalProblem problem;
  problem.A = builder.finish();
  problem.c = core::RealVector(3, 0.0);
  problem.c[0] = 1.0;
  problem.c[2] = 1.0;
  problem.b = core::RealVector(1, 4.0);
  problem.col_lower = core::RealVector(3, 0.0);
  problem.col_lower[1] = 2.0;
  problem.col_upper = core::RealVector(3, 10.0);
  problem.num_equality = 1;

  auto transforms = identity_stack(1, 3);
  model::Options options;
  CHECK(solver::presolve(problem, options, transforms).ok());

  CHECK_EQ(problem.num_cols(), std::size_t{2});
  CHECK_EQ(problem.num_rows(), std::size_t{1});

  bool found = false;
  for (const auto& rec : transforms.records()) {
    if (rec.kind == model::TransformKind::RemoveFixedVariable && rec.primary == 1) {
      found = true;
      CHECK_NEAR(rec.value, 2.0, 1e-12);
    }
  }
  CHECK(found);
}

void test_empty_column_unbounded_direction_is_unbounded() {
  // Column 0 has negative cost, no entries in A, and an infinite upper
  // bound: minimizing c0*x0 with c0 < 0 improves without limit as x0 -> +inf.
  core::SparseBuilder builder(1, 1);
  CHECK(builder.allocate().ok());  // zero counted entries -- column 0 is empty

  model::CanonicalProblem problem;
  problem.A = builder.finish();
  problem.c = core::RealVector(1, -1.0);
  problem.b = core::RealVector(1, 0.0);
  problem.col_lower = core::RealVector(1, 0.0);
  problem.col_upper = core::RealVector(1, INF);
  problem.num_equality = 1;

  auto transforms = identity_stack(1, 1);
  model::Options options;
  const auto status = solver::presolve(problem, options, transforms);
  CHECK(!status.ok());
  CHECK(status.error().code == core::ErrorCode::Unbounded);
}

void test_free_column_singleton_round_trip() {
  // min x0 + 2*x1 + 3*x2   s.t.   x0 + x1 + x2 = 10   (row 0: x2's one row)
  //                               x0 - x1       = 1   (row 1: untouched)
  //                        0 <= x0,x1 <= 20, x2 free
  //
  // x2 is free and appears only in row 0 -- presolve must eliminate row 0
  // and column 2 together, folding x2's cost into the remaining objective.
  // A direct solve of the ORIGINAL 2-row problem gives x0=5.5, x1=4.5,
  // x2=0 (from row 0). This test builds that SAME point as the REDUCED
  // problem's canonical solution (row 1 only, columns 0/1 only) and checks
  // recover_solution reconstructs x2, y0, and the objective correctly --
  // the reduced problem never sees column 2 or row 0 at all.
  core::SparseBuilder builder(2, 3);
  builder.count(0, 0);
  builder.count(0, 1);
  builder.count(0, 2);
  builder.count(1, 0);
  builder.count(1, 1);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 1.0);
  builder.insert(0, 1, 1.0);
  builder.insert(0, 2, 1.0);
  builder.insert(1, 0, 1.0);
  builder.insert(1, 1, -1.0);

  model::Problem original;
  original.sense = core::ObjSense::Minimize;
  original.A = builder.finish();
  original.c = core::RealVector(3);
  original.c[0] = 1.0;
  original.c[1] = 2.0;
  original.c[2] = 3.0;
  original.row_lower = core::RealVector(2);
  original.row_upper = core::RealVector(2);
  original.row_lower[0] = original.row_upper[0] = 10.0;
  original.row_lower[1] = original.row_upper[1] = 1.0;
  original.col_lower = core::RealVector(3, 0.0);
  original.col_upper = core::RealVector(3, 20.0);
  original.col_lower[2] = -INF;
  original.col_upper[2] = INF;

  model::Options options;
  auto canon = model::canonicalize(original, options);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;

  CHECK(solver::presolve(canon->problem, options, canon->transforms).ok());

  // Row 0 and column 2 must be gone; row 1 and columns 0/1 remain.
  CHECK_EQ(canon->problem.num_rows(), std::size_t{1});
  CHECK_EQ(canon->problem.num_cols(), std::size_t{2});

  // x0 - x1 = 1, 0<=x0,x1<=20: any point on that line is "canonically
  // feasible" for this reduced row; pick the one matching the original
  // problem's actual solution (x0=5.5, x1=4.5) so recovery can be checked
  // against a known answer rather than an arbitrary feasible point.
  model::Solution reduced;
  reduced.status = core::SolverStatus::Optimal;
  reduced.x = core::RealVector(2);
  reduced.x[0] = 5.5;
  reduced.x[1] = 4.5;
  reduced.y = core::RealVector(1, 0.0);
  reduced.z = core::RealVector(2, 0.0);
  reduced.v = core::RealVector(2, 0.0);
  reduced.s = core::RealVector(1, 0.0);

  auto recovered =
      model::recover_solution(original, canon->problem, canon->transforms, reduced);
  CHECK(recovered.has_value());
  if (!recovered.has_value()) return;

  CHECK_EQ(recovered->x.size(), std::size_t{3});
  CHECK_NEAR(recovered->x[0], 5.5, 1e-9);
  CHECK_NEAR(recovered->x[1], 4.5, 1e-9);
  CHECK_NEAR(recovered->x[2], 0.0, 1e-9);  // 10 - 5.5 - 4.5

  // y0 = c2 / a_02 = 3 / 1 = 3 (stationarity, free column, singleton row).
  CHECK_NEAR(recovered->y[0], 3.0, 1e-9);

  // Objective must match c'x over the ORIGINAL, unreduced problem exactly --
  // this is the check that would fail if the objective-folding fix
  // (Presolver.cpp's factor/obj_offset adjustment) were missing or wrong.
  const Real expected_obj = 1.0 * 5.5 + 2.0 * 4.5 + 3.0 * 0.0;
  CHECK_NEAR(recovered->objective, expected_obj, 1e-9);
}

void test_duplicate_column_merge_round_trip() {
  // min 2*x0 + 2*x1 + x2   s.t.   x0 + x1 + 2*x2 = 6
  //                        0<=x0<=4, 0<=x1<=5, 0<=x2<=10
  //
  // x0/x1 have an identical A pattern (row 0, coefficient 1.0) AND identical
  // cost (2.0) -- Presolver must merge them into one column (survivor x0,
  // widened to [0,9]) rather than eliminate row 0 outright: x2's DIFFERENT
  // coefficient (2.0) keeps row 0 at two entries after the merge, so it does
  // not degenerate into a singleton row Phase 1 would go on to fix.
  //
  // A direct solve of the ORIGINAL problem at x0=0, x1=3, x2=1.5 (any split
  // of x0+x1=3 works -- x0/x1 share a cost, so the objective genuinely does
  // not care which one holds how much) satisfies row 0 exactly (0+3+2*1.5=6).
  // This test builds that SAME point as the REDUCED problem's canonical
  // solution (row 0, merged column y=x0+x1=3, plus x2) and checks
  // recover_solution splits y back into x0=0 (at ITS OWN lower bound) and
  // x1=3 (interior -- no dual of its own), and assigns the shared
  // stationarity value to x0 alone.
  core::SparseBuilder builder(1, 3);
  builder.count(0, 0);
  builder.count(0, 1);
  builder.count(0, 2);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 1.0);
  builder.insert(0, 1, 1.0);
  builder.insert(0, 2, 2.0);

  model::Problem original;
  original.sense = core::ObjSense::Minimize;
  original.A = builder.finish();
  original.c = core::RealVector(3);
  original.c[0] = 2.0;
  original.c[1] = 2.0;
  original.c[2] = 1.0;
  original.row_lower = core::RealVector(1, 6.0);
  original.row_upper = core::RealVector(1, 6.0);
  original.col_lower = core::RealVector(3, 0.0);
  original.col_upper = core::RealVector(3);
  original.col_upper[0] = 4.0;
  original.col_upper[1] = 5.0;
  original.col_upper[2] = 10.0;

  model::Options options;
  auto canon = model::canonicalize(original, options);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;

  CHECK(solver::presolve(canon->problem, options, canon->transforms).ok());

  // x1 merged away; row 0 survives (x2's mismatched coefficient stops it
  // from degenerating into a singleton Phase 1 would go on to fix).
  CHECK_EQ(canon->problem.num_rows(), std::size_t{1});
  CHECK_EQ(canon->problem.num_cols(), std::size_t{2});

  bool found_merge = false;
  for (const auto& rec : canon->transforms.records()) {
    if (rec.kind == model::TransformKind::MergeDuplicateColumn) {
      found_merge = true;
      CHECK_EQ(rec.primary, core::Index{1});    // x1 dropped
      CHECK_EQ(rec.secondary, core::Index{0});  // x0 survives
    }
  }
  CHECK(found_merge);
  CHECK_NEAR(canon->problem.col_lower[0], 0.0, 1e-12);
  CHECK_NEAR(canon->problem.col_upper[0], 9.0, 1e-12);  // 4 + 5, Minkowski sum

  model::Solution reduced;
  reduced.status = core::SolverStatus::Optimal;
  reduced.x = core::RealVector(2);
  reduced.x[0] = 3.0;  // merged y = x0 + x1
  reduced.x[1] = 1.5;  // x2
  reduced.y = core::RealVector(1, 1.0);
  reduced.z = core::RealVector(2, 0.0);  // overwritten for column 0 either way
  reduced.v = core::RealVector(2, 0.0);
  reduced.s = core::RealVector(1, 0.0);

  auto recovered =
      model::recover_solution(original, canon->problem, canon->transforms, reduced);
  CHECK(recovered.has_value());
  if (!recovered.has_value()) return;

  CHECK_EQ(recovered->x.size(), std::size_t{3});
  CHECK_NEAR(recovered->x[0], 0.0, 1e-9);  // x0 at its OWN lower bound
  CHECK_NEAR(recovered->x[1], 3.0, 1e-9);  // x1 = y - x0, interior
  CHECK_NEAR(recovered->x[2], 1.5, 1e-9);

  // d = c0 - a_00*y0 = 2.0 - 1.0*1.0 = 1.0 >= 0 -- x0 sits at its own lower
  // bound, so z0 = d; x1 is interior (not at either of ITS bounds), so it
  // gets no dual of its own even though it shares x0's stationarity value.
  CHECK_NEAR(recovered->z[0], 1.0, 1e-9);
  CHECK_NEAR(recovered->v[0], 0.0, 1e-9);
  CHECK_NEAR(recovered->z[1], 0.0, 1e-9);
  CHECK_NEAR(recovered->v[1], 0.0, 1e-9);

  const Real expected_obj = 2.0 * 0.0 + 2.0 * 3.0 + 1.0 * 1.5;
  CHECK_NEAR(recovered->objective, expected_obj, 1e-9);
}

void test_duplicate_column_candidate_with_different_cost_is_not_merged() {
  // Same A pattern as the merge test (row 0: both coefficient 1.0), but
  // DIFFERENT cost -- the hash covers only the matrix, never the objective,
  // so this pair is a hash "candidate" Presolver must still reject: merging
  // columns with different costs would silently change which one the
  // optimizer prefers.
  core::SparseBuilder builder(1, 2);
  builder.count(0, 0);
  builder.count(0, 1);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 1.0);
  builder.insert(0, 1, 1.0);

  model::CanonicalProblem problem;
  problem.A = builder.finish();
  problem.c = core::RealVector(2);
  problem.c[0] = 1.0;
  problem.c[1] = 2.0;  // different from c[0] -- must block the merge
  problem.b = core::RealVector(1, 6.0);
  problem.col_lower = core::RealVector(2, 0.0);
  problem.col_upper = core::RealVector(2, 10.0);
  problem.num_equality = 1;

  auto transforms = identity_stack(1, 2);
  model::Options options;
  CHECK(solver::presolve(problem, options, transforms).ok());

  CHECK_EQ(problem.num_rows(), std::size_t{1});
  CHECK_EQ(problem.num_cols(), std::size_t{2});
  for (const auto& rec : transforms.records()) {
    CHECK(rec.kind != model::TransformKind::MergeDuplicateColumn);
  }
}

void test_singleton_column_zero_cost_maximizes_slack() {
  // Row 0 (inequality): 2*x0 <= 10, x0 in [3,8], c0=0 -- the objective is
  // indifferent to x0, so Presolver picks x0=3 (its OWN lower bound, since
  // a_00>0 minimizes 2*x0 there, maximizing whatever slack the row has left
  // for other columns -- here there are none, so the row folds straight to
  // empty and is removed on the very next internal pass, all within this
  // one presolve() call).
  core::SparseBuilder builder(1, 1);
  builder.count(0, 0);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 2.0);

  model::CanonicalProblem problem;
  problem.A = builder.finish();
  problem.c = core::RealVector(1, 0.0);
  problem.b = core::RealVector(1, 10.0);
  problem.col_lower = core::RealVector(1, 3.0);
  problem.col_upper = core::RealVector(1, 8.0);
  problem.num_equality = 0;  // row 0 is inequality

  auto transforms = identity_stack(1, 1);
  model::Options options;
  CHECK(solver::presolve(problem, options, transforms).ok());

  CHECK_EQ(problem.num_rows(), std::size_t{0});
  CHECK_EQ(problem.num_cols(), std::size_t{0});
  CHECK_NEAR(problem.obj_offset, 0.0, 1e-12);

  bool found = false;
  for (const auto& rec : transforms.records()) {
    if (rec.kind == model::TransformKind::RemoveFixedVariable && rec.primary == 0) {
      found = true;
      CHECK_NEAR(rec.value, 3.0, 1e-12);
    }
  }
  CHECK(found);
}

void test_singleton_column_sign_matched_cost_is_fixed() {
  // Row 0 (inequality): 3*x0 <= 20, x0 in [1,5], c0=2.0 -- c0>0 and a_00>0
  // both prefer x0 SMALL, so the SAME slack-maximizing choice (Case A above)
  // is ALSO cost-improving here: fix x0 at its lower bound unconditionally,
  // a pure win with no feasibility cost to row 0.
  core::SparseBuilder builder(1, 1);
  builder.count(0, 0);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 3.0);

  model::CanonicalProblem problem;
  problem.A = builder.finish();
  problem.c = core::RealVector(1, 2.0);
  problem.b = core::RealVector(1, 20.0);
  problem.col_lower = core::RealVector(1, 1.0);
  problem.col_upper = core::RealVector(1, 5.0);
  problem.num_equality = 0;

  auto transforms = identity_stack(1, 1);
  model::Options options;
  CHECK(solver::presolve(problem, options, transforms).ok());

  CHECK_EQ(problem.num_rows(), std::size_t{0});
  CHECK_EQ(problem.num_cols(), std::size_t{0});
  CHECK_NEAR(problem.obj_offset, 2.0, 1e-12);  // c0 * 1

  bool found = false;
  for (const auto& rec : transforms.records()) {
    if (rec.kind == model::TransformKind::RemoveFixedVariable && rec.primary == 0) {
      found = true;
      CHECK_NEAR(rec.value, 1.0, 1e-12);
    }
  }
  CHECK(found);
}

void test_singleton_column_sign_matched_unbounded_direction_is_unbounded() {
  // Row 0 (inequality): x0 <= 10, x0 in (-inf, 10], c0=3.0 -- c0>0 wants x0
  // as small as possible, which never costs row 0 any slack either (a_00>0
  // wants the same thing), but x0 has NO lower bound to stop it: a second,
  // genuine Unbounded certificate, distinct from the empty-column one, this
  // time sourced from a row-bounded rather than totally free column.
  core::SparseBuilder builder(1, 1);
  builder.count(0, 0);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 1.0);

  model::CanonicalProblem problem;
  problem.A = builder.finish();
  problem.c = core::RealVector(1, 3.0);
  problem.b = core::RealVector(1, 10.0);
  problem.col_lower = core::RealVector(1, -INF);
  problem.col_upper = core::RealVector(1, 10.0);
  problem.num_equality = 0;

  auto transforms = identity_stack(1, 1);
  model::Options options;
  const auto status = solver::presolve(problem, options, transforms);
  CHECK(!status.ok());
  CHECK(status.error().code == core::ErrorCode::Unbounded);
}

void test_singleton_column_sign_mismatched_is_untouched() {
  // Row 0 (inequality): x0 <= 10, x0 in [0,5], c0=-2.0 -- a_00>0 wants x0
  // SMALL (more row slack), but c0<0 wants x0 LARGE (lower cost): genuinely
  // coupled to the rest of the problem, not reducible by this rule at all,
  // so x0 (and row 0) must survive untouched.
  core::SparseBuilder builder(1, 1);
  builder.count(0, 0);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 1.0);

  model::CanonicalProblem problem;
  problem.A = builder.finish();
  problem.c = core::RealVector(1, -2.0);
  problem.b = core::RealVector(1, 10.0);
  problem.col_lower = core::RealVector(1, 0.0);
  problem.col_upper = core::RealVector(1, 5.0);
  problem.num_equality = 0;

  auto transforms = identity_stack(1, 1);
  model::Options options;
  CHECK(solver::presolve(problem, options, transforms).ok());

  CHECK_EQ(problem.num_rows(), std::size_t{1});
  CHECK_EQ(problem.num_cols(), std::size_t{1});
  for (const auto& rec : transforms.records()) {
    CHECK(rec.kind != model::TransformKind::RemoveFixedVariable);
  }
}

void test_disabled_presolve_is_a_no_op() {
  core::SparseBuilder builder(2, 2);
  builder.count(0, 0);
  builder.count(0, 1);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 1.0);
  builder.insert(0, 1, 1.0);

  model::CanonicalProblem problem;
  problem.A = builder.finish();
  problem.c = core::RealVector(2, 1.0);
  problem.b = core::RealVector(2, 0.0);
  problem.b[0] = 5.0;
  problem.col_lower = core::RealVector(2, 0.0);
  problem.col_upper = core::RealVector(2, 10.0);
  problem.num_equality = 2;

  auto transforms = identity_stack(2, 2);
  model::Options options;
  options.presolve.enabled = false;
  CHECK(solver::presolve(problem, options, transforms).ok());

  // Row 1 is empty and would normally be dropped -- disabled means untouched.
  CHECK_EQ(problem.num_rows(), std::size_t{2});
  CHECK_EQ(problem.num_cols(), std::size_t{2});
}

}  // namespace

int main() {
  test_empty_row_consistent_is_removed();
  test_empty_row_inconsistent_is_infeasible();
  test_empty_inequality_row_with_positive_slack_is_feasible();
  test_empty_inequality_row_with_negative_rhs_is_infeasible();
  test_singleton_row_tightens_without_fixing();
  test_singleton_row_collapses_to_fix();
  test_singleton_row_fix_folds_its_own_row();
  test_singleton_row_implies_infeasible();
  test_empty_column_zero_cost_is_fixed();
  test_empty_column_unbounded_direction_is_unbounded();
  test_free_column_singleton_round_trip();
  test_duplicate_column_merge_round_trip();
  test_duplicate_column_candidate_with_different_cost_is_not_merged();
  test_singleton_column_zero_cost_maximizes_slack();
  test_singleton_column_sign_matched_cost_is_fixed();
  test_singleton_column_sign_matched_unbounded_direction_is_unbounded();
  test_singleton_column_sign_mismatched_is_untouched();
  test_disabled_presolve_is_a_no_op();
  return sovsolve::test::report("presolver");
}
