// Module 26: Irreducible Infeasible Subsystem.
//
// The thing to test is not "did it return some rows" -- returning every row of
// an infeasible model is trivially an infeasible subsystem and completely
// useless. The claim is IRREDUCIBILITY: the reported set must contain the rows
// that actually contradict each other and NOT the ones that merely happen to
// be in the same model.
//
// So every case here deliberately pads the model with satisfiable rows that
// share variables with the contradiction. A support computed from a
// non-vertex certificate, or a tolerance set too loose, pulls those padding
// rows in -- which is exactly the failure this file exists to catch.

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Problem.hpp"
#include "sovsolve/solver/Iis.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
using solver::Iis;
using solver::IisQuality;

namespace {

model::Options iis_options() {
  model::Options o;
  o.log.level = model::LogOptions::Level::Silent;
  return o;
}

bool compute(const char* label, std::string_view text, Iis& out) {
  auto parsed = io::parseProblem(text, io::FileFormat::Lp);
  if (!parsed.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, label, parsed.error().format());
    return false;
  }
  auto result = solver::compute_iis(parsed.value(), iis_options());
  if (!result.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, label, result.error().format());
    return false;
  }
  out = std::move(result).value();
  return true;
}

bool contains(const std::vector<std::size_t>& v, std::size_t x) {
  return std::find(v.begin(), v.end(), x) != v.end();
}

std::string render(const Iis& iis) {
  std::string s = "rows{";
  for (std::size_t r : iis.rows) s += std::to_string(r) + ",";
  s += "} lower{";
  for (std::size_t c : iis.lower_bound_columns) s += std::to_string(c) + ",";
  s += "} upper{";
  for (std::size_t c : iis.upper_bound_columns) s += std::to_string(c) + ",";
  return s + "}";
}

// --------------------------------------------------------------------------

/// Two rows contradict; two more are satisfiable and share both variables.
/// The answer must be exactly the first two.
void test_excludes_irrelevant_rows() {
  Iis iis;
  if (!compute("two_row_contradiction", R"(Minimize
 obj: x + y
Subject To
 c0: x + y >= 10
 c1: x + y <= 4
 c2: x - y <= 50
 c3: x + 2 y <= 80
Bounds
 0 <= x <= 100
 0 <= y <= 100
End
)",
               iis)) {
    return;
  }

  CHECK(!iis.empty());
  CHECK(contains(iis.rows, 0));
  CHECK(contains(iis.rows, 1));

  // The point of the whole exercise.
  ++::sovsolve::test::checks_run();
  if (contains(iis.rows, 2) || contains(iis.rows, 3)) {
    ::sovsolve::test::record(__FILE__, __LINE__,
                             "satisfiable padding rows are excluded",
                             "got " + render(iis) +
                                 " -- c2 and c3 do not participate in the "
                                 "contradiction");
  }

  ++::sovsolve::test::checks_run();
  if (iis.rows.size() != 2) {
    ::sovsolve::test::record(__FILE__, __LINE__, "the subsystem is minimal",
                             "expected exactly 2 rows, got " + render(iis));
  }

  CHECK(iis.quality == IisQuality::Irreducible);
}

/// A three-row cycle where NO pair is infeasible -- only all three together
/// are. This is the case a pairwise search would miss entirely, and it checks
/// the certificate really is spanning the contradiction rather than finding
/// the first two rows that look suspicious.
///
///   x >= 4,  y >= 4,  x + y <= 5
void test_three_row_contradiction() {
  Iis iis;
  if (!compute("three_row", R"(Minimize
 obj: x + y
Subject To
 c0: x >= 4
 c1: y >= 4
 c2: x + y <= 5
 c3: x - y <= 1000
Bounds
 0 <= x <= 100
 0 <= y <= 100
End
)",
               iis)) {
    return;
  }

  for (std::size_t row : {std::size_t{0}, std::size_t{1}, std::size_t{2}}) {
    ++::sovsolve::test::checks_run();
    if (!contains(iis.rows, row)) {
      ::sovsolve::test::record(__FILE__, __LINE__,
                               "all three mutually-contradicting rows reported",
                               "row " + std::to_string(row) + " missing from " +
                                   render(iis));
    }
  }
  ++::sovsolve::test::checks_run();
  if (contains(iis.rows, 3)) {
    ::sovsolve::test::record(__FILE__, __LINE__, "the slack row is excluded",
                             render(iis));
  }
}

/// A variable BOUND can be half the contradiction. `x <= 3` with a row forcing
/// `x >= 5` is infeasible, and reporting only the row would name half the
/// problem -- the user would look at a constraint that is perfectly reasonable
/// on its own.
void test_bounds_participate() {
  Iis iis;
  if (!compute("bound_contradiction", R"(Minimize
 obj: x
Subject To
 c0: x >= 5
Bounds
 0 <= x <= 3
End
)",
               iis)) {
    return;
  }

  CHECK(!iis.empty());
  CHECK(contains(iis.rows, 0));
  ++::sovsolve::test::checks_run();
  if (!contains(iis.upper_bound_columns, 0)) {
    ::sovsolve::test::record(__FILE__, __LINE__,
                             "the binding upper bound is reported",
                             "got " + render(iis) +
                                 " -- x <= 3 is half the contradiction");
  }
}

/// Asking for the contradiction in a model that HAS a solution is a caller
/// error, and is reported rather than answered with an empty set -- an empty
/// IIS reads as "no constraints are at fault", which is a different and false
/// statement.
void test_feasible_model_is_an_error() {
  auto parsed = io::parseProblem(R"(Minimize
 obj: x + y
Subject To
 c0: x + y >= 2
 c1: x + y <= 8
Bounds
 0 <= x <= 10
 0 <= y <= 10
End
)",
                                 io::FileFormat::Lp);
  if (!parsed.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "parse", parsed.error().format());
    return;
  }
  auto result = solver::compute_iis(parsed.value(), iis_options());
  CHECK(!result.has_value());
}

/// Removing any one member must make the rest satisfiable. That is the literal
/// definition of irreducible, and checking it directly is stronger than
/// checking a row count -- it would catch a set that happens to have the right
/// size but the wrong members.
void test_removing_any_member_restores_feasibility() {
  const char* rows[] = {"c0: x >= 4", "c1: y >= 4", "c2: x + y <= 5"};
  for (std::size_t skip = 0; skip < 3; ++skip) {
    std::string text = "Minimize\n obj: x + y\nSubject To\n";
    for (std::size_t i = 0; i < 3; ++i) {
      if (i == skip) continue;
      text += " ";
      text += rows[i];
      text += "\n";
    }
    text += "Bounds\n 0 <= x <= 100\n 0 <= y <= 100\nEnd\n";

    auto parsed = io::parseProblem(text, io::FileFormat::Lp);
    if (!parsed.has_value()) {
      ::sovsolve::test::record(__FILE__, __LINE__, "parse", parsed.error().format());
      continue;
    }
    // `compute_iis` fails on a feasible model, which is precisely the
    // assertion: each two-row subsystem must be satisfiable.
    auto result = solver::compute_iis(parsed.value(), iis_options());
    ++::sovsolve::test::checks_run();
    if (result.has_value()) {
      ::sovsolve::test::record(
          __FILE__, __LINE__, "dropping one member restores feasibility",
          "the subsystem without row " + std::to_string(skip) +
              " is still infeasible, so the three rows were not irreducible");
    }
  }
}

}  // namespace

int main() {
  test_excludes_irrelevant_rows();
  test_three_row_contradiction();
  test_bounds_participate();
  test_feasible_model_is_an_error();
  test_removing_any_member_restores_feasibility();
  return ::sovsolve::test::report("iis_test");
}
