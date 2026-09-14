// Module 5, with Module 24's addition: Ruiz + Pock-Chambolle equilibration
// alongside the existing geometric-mean scaling.
//
// Two things are worth asserting about a scaler, and they are different kinds
// of claim:
//
//   1. The equilibration does what its definition says. Ruiz's own result is
//      that iterating the rescaling drives every row and column infinity norm
//      to 1, so that is a property the code either has or does not -- checked
//      here directly on the scaled matrix, not inferred from a solve.
//
//   2. Scaling changes nothing observable. A scaled problem is a different
//      matrix with the same solution set, so both modes must produce the same
//      objective on a real instance. This is the check that catches a scale
//      applied to `A` but not to `b`, or to a bound in the wrong direction --
//      errors that leave the matrix beautifully equilibrated and the answer
//      wrong.

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/solver/LpSolve.hpp"
#include "sovsolve/solver/Presolver.hpp"
#include "sovsolve/solver/Scaler.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
using core::Real;
using core::SolverStatus;
using model::Options;
using model::ScalingMode;

namespace {

Options options_with(ScalingMode mode, model::Method method) {
  Options o;
  o.scaling.mode = mode;
  o.simplex.method = method;
  o.log.level = model::LogOptions::Level::Silent;
  return o;
}

/// Canonicalize + presolve + scale a real instance, and hand back the scaled
/// canonical problem so its norms can be measured directly.
bool prepare(const std::string& path, const Options& options,
             model::CanonicalResult& out) {
  auto loaded = io::loadProblem(path);
  if (!loaded.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "load instance",
                             loaded.error().format());
    return false;
  }
  auto canon = model::canonicalize(loaded.value(), options);
  if (!canon.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "canonicalize",
                             canon.error().format());
    return false;
  }
  out = std::move(canon.value());
  core::Status st = solver::presolve(out.problem, options, out.transforms);
  if (!st.ok()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "presolve", st.error().format());
    return false;
  }
  st = solver::scale(out.problem, options, out.transforms);
  if (!st.ok()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "scale", st.error().format());
    return false;
  }
  return true;
}

Real max_row_inf_deviation(const model::CanonicalProblem& p) {
  const auto& csr = p.A.csr;
  Real worst = 0.0;
  for (std::size_t i = 0; i < p.num_rows(); ++i) {
    Real norm = 0.0;
    for (std::size_t k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      norm = std::max(norm, std::fabs(csr.values()[k]));
    }
    if (norm > 0.0) worst = std::max(worst, std::fabs(norm - 1.0));
  }
  return worst;
}

Real max_col_inf_deviation(const model::CanonicalProblem& p) {
  const auto& csc = p.A.csc;
  Real worst = 0.0;
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    Real norm = 0.0;
    for (std::size_t k = csc.slice_begin(j); k < csc.slice_end(j); ++k) {
      norm = std::max(norm, std::fabs(csc.values()[k]));
    }
    if (norm > 0.0) worst = std::max(worst, std::fabs(norm - 1.0));
  }
  return worst;
}

/// Largest / smallest nonzero magnitude in `A` -- the dynamic range the
/// downstream solver actually sees.
Real dynamic_range(const model::CanonicalProblem& p) {
  const auto& csr = p.A.csr;
  Real lo = 0.0;
  Real hi = 0.0;
  bool any = false;
  for (std::size_t k = 0; k < csr.values().size(); ++k) {
    const Real v = std::fabs(csr.values()[k]);
    if (v <= 0.0) continue;
    if (!any) {
      lo = v;
      hi = v;
      any = true;
    } else {
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
  }
  return any ? hi / lo : 1.0;
}

// --------------------------------------------------------------------------

/// Ruiz's defining property, checked directly: iterating the rescaling drives
/// every row and column infinity norm of `A` to 1.
///
/// The Pock-Chambolle pass is switched off here, because it deliberately moves
/// the norms OFF that fixed point -- measuring after it runs would test
/// nothing. (A first version of this test did exactly that and reported the
/// combined pipeline as a regression; the premise was wrong, not the code.)
void test_ruiz_drives_norms_to_one() {
  auto deviations = [](std::size_t iters, Real& row_dev, Real& col_dev) -> bool {
    Options o = options_with(ScalingMode::RuizPockChambolle, model::Method::DualSimplex);
    o.scaling.ruiz_iterations = iters;
    o.scaling.pock_chambolle = false;
    model::CanonicalResult r;
    if (!prepare(std::string(SOVSOLVE_TEST_DATA_DIR) + "/netlib/afiro.mps", o, r)) {
      return false;
    }
    row_dev = max_row_inf_deviation(r.problem);
    col_dev = max_col_inf_deviation(r.problem);
    return true;
  };

  Real row0 = 0.0;
  Real col0 = 0.0;
  Real row10 = 0.0;
  Real col10 = 0.0;
  Real row30 = 0.0;
  Real col30 = 0.0;
  if (!deviations(0, row0, col0)) return;
  if (!deviations(10, row10, col10)) return;
  if (!deviations(30, row30, col30)) return;

  // Zero iterations means no scaling at all, so afiro's raw norms are far
  // from 1 -- this is the baseline the fixed point is measured against. If
  // this ever passes trivially the rest of the test proves nothing.
  ++::sovsolve::test::checks_run();
  if (!(row0 > 1e-3)) {
    ::sovsolve::test::record(__FILE__, __LINE__, "unscaled afiro is not equilibrated",
                             "row deviation " + std::to_string(row0));
  }

  // Ruiz converges LINEARLY, so 10 iterations -- PDLP's own choice -- lands
  // around 1e-4 rather than at machine precision. Demanding 1e-6 here would
  // be demanding something neither Ruiz nor PDLP claims. What is checked is
  // the claim that IS made: iterating drives the norms toward 1, by orders of
  // magnitude, and continues to.
  ++::sovsolve::test::checks_run();
  if (!(row10 < row0 / 1000.0)) {
    ::sovsolve::test::record(__FILE__, __LINE__,
                             "10 Ruiz iterations: row inf-norms approach 1",
                             "unscaled " + std::to_string(row0) + " -> " +
                                 std::to_string(row10));
  }
  ++::sovsolve::test::checks_run();
  if (!(col10 < col0 / 1000.0)) {
    ::sovsolve::test::record(__FILE__, __LINE__,
                             "10 Ruiz iterations: column inf-norms approach 1",
                             "unscaled " + std::to_string(col0) + " -> " +
                                 std::to_string(col10));
  }

  // Still converging at 30, which is what distinguishes "approaching 1" from
  // "stuck at some other fixed point that happens to be closer".
  ++::sovsolve::test::checks_run();
  if (!(row30 < row10)) {
    ::sovsolve::test::record(__FILE__, __LINE__, "Ruiz keeps converging on rows",
                             "10 iters " + std::to_string(row10) + ", 30 iters " +
                                 std::to_string(row30));
  }
  ++::sovsolve::test::checks_run();
  if (!(col30 < col10)) {
    ::sovsolve::test::record(__FILE__, __LINE__, "Ruiz keeps converging on columns",
                             "10 iters " + std::to_string(col10) + ", 30 iters " +
                                 std::to_string(col30));
  }
}

/// Ruiz must also bound the dynamic range the downstream solver sees, which is
/// the property the scaling exists for in the first place.
void test_ruiz_bounds_dynamic_range() {
  Options o = options_with(ScalingMode::RuizPockChambolle, model::Method::DualSimplex);
  model::CanonicalResult ruiz;
  if (!prepare(std::string(SOVSOLVE_TEST_DATA_DIR) + "/netlib/afiro.mps", o, ruiz)) {
    return;
  }

  Options raw = o;
  raw.scaling.mode = ScalingMode::RuizPockChambolle;
  raw.scaling.ruiz_iterations = 0;
  raw.scaling.pock_chambolle = false;
  model::CanonicalResult unscaled;
  if (!prepare(std::string(SOVSOLVE_TEST_DATA_DIR) + "/netlib/afiro.mps", raw, unscaled)) {
    return;
  }

  const Real after = dynamic_range(ruiz.problem);
  const Real before = dynamic_range(unscaled.problem);
  ++::sovsolve::test::checks_run();
  if (!(after < before)) {
    ::sovsolve::test::record(__FILE__, __LINE__, "scaling narrows the dynamic range",
                             "before " + std::to_string(before) + ", after " +
                                 std::to_string(after));
  }
}

/// The check that actually matters: scaling must not change the answer.
///
/// A scale factor applied to `A` but forgotten on `b`, or divided into a bound
/// that should have been multiplied, leaves a perfectly equilibrated matrix
/// and a wrong optimum. Only an end-to-end solve catches that, and only
/// because postsolve has to undo the scaling to report in original units.
void test_objective_is_scaling_invariant() {
  struct Case {
    const char* file;
    Real expected;
  };
  const Case cases[] = {
      {"/netlib/afiro.mps", -464.75314286},
      {"/netlib/adlittle.mps", 225494.96316},
  };

  for (const Case& c : cases) {
    for (const model::Method method :
         {model::Method::DualSimplex, model::Method::PrimalSimplex}) {
      for (const ScalingMode mode :
           {ScalingMode::GeometricMean, ScalingMode::RuizPockChambolle}) {
        const Options o = options_with(mode, method);
        auto loaded = io::loadProblem(std::string(SOVSOLVE_TEST_DATA_DIR) + c.file);
        if (!loaded.has_value()) {
          ::sovsolve::test::record(__FILE__, __LINE__, "load", loaded.error().format());
          continue;
        }
        auto result = solver::solve_lp(loaded.value(), o);
        if (!result.has_value()) {
          ::sovsolve::test::record(__FILE__, __LINE__, "solve", result.error().format());
          continue;
        }
        CHECK(result->status == SolverStatus::Optimal);
        ++::sovsolve::test::checks_run();
        if (!::sovsolve::test::close(result->objective, c.expected, 1e-6)) {
          ::sovsolve::test::record(
              __FILE__, __LINE__, "objective independent of scaling mode",
              std::string(c.file) + " mode=" +
                  (mode == ScalingMode::GeometricMean ? "geometric" : "ruiz") +
                  ": got " + std::to_string(result->objective) + ", expected " +
                  std::to_string(c.expected));
        }
      }
    }
  }
}

}  // namespace

int main() {
  test_ruiz_drives_norms_to_one();
  test_ruiz_bounds_dynamic_range();
  test_objective_is_scaling_invariant();
  return ::sovsolve::test::report("scaler_test");
}
