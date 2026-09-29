// LP presolve (solver/LpPresolve.hpp): the optimum is preserved AND the
// postsolved point is a KKT point of the ORIGINAL model.
//
// A presolve bug rarely shows in the objective -- a wrong dual postsolve leaves
// the primal untouched. So each model is solved twice through `solve_lp`, with
// the LP presolve and with presolve off, and the presolved answer is held to
// the full optimality conditions on the original Problem:
//
//   primal   row activity inside [row_lower, row_upper], x inside its bounds
//   dual     sigma c - A'y - z + v = 0, z >= 0, v >= 0
//   signs    z > 0 only on a finite lower bound, v > 0 only on a finite upper,
//            y > 0 only on a finite row_lower, y < 0 only on a finite row_upper
//   compl.   each multiplier times the slack of the bound it sits on
//
// Every measure is compared with the unpresolved solve of the same engine, so
// the test asks "did presolve make it worse", not "is the engine perfect".
//
// Runs the fetched Netlib models under tests/data/netlib; set
// SOVSOLVE_LP_PRESOLVE_CORPUS to a directory of .mps files to run more.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Problem.hpp"
#include "sovsolve/model/Solution.hpp"
#include "sovsolve/solver/LpPresolve.hpp"
#include "sovsolve/solver/LpSolve.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
using core::is_finite_bound;
using core::Real;
using core::SolverStatus;
namespace fs = std::filesystem;

namespace {

struct Kkt {
  Real primal = 0.0;  ///< worst row or bound violation, relative to the bound
  Real dual = 0.0;    ///< worst stationarity residual, relative to |c_j|
  Real sign = 0.0;    ///< worst multiplier on a side that has no bound
  Real compl_ = 0.0;  ///< worst multiplier * slack, relative to |objective|
};

Kkt kkt(const model::Problem& p, const model::Solution& s) {
  Kkt k;
  const std::size_t m = p.num_rows();
  const std::size_t n = p.num_cols();
  const Real sigma = p.sense == core::ObjSense::Maximize ? -1.0 : 1.0;
  const Real obj_scale = 1.0 + std::fabs(s.objective);

  std::vector<Real> act(m, 0.0), aty(n, 0.0);
  const auto& csr = p.A.csr;
  for (std::size_t i = 0; i < m; ++i) {
    for (auto e = csr.slice_begin(i); e < csr.slice_end(i); ++e) {
      const auto j = static_cast<std::size_t>(csr.indices()[e]);
      act[i] += csr.values()[e] * s.x[j];
      aty[j] += csr.values()[e] * s.y[i];
    }
  }

  for (std::size_t i = 0; i < m; ++i) {
    const Real lo = p.row_lower[i], hi = p.row_upper[i];
    if (is_finite_bound(lo)) k.primal = std::max(k.primal, (lo - act[i]) / (1.0 + std::fabs(lo)));
    if (is_finite_bound(hi)) k.primal = std::max(k.primal, (act[i] - hi) / (1.0 + std::fabs(hi)));
    const Real y = s.y[i];
    if (y > 0.0) {
      if (!is_finite_bound(lo)) k.sign = std::max(k.sign, y);
      else k.compl_ = std::max(k.compl_, y * std::fabs(act[i] - lo) / obj_scale);
    } else if (y < 0.0) {
      if (!is_finite_bound(hi)) k.sign = std::max(k.sign, -y);
      else k.compl_ = std::max(k.compl_, -y * std::fabs(hi - act[i]) / obj_scale);
    }
  }
  for (std::size_t j = 0; j < n; ++j) {
    const Real lo = p.col_lower[j], hi = p.col_upper[j];
    if (is_finite_bound(lo)) k.primal = std::max(k.primal, (lo - s.x[j]) / (1.0 + std::fabs(lo)));
    if (is_finite_bound(hi)) k.primal = std::max(k.primal, (s.x[j] - hi) / (1.0 + std::fabs(hi)));
    const Real r = sigma * p.c[j] - aty[j] - s.z[j] + s.v[j];
    k.dual = std::max(k.dual, std::fabs(r) / (1.0 + std::fabs(p.c[j])));
    k.sign = std::max({k.sign, -s.z[j], -s.v[j]});
    if (s.z[j] > 0.0) {
      if (!is_finite_bound(lo)) k.sign = std::max(k.sign, s.z[j]);
      else k.compl_ = std::max(k.compl_, s.z[j] * std::fabs(s.x[j] - lo) / obj_scale);
    }
    if (s.v[j] > 0.0) {
      if (!is_finite_bound(hi)) k.sign = std::max(k.sign, s.v[j]);
      else k.compl_ = std::max(k.compl_, s.v[j] * std::fabs(hi - s.x[j]) / obj_scale);
    }
  }
  return k;
}

model::Options options_with(bool lp_presolve) {
  model::Options o;
  o.simplex.method = model::Method::DualSimplex;
  o.log.level = model::LogOptions::Level::Silent;
  o.presolve.enabled = lp_presolve;
  o.presolve.lp_reductions = lp_presolve;
  return o;
}

/// The engine's own error sets the floor; presolve may not add more than a
/// small factor on top of it.
bool within(Real presolved, Real baseline, Real floor) {
  return presolved <= std::max(floor, 10.0 * baseline);
}

void run_model(const fs::path& path) {
  auto loaded = io::loadProblem(path.string());
  if (!loaded.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "load", path.string());
    return;
  }
  const model::Problem& problem = *loaded;
  if (problem.has_discrete() || problem.has_quadratic()) return;
  const std::string name = path.stem().string();

  auto base = solver::solve_lp(problem, options_with(false));
  auto pre = solver::solve_lp(problem, options_with(true));
  if (!base.has_value() || !pre.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "solve_lp", name);
    return;
  }

  // Never a verdict presolve invented: whatever the plain solve concludes,
  // the presolved one must not claim Infeasible or Unbounded against it.
  if (pre->status == SolverStatus::Infeasible || pre->status == SolverStatus::Unbounded) {
    if (base->status != pre->status) {
      ::sovsolve::test::record(__FILE__, __LINE__, "presolve verdict disagrees", name);
    }
  }
  if (base->status != SolverStatus::Optimal) {
    std::printf("  %-10s base status not optimal, skipped\n", name.c_str());
    return;
  }

  const Kkt kb = kkt(problem, *base);
  const Kkt kp = kkt(problem, *pre);
  const Real obj_diff =
      std::fabs(pre->objective - base->objective) / (1.0 + std::fabs(base->objective));

  std::printf("  %-10s rows %6zu -> %6zu  cols %6zu -> %6zu  nnz %7zu -> %7zu  "
              "dobj %.1e  primal %.1e/%.1e  dual %.1e/%.1e  sign %.1e  compl %.1e/%.1e\n",
              name.c_str(), problem.num_rows(), pre->presolved_rows, problem.num_cols(),
              pre->presolved_cols, problem.A.nnz(), pre->presolved_nnz, obj_diff, kp.primal,
              kb.primal, kp.dual, kb.dual, kp.sign, kp.compl_, kb.compl_);

  CHECK(pre->status == SolverStatus::Optimal);
  if (!(obj_diff <= 1e-7)) {
    ::sovsolve::test::record(__FILE__, __LINE__, "objective moved", name);
  }
  if (!within(kp.primal, kb.primal, 1e-7)) {
    ::sovsolve::test::record(__FILE__, __LINE__, "primal worse", name);
  }
  if (!within(kp.dual, kb.dual, 1e-7)) {
    ::sovsolve::test::record(__FILE__, __LINE__, "dual worse", name);
  }
  if (!(kp.sign <= std::max(1e-7, 10.0 * kb.sign))) {
    ::sovsolve::test::record(__FILE__, __LINE__, "dual sign", name);
  }
  if (!within(kp.compl_, kb.compl_, 1e-7)) {
    ::sovsolve::test::record(__FILE__, __LINE__, "complementarity worse", name);
  }
}

void run_directory(const fs::path& dir) {
  if (!fs::is_directory(dir)) return;
  std::vector<fs::path> files;
  for (const auto& entry : fs::directory_iterator(dir)) {
    const auto ext = entry.path().extension().string();
    if (ext == ".mps" || ext == ".gz") files.push_back(entry.path());
  }
  std::sort(files.begin(), files.end());
  std::printf("%s (%zu files)\n", dir.string().c_str(), files.size());
  for (const auto& f : files) run_model(f);
}

}  // namespace

int main() {
#ifdef SOVSOLVE_TEST_DATA_DIR
  run_directory(fs::path(SOVSOLVE_TEST_DATA_DIR) / "netlib");
#endif
  if (const char* extra = std::getenv("SOVSOLVE_LP_PRESOLVE_CORPUS")) run_directory(extra);
  return ::sovsolve::test::report("lp_presolve_test");
}
