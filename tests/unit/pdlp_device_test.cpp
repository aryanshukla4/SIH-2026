// Module 24F: the device-resident PDLP backend, checked against the host one.
//
// THE ORACLE IS THE HOST BACKEND, operation by operation. Both implement
// `pdlp::IterationBackend`; the host one is the pre-refactor inner loop moved
// verbatim (and proven bit-identical to it across the whole corpus on two
// compilers). So the test puts the SAME state into both and compares every
// output -- trial metrics, the accepted iterate, the fixed step, the
// infeasibility sequences, the restart average.
//
// Tolerances are small but not zero, and deliberately so. The device sums in a
// different ORDER (block partials, then one block) and nvcc contracts
// `x - tau*(c - k)` into a fused multiply-add. Elementwise results therefore
// differ by about an ulp and sums by a few; what must NOT happen is a
// difference of structure -- a projection applied to the wrong rows, a sign,
// a term left out -- and those show up at 1e-3, not 1e-14.
//
// The second thing checked is REPRODUCIBILITY: the device reductions use a
// fixed grid and no atomics, so two runs from the same state must agree to the
// last bit. Atomics would pass every other test here and fail this one.

#include <cmath>
#include <cstddef>
#include <string>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/solver/Presolver.hpp"
#include "sovsolve/solver/Scaler.hpp"
#include "sovsolve/solver/gpu/PdlpDevice.hpp"
#include "sovsolve/solver/pdlp/IterationBackend.hpp"
#include "sovsolve/solver/pdlp/MatVec.hpp"
#include "sovsolve/solver/pdlp/Pdlp.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
using core::Real;
using solver::pdlp::BackendVector;
using solver::pdlp::HostIterationBackend;
using solver::pdlp::HostMatVec;
using solver::pdlp::TrialMetrics;
using solver::gpu::DevicePdlpBackend;

namespace {

bool load_canonical(const std::string& name, model::CanonicalResult& out) {
  const std::string path = std::string(SOVSOLVE_TEST_DATA_DIR) + "/netlib/" + name;
  auto parsed = io::loadProblem(path);
  if (!parsed.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "load", parsed.error().format());
    return false;
  }
  model::Options options;
  options.log.level = model::LogOptions::Level::Silent;
  auto canon = model::canonicalize(parsed.value(), options);
  if (!canon.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "canonicalize", canon.error().format());
    return false;
  }
  out = std::move(canon.value());
  return true;
}

/// A synthetic LP large enough to span several 256-thread blocks, so the
/// second, cross-block reduction stage is exercised. Generated here rather than
/// loaded, because the larger Netlib files are gitignored and a test that needs
/// an untracked file passes on this machine and fails on a fresh clone.
///
/// 2000 columns and 600 rows: 8 blocks over columns. Every structural case is
/// present -- equality and `<=` rows, columns bounded on both sides, one side,
/// and free.
bool build_synthetic(model::CanonicalResult& out) {
  constexpr int kCols = 2000;
  constexpr int kRows = 600;
  constexpr int kEqualities = 150;
  unsigned seed = 12345u;
  auto next = [&seed]() {
    seed = seed * 1664525u + 1013904223u;
    return seed >> 8;
  };

  std::string text = "Minimize\n obj:";
  for (int j = 0; j < kCols; ++j) {
    const double coef = static_cast<double>(static_cast<int>(next() % 200u) - 100) / 37.0;
    text += (coef < 0 ? " - " : " + ") + std::to_string(std::fabs(coef)) + " x" +
            std::to_string(j);
  }
  text += "\nSubject To\n";
  for (int i = 0; i < kRows; ++i) {
    text += (i < kEqualities ? " e" : " c") + std::to_string(i) + ":";
    for (int k = 0; k < 5; ++k) {
      const unsigned col = next() % static_cast<unsigned>(kCols);
      const double a = 0.5 + static_cast<double>(next() % 40u) / 10.0;
      text += " + " + std::to_string(a) + " x" + std::to_string(col);
    }
    text += (i < kEqualities ? " = " : " <= ") + std::to_string(3 + static_cast<int>(next() % 20u)) +
            "\n";
  }
  text += "Bounds\n";
  for (int j = 0; j < kCols; ++j) {
    switch (j % 4) {
      case 0: text += " -2 <= x" + std::to_string(j) + " <= 5\n"; break;
      case 1: text += " x" + std::to_string(j) + " >= -1\n"; break;
      case 2: text += " x" + std::to_string(j) + " <= 4\n"; break;
      default: text += " x" + std::to_string(j) + " free\n"; break;
    }
  }
  text += "End\n";

  auto parsed = io::parseProblem(text, io::FileFormat::Lp);
  if (!parsed.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "parse", parsed.error().format());
    return false;
  }
  model::Options options;
  options.log.level = model::LogOptions::Level::Silent;
  auto canon = model::canonicalize(parsed.value(), options);
  if (!canon.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "canonicalize", canon.error().format());
    return false;
  }
  out = std::move(canon.value());
  return true;
}

bool load_any(const std::string& name, model::CanonicalResult& out) {
  return name == "synthetic" ? build_synthetic(out) : load_canonical(name, out);
}

Real rel(Real a, Real b) { return std::fabs(a - b) / std::fmax(1.0, std::fabs(b)); }

/// A state with every structural case live: primal values at, below and above
/// their bounds (so the clamp matters), and duals of both signs on both row
/// types (so the inequality-row projection matters).
void seed_state(const model::CanonicalProblem& p, core::RealVector& x,
                core::RealVector& y) {
  x = core::RealVector(p.num_cols());
  y = core::RealVector(p.num_rows());
  for (std::size_t j = 0; j < x.size(); ++j) {
    const Real raw = 0.7 * static_cast<Real>(j % 7) - 1.3;
    x[j] = std::fmin(std::fmax(raw, p.col_lower[j]), p.col_upper[j]);
  }
  for (std::size_t i = 0; i < y.size(); ++i) {
    y[i] = (i % 2 == 0 ? 0.4 : -0.9) * static_cast<Real>(1 + i % 3);
  }
}

core::HostSpan<const Real> in(const core::RealVector& v) { return {v.data(), v.size()}; }
core::HostSpan<Real> out(core::RealVector& v) { return {v.data(), v.size()}; }

Real max_rel_diff(solver::pdlp::IterationBackend& a, solver::pdlp::IterationBackend& b,
                  BackendVector which, std::size_t len) {
  core::RealVector va(len);
  core::RealVector vb(len);
  a.download(which, out(va));
  b.download(which, out(vb));
  Real worst = 0.0;
  for (std::size_t k = 0; k < len; ++k) worst = std::fmax(worst, rel(va[k], vb[k]));
  return worst;
}

// -------------------------------------------------------------------------

/// Every hot-path operation, host against device, from one shared state.
void test_every_operation_matches_the_host() {
  // afiro and adlittle each fit in ONE 256-thread block, so on their own they
  // never reach the cross-block reduction stage. The synthetic model spans 8.
  for (const char* name : {"afiro.mps", "adlittle.mps", "synthetic"}) {
    model::CanonicalResult canon;
    if (!load_any(name, canon)) return;
    CHECK(name != std::string("synthetic") || canon.problem.num_cols() > 4 * 256);
    const model::CanonicalProblem& p = canon.problem;
    const std::size_t n = p.num_cols();
    const std::size_t m = p.num_rows();

    HostMatVec host_matvec(p);
    HostIterationBackend host(p, host_matvec);
    auto created = DevicePdlpBackend::create(p);
    CHECK(created.has_value());
    if (!created.has_value()) return;
    DevicePdlpBackend& device = **created;

    core::RealVector x;
    core::RealVector y;
    seed_state(p, x, y);
    host.set_iterate(in(x), in(y));
    device.set_iterate(in(x), in(y));
    CHECK(max_rel_diff(host, device, BackendVector::X, n) == 0.0);  // exact upload

    // --- Algorithm 2: begin, trial, accept ---------------------------------
    host.begin_step();
    device.begin_step();
    for (Real tau : {0.05, 0.4, 2.0}) {
      const TrialMetrics h = host.trial(tau, 1.0 / tau);
      const TrialMetrics d = device.trial(tau, 1.0 / tau);
      CHECK(rel(d.interaction, h.interaction) < 1e-10);
      CHECK(rel(d.dx_sq, h.dx_sq) < 1e-10);
      CHECK(rel(d.dy_sq, h.dy_sq) < 1e-10);
      // Nonzero, or the comparisons above prove nothing.
      CHECK(h.dx_sq > 0.0);
      CHECK(h.dy_sq > 0.0);
    }
    host.accept_trial();
    device.accept_trial();
    CHECK(max_rel_diff(host, device, BackendVector::X, n) < 1e-12);
    CHECK(max_rel_diff(host, device, BackendVector::Y, m) < 1e-12);

    // --- infeasibility sequences, around a fixed step ----------------------
    host.snapshot_iterate();
    device.snapshot_iterate();
    host.fixed_step(0.3, 0.5);
    device.fixed_step(0.3, 0.5);
    host.finish_difference();
    device.finish_difference();
    CHECK(max_rel_diff(host, device, BackendVector::X, n) < 1e-12);
    CHECK(max_rel_diff(host, device, BackendVector::Y, m) < 1e-12);
    CHECK(max_rel_diff(host, device, BackendVector::DifferenceX, n) < 1e-10);
    CHECK(max_rel_diff(host, device, BackendVector::DifferenceY, m) < 1e-10);
    CHECK(max_rel_diff(host, device, BackendVector::IterateSumX, n) < 1e-12);
    CHECK(max_rel_diff(host, device, BackendVector::IterateSumY, m) < 1e-12);

    // --- restart average -----------------------------------------------------
    host.accumulate_average(0.25);
    device.accumulate_average(0.25);
    host.accumulate_average(1.5);
    device.accumulate_average(1.5);
    CHECK(max_rel_diff(host, device, BackendVector::AverageX, n) < 1e-12);
    CHECK(max_rel_diff(host, device, BackendVector::AverageY, m) < 1e-12);
    host.reset_average();
    device.reset_average();
    core::RealVector zeroed(n);
    device.download(BackendVector::AverageX, out(zeroed));
    for (std::size_t j = 0; j < n; ++j) CHECK_NEAR(zeroed[j], 0.0, 0.0);

    // Products: begin (2) + three trials (3) + fixed step (2), all counted by
    // the device itself since it does not route through a host MatVec.
    CHECK_EQ(device.own_products(), std::size_t{7});
    CHECK_EQ(host_matvec.products(), std::size_t{7});
    CHECK(device.status().ok());
  }
}

/// No atomics: the same state must give the same bits, every time.
void test_reductions_are_reproducible() {
  model::CanonicalResult canon;
  // Multi-block, or there is no cross-block order to be nondeterministic about.
  if (!build_synthetic(canon)) return;
  const model::CanonicalProblem& p = canon.problem;
  auto a = DevicePdlpBackend::create(p);
  auto b = DevicePdlpBackend::create(p);
  CHECK(a.has_value() && b.has_value());
  if (!a.has_value() || !b.has_value()) return;

  core::RealVector x;
  core::RealVector y;
  seed_state(p, x, y);
  (*a)->set_iterate(in(x), in(y));
  (*b)->set_iterate(in(x), in(y));
  (*a)->begin_step();
  (*b)->begin_step();
  for (int k = 0; k < 5; ++k) {
    const TrialMetrics ma = (*a)->trial(0.3, 2.0);
    const TrialMetrics mb = (*b)->trial(0.3, 2.0);
    CHECK(ma.interaction == mb.interaction);
    CHECK(ma.dx_sq == mb.dx_sq);
    CHECK(ma.dy_sq == mb.dy_sq);
  }
}

/// The whole solver, end to end, on the device -- same verdict and objective
/// as the host.
void test_end_to_end_matches_the_host() {
  model::CanonicalResult canon;
  if (!load_canonical("afiro.mps", canon)) return;
  model::Options options;
  options.log.level = model::LogOptions::Level::Silent;
  options.scaling.mode = model::ScalingMode::RuizPockChambolle;
  CHECK(solver::presolve(canon.problem, options, canon.transforms).ok());
  CHECK(solver::scale(canon.problem, options, canon.transforms).ok());
  const model::CanonicalProblem& p = canon.problem;

  auto host = solver::pdlp::solve_pdlp(p, options);
  CHECK(host.has_value());

  auto created = DevicePdlpBackend::create(p);
  CHECK(created.has_value());
  if (!host.has_value() || !created.has_value()) return;
  auto device = solver::pdlp::solve_pdlp(p, options, (*created)->cold_matvec(), **created);
  CHECK(device.has_value());
  if (!device.has_value()) return;

  CHECK(host->status == core::SolverStatus::Optimal);
  CHECK(device->status == core::SolverStatus::Optimal);
  CHECK(rel(device->objective, host->objective) < 1e-7);
  // Same algorithm, rounding-level differences only: the step counts may
  // drift, but not by much on an instance this small.
  const Real drift = std::fabs(static_cast<Real>(device->matrix_products) -
                               static_cast<Real>(host->matrix_products)) /
                     static_cast<Real>(host->matrix_products);
  CHECK(drift < 0.05);
  CHECK((*created)->status().ok());
}

}  // namespace

int main() {
  test_every_operation_matches_the_host();
  test_reductions_are_reproducible();
  test_end_to_end_matches_the_host();
  return ::sovsolve::test::report("pdlp_device_test");
}
