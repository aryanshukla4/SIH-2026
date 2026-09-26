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
#include <cstdint>
#include <optional>
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
using solver::pdlp::AnchorDistance;
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

/// Module 31: the three Halpern operations, host against device, from one
/// shared state -- across several steps and a restart, because the design
/// point is a RECURRENCE (`K x` and `K' y` are blended, never recomputed),
/// and a recurrence can agree for one step and drift for the next.
///
/// Every quantity the recurrence carries is checked indirectly but
/// completely: a wrong cached image corrupts the next step's `T(z)`, which is
/// compared directly, and the metrics depend on `K x` directly. `gamma` is
/// 0.7 rather than 1 so neither blend coefficient is special-cased away.
void test_halpern_matches_the_host() {
  for (const char* name : {"afiro.mps", "synthetic"}) {
    model::CanonicalResult canon;
    if (!load_any(name, canon)) return;
    const model::CanonicalProblem& p = canon.problem;
    const std::size_t n = p.num_cols();
    const std::size_t m = p.num_rows();

    HostMatVec host_matvec(p);
    HostIterationBackend host(p, host_matvec);
    auto created = DevicePdlpBackend::create(p);
    CHECK(created.has_value());
    if (!created.has_value()) return;
    DevicePdlpBackend& device = **created;
    CHECK(device.supports_halpern());

    core::RealVector x;
    core::RealVector y;
    seed_state(p, x, y);
    host.set_iterate(in(x), in(y));
    device.set_iterate(in(x), in(y));
    host.begin_halpern();
    device.begin_halpern();

    const Real eta = 0.05;
    const Real omega = 1.3;
    const Real gamma = 0.7;
    for (int k = 0; k < 6; ++k) {
      const Real lambda = (k + 1.0) / (k + 2.0);
      const TrialMetrics h = host.halpern_step(eta, omega, gamma, lambda);
      const TrialMetrics d = device.halpern_step(eta, omega, gamma, lambda);
      CHECK(rel(d.interaction, h.interaction) < 1e-9);
      CHECK(rel(d.dx_sq, h.dx_sq) < 1e-9);
      CHECK(rel(d.dy_sq, h.dy_sq) < 1e-9);
      CHECK(h.dx_sq > 0.0);  // or the comparison above proves nothing
      CHECK(max_rel_diff(host, device, BackendVector::X, n) < 1e-10);
      CHECK(max_rel_diff(host, device, BackendVector::Y, m) < 1e-10);
      CHECK(max_rel_diff(host, device, BackendVector::PdhgX, n) < 1e-10);
      CHECK(max_rel_diff(host, device, BackendVector::PdhgY, m) < 1e-10);
    }

    // The restart: zero products, device-side copies, and the two distances
    // the PID reads.
    const std::size_t before = device.own_products();
    const AnchorDistance hd = host.restart_at_pdhg_point();
    const AnchorDistance dd = device.restart_at_pdhg_point();
    CHECK_EQ(device.own_products(), before);
    CHECK(rel(dd.dx, hd.dx) < 1e-9);
    CHECK(rel(dd.dy, hd.dy) < 1e-9);
    CHECK(hd.dx > 0.0);
    CHECK(max_rel_diff(host, device, BackendVector::X, n) < 1e-10);
    CHECK(max_rel_diff(host, device, BackendVector::Y, m) < 1e-10);

    // And the step AFTER the restart, which reads the images the restart
    // installed -- the path where a missed copy would first show.
    const TrialMetrics h = host.halpern_step(eta, omega, gamma, 0.5);
    const TrialMetrics d = device.halpern_step(eta, omega, gamma, 0.5);
    CHECK(rel(d.dx_sq, h.dx_sq) < 1e-9);
    CHECK(rel(d.interaction, h.interaction) < 1e-9);
    CHECK(max_rel_diff(host, device, BackendVector::PdhgX, n) < 1e-10);

    // begin (2) + seven steps at two each. Nothing else.
    CHECK_EQ(device.own_products(), std::size_t{2 + 7 * 2});
    CHECK_EQ(host_matvec.products(), std::size_t{2 + 7 * 2});
    CHECK(device.status().ok());
  }
}

/// The device's `||A||_2` against the host routine, restated here (it is
/// file-local in Pdlp.cpp). The step size is `0.998 / ||A||_2`, so a wrong
/// norm is a wrong step: too large and the method diverges, too small and it
/// crawls, and neither fails loudly.
void test_spectral_norm_matches_the_host() {
  for (const char* name : {"afiro.mps", "synthetic"}) {
    model::CanonicalResult canon;
    if (!load_any(name, canon)) return;
    const model::CanonicalProblem& p = canon.problem;
    const std::size_t n = p.num_cols();
    const std::size_t m = p.num_rows();

    HostMatVec mv(p);
    core::RealVector v(n);
    core::RealVector kv(m);
    Real norm_sq = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      v[j] = 1.0 + static_cast<Real>(j % 7) * 0.1;
      norm_sq += v[j] * v[j];
    }
    for (std::size_t j = 0; j < n; ++j) v[j] /= std::sqrt(norm_sq);
    Real host_sigma = 0.0;
    for (int it = 0; it < 100; ++it) {
      mv.multiply(in(v), out(kv));
      mv.multiply_transpose(in(kv), out(v));
      Real s = 0.0;
      for (std::size_t j = 0; j < n; ++j) s += v[j] * v[j];
      const Real next = std::sqrt(s);
      for (std::size_t j = 0; j < n; ++j) v[j] /= next;
      const Real candidate = std::sqrt(next);
      if (host_sigma > 0.0 && std::fabs(candidate - host_sigma) <= 1e-6 * host_sigma) {
        host_sigma = candidate;
        break;
      }
      host_sigma = candidate;
    }

    auto created = DevicePdlpBackend::create(p);
    CHECK(created.has_value());
    if (!created.has_value()) return;
    const std::optional<Real> device_sigma = (*created)->spectral_norm(100, 1e-6);
    CHECK(device_sigma.has_value());
    if (!device_sigma.has_value()) return;
    CHECK(host_sigma > 0.0);
    // Same seed, same update, same stopping rule; only the reduction order
    // differs, and on a converged power iteration that moves the last digits.
    CHECK(rel(*device_sigma, host_sigma) < 1e-9);
    CHECK((*created)->status().ok());
  }
}

/// The resident controller loop against the host's, over a chunk long enough
/// to RESTART several times.
///
/// This is the test that matters for the no-sync design, because every piece
/// of it is conditional on a flag the host never sees: the anchor-distance
/// reduction, the PID and the move to `T(z)` all run only when the device-side
/// controller set `restart_pending`. A chunk with no restart would pass with
/// all three broken. So the restart constants are loosened until restarts are
/// frequent, and the test asserts they happened before asserting anything
/// else agrees.
///
/// Agreement is to rounding, not bits: `log`/`exp` in the PID are libm on the
/// host and CUDA's on the device, and those are not required to agree in the
/// last place.
void test_resident_controller_matches_the_host() {
  for (const char* name : {"afiro.mps", "synthetic"}) {
    model::CanonicalResult canon;
    if (!load_any(name, canon)) return;
    const model::CanonicalProblem& p = canon.problem;
    const std::size_t n = p.num_cols();
    const std::size_t m = p.num_rows();

    HostMatVec host_matvec(p);
    HostIterationBackend host(p, host_matvec);
    auto created = DevicePdlpBackend::create(p);
    CHECK(created.has_value());
    if (!created.has_value()) return;
    DevicePdlpBackend& device = **created;

    const std::optional<Real> sigma = device.spectral_norm(100, 1e-6);
    CHECK(sigma.has_value() && *sigma > 0.0);
    if (!sigma.has_value() || !(*sigma > 0.0)) return;

    solver::pdlp::HalpernParams params;
    params.eta = 0.998 / *sigma;
    params.gamma = 1.0;
    params.sufficient = 0.6;  // loosened: restart often
    params.necessary = 0.9;
    params.check_interval = 40;

    core::RealVector x;
    core::RealVector y;
    seed_state(p, x, y);
    for (auto* b : {static_cast<solver::pdlp::IterationBackend*>(&host),
                    static_cast<solver::pdlp::IterationBackend*>(&device)}) {
      b->set_iterate(in(x), in(y));
      b->begin_halpern();
      b->write_halpern_state(solver::pdlp::HalpernState{});
    }

    // Products are compared as a DELTA: `spectral_norm` above counts its own.
    const std::size_t before = device.own_products();

    // Two chunks, so the second starts from state the first left ON the
    // device -- the handoff the solver relies on between checks.
    for (std::uint64_t first : {std::uint64_t{0}, std::uint64_t{80}}) {
      host.run_halpern(80, first, params, true);
      device.run_halpern(80, first, params, true);
    }
    const solver::pdlp::HalpernState hs = host.read_halpern_state();
    const solver::pdlp::HalpernState ds = device.read_halpern_state();

    CHECK(hs.restarts >= 3);  // or the conditional kernels went untested
    CHECK_EQ(ds.restarts, hs.restarts);
    CHECK_EQ(ds.inner, hs.inner);
    CHECK(rel(ds.omega, hs.omega) < 1e-8);
    CHECK(hs.omega != 1.0);  // the PID moved it, so its path is covered
    CHECK(max_rel_diff(host, device, BackendVector::PdhgX, n) < 1e-8);
    CHECK(max_rel_diff(host, device, BackendVector::PdhgY, m) < 1e-8);
    CHECK(max_rel_diff(host, device, BackendVector::DifferenceX, n) < 1e-6);
    CHECK(max_rel_diff(host, device, BackendVector::IterateSumX, n) < 1e-8);
    CHECK_EQ(device.own_products() - before, std::size_t{160 * 2});  // two per step
    CHECK(device.status().ok());
  }
}

/// CUDA graph replay against direct launches of the SAME sequence.
///
/// This oracle is stronger than the host comparison: a graph replays exactly
/// the kernels `enqueue_halpern_iteration` issues, in the same order, with the
/// same arguments -- so the two must agree to the BIT, not to a tolerance.
/// Anything short of that means the graph baked in something that should have
/// varied (a stale buffer address, an iteration number) or dropped a node.
///
/// Chunk lengths are mixed on purpose: 40 exercises the whole-interval graph,
/// 7 the replayed single-iteration graph, and restarts are frequent so the
/// flag-guarded restart kernels run inside the captured graph. The test also
/// asserts graphs actually ENGAGED, or a silent fallback to direct launches
/// would pass it trivially.
void test_graphs_match_direct_launches() {
  for (const char* name : {"afiro.mps", "synthetic"}) {
    model::CanonicalResult canon;
    if (!load_any(name, canon)) return;
    const model::CanonicalProblem& p = canon.problem;
    const std::size_t n = p.num_cols();
    const std::size_t m = p.num_rows();

    auto a = DevicePdlpBackend::create(p);
    auto b = DevicePdlpBackend::create(p);
    CHECK(a.has_value() && b.has_value());
    if (!a.has_value() || !b.has_value()) return;
    DevicePdlpBackend& graphed = **a;
    DevicePdlpBackend& direct = **b;
    graphed.set_use_graphs(true);
    direct.set_use_graphs(false);

    const std::optional<Real> sigma = graphed.spectral_norm(100, 1e-6);
    CHECK(sigma.has_value() && *sigma > 0.0);
    if (!sigma.has_value() || !(*sigma > 0.0)) return;
    solver::pdlp::HalpernParams params;
    params.eta = 0.998 / *sigma;
    params.sufficient = 0.6;  // restart often, inside the graph
    params.necessary = 0.9;
    params.check_interval = 40;

    core::RealVector x;
    core::RealVector y;
    seed_state(p, x, y);
    for (DevicePdlpBackend* d : {&graphed, &direct}) {
      d->set_iterate(in(x), in(y));
      d->begin_halpern();
      d->write_halpern_state(solver::pdlp::HalpernState{});
    }

    std::uint64_t done = 0;
    for (std::size_t chunk : {std::size_t{40}, std::size_t{40}, std::size_t{7},
                              std::size_t{40}, std::size_t{7}}) {
      graphed.run_halpern(chunk, done, params, true);
      direct.run_halpern(chunk, done, params, true);
      done += chunk;
    }
    CHECK(graphed.graphs_active());
    CHECK(graphed.graph_note().empty());
    CHECK(!direct.graphs_active());
    // 3 whole-interval launches + 7 + 7 single replays, against 134 direct.
    CHECK_EQ(graphed.graph_launches(), std::size_t{3 + 7 + 7});
    CHECK_EQ(direct.graph_launches(), std::size_t{134});

    const solver::pdlp::HalpernState gs = graphed.read_halpern_state();
    const solver::pdlp::HalpernState ds = direct.read_halpern_state();
    CHECK(ds.restarts >= 3);
    CHECK_EQ(gs.restarts, ds.restarts);
    CHECK_EQ(gs.inner, ds.inner);
    CHECK_EQ(gs.total, std::uint64_t{134});
    CHECK_EQ(ds.total, std::uint64_t{134});
    CHECK(gs.omega == ds.omega);  // bitwise
    for (BackendVector v : {BackendVector::X, BackendVector::PdhgX,
                            BackendVector::DifferenceX, BackendVector::IterateSumX}) {
      CHECK(max_rel_diff(graphed, direct, v, n) == 0.0);
    }
    for (BackendVector v : {BackendVector::Y, BackendVector::PdhgY}) {
      CHECK(max_rel_diff(graphed, direct, v, m) == 0.0);
    }
    CHECK(graphed.status().ok());
    CHECK(direct.status().ok());
  }
}

/// `--method=pdlpx --gpu-resident=1`, end to end, against the host run.
void test_halpern_end_to_end_matches_the_host() {
  model::CanonicalResult canon;
  if (!load_canonical("afiro.mps", canon)) return;
  model::Options options;
  options.log.level = model::LogOptions::Level::Silent;
  options.scaling.mode = model::ScalingMode::RuizPockChambolle;
  options.pdlp.halpern = true;
  CHECK(solver::presolve(canon.problem, options, canon.transforms).ok());
  CHECK(solver::scale(canon.problem, options, canon.transforms).ok());
  const model::CanonicalProblem& p = canon.problem;

  auto host = solver::pdlp::solve_pdlp(p, options);
  auto created = DevicePdlpBackend::create(p);
  CHECK(host.has_value() && created.has_value());
  if (!host.has_value() || !created.has_value()) return;
  auto device = solver::pdlp::solve_pdlp(p, options, (*created)->cold_matvec(), **created);
  CHECK(device.has_value());
  if (!device.has_value()) return;

  CHECK(host->status == core::SolverStatus::Optimal);
  CHECK(device->status == core::SolverStatus::Optimal);
  CHECK(rel(device->objective, host->objective) < 1e-7);
  // No adaptive step here, so no accept/reject to flip on a rounding
  // difference: the two runs should take nearly the same path.
  const Real drift = std::fabs(static_cast<Real>(device->iterations) -
                               static_cast<Real>(host->iterations)) /
                     static_cast<Real>(host->iterations);
  CHECK(drift < 0.05);
  CHECK((*created)->status().ok());
}

}  // namespace

int main() {
  test_every_operation_matches_the_host();
  test_reductions_are_reproducible();
  test_end_to_end_matches_the_host();
  test_halpern_matches_the_host();
  test_spectral_norm_matches_the_host();
  test_resident_controller_matches_the_host();
  test_graphs_match_direct_launches();
  test_halpern_end_to_end_matches_the_host();
  return ::sovsolve::test::report("pdlp_device_test");
}
