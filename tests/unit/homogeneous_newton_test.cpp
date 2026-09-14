// Module 25 stage 3: the bordered Newton solve.
//
// TWO ORACLES, catching different things.
//
// 1. OUR OWN SPECIFICATION. FORMULATION.md section 13.1's embedding is
//    transcribed here as a dense Jacobian and factorized whole. The module's
//    reduced bordered solve must reproduce that direction exactly. This catches
//    an error in the ELIMINATION -- a dropped term when `dz`, `dv`, `ds` or
//    `dkappa` are substituted out.
//
// 2. THE PUBLISHED SYSTEM. Andersen & Andersen (2000) give the bordered Newton
//    system for STANDARD FORM only. The shift
//
//        x = l tau + s1,   s2 = (u - l) tau - s1,   s1, s2 >= 0
//
//    carries our bounded embedding onto theirs exactly. So the second oracle
//    builds the shifted instance, writes down [AA]'s system VERBATIM with no
//    bound terms anywhere in it, maps our direction across, and requires it to
//    satisfy that system. This catches an error in the BORDER ITSELF -- a wrong
//    `h_x`, `g_x` or `w` -- which oracle 1 cannot, because oracle 1 and the
//    module are both derived from section 13.1 and could share a mistake.
//
// Between them the bound terms are pinned from both directions: by our
// specification, and by the published standard-form algorithm the specification
// claims to generalize. Neither oracle alone is enough.
//
// The shift needs finite bounds, since a column with an infinite bound cannot
// be shifted. That is not a hole: such a column contributes no complementarity
// pair and therefore no bound term, so its border entry is already [AA]'s
// `-c_j`. Oracle 2 covers exactly what [AA] does not, and a separate test
// checks the infinite-bound case collapses onto [AA] directly.

#include <cmath>
#include <cstddef>
#include <vector>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/solver/HomogeneousNewton.hpp"
#include "sovsolve/solver/SolverState.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
using core::Real;
using solver::HomogeneousBorder;
using solver::HomogeneousNewtonRhs;
using solver::HomogeneousNewtonWorkspace;
using solver::SolverState;

namespace {

// -------------------------------------------------------------------------
// Small dense linear algebra. Test scaffolding only -- the production path
// injects a real backend through `solver::KktSolver`.
// -------------------------------------------------------------------------

struct Dense {
  std::size_t n = 0;
  std::vector<Real> a;
  explicit Dense(std::size_t size) : n(size), a(size * size, 0.0) {}
  Real& operator()(std::size_t i, std::size_t j) { return a[i * n + j]; }
  Real operator()(std::size_t i, std::size_t j) const { return a[i * n + j]; }
};

/// Gaussian elimination with partial pivoting. Returns false on a singular
/// matrix rather than producing infinities.
bool dense_solve(Dense m, std::vector<Real> rhs, std::vector<Real>& out) {
  const std::size_t n = m.n;
  std::vector<std::size_t> perm(n);
  for (std::size_t i = 0; i < n; ++i) perm[i] = i;

  for (std::size_t k = 0; k < n; ++k) {
    std::size_t best = k;
    for (std::size_t i = k + 1; i < n; ++i) {
      if (std::fabs(m(i, k)) > std::fabs(m(best, k))) best = i;
    }
    if (std::fabs(m(best, k)) < 1e-13) return false;
    if (best != k) {
      for (std::size_t j = 0; j < n; ++j) std::swap(m(k, j), m(best, j));
      std::swap(rhs[k], rhs[best]);
    }
    for (std::size_t i = k + 1; i < n; ++i) {
      const Real factor = m(i, k) / m(k, k);
      if (factor == 0.0) continue;
      for (std::size_t j = k; j < n; ++j) m(i, j) -= factor * m(k, j);
      rhs[i] -= factor * rhs[k];
    }
  }
  out.assign(n, 0.0);
  for (std::size_t ii = n; ii-- > 0;) {
    Real acc = rhs[ii];
    for (std::size_t j = ii + 1; j < n; ++j) acc -= m(ii, j) * out[j];
    out[ii] = acc / m(ii, ii);
  }
  return true;
}

std::vector<std::vector<Real>> densify(const model::CanonicalProblem& p) {
  const std::size_t m = p.num_rows();
  const std::size_t n = p.num_cols();
  std::vector<std::vector<Real>> a(m, std::vector<Real>(n, 0.0));
  const auto& csr = p.A.csr;
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      a[i][static_cast<std::size_t>(csr.indices()[k])] += csr.values()[k];
    }
  }
  return a;
}

/// `K = [ -Theta^-1  A' ; A  D_s ]`, factorized densely. Section 10.2's
/// augmented KKT with `Q = 0`.
class DenseKktSolver final : public solver::KktSolver {
 public:
  DenseKktSolver(const model::CanonicalProblem& p, const HomogeneousBorder& border)
      : n_(p.num_cols()), m_(p.num_rows()), k_(n_ + m_) {
    const auto a = densify(p);
    const std::size_t m_e = p.num_equality;
    Dense matrix(k_);
    for (std::size_t j = 0; j < n_; ++j) matrix(j, j) = -border.theta_inv[j];
    for (std::size_t i = 0; i < m_; ++i) {
      for (std::size_t j = 0; j < n_; ++j) {
        matrix(j, n_ + i) = a[i][j];
        matrix(n_ + i, j) = a[i][j];
      }
    }
    for (std::size_t k = 0; k < p.num_inequality_rows(); ++k) {
      matrix(n_ + m_e + k, n_ + m_e + k) = border.d_slack[k];
    }
    matrix_ = matrix;
  }

  core::Status solve(core::HostSpan<const Real> rhs_x, core::HostSpan<const Real> rhs_y,
                     core::HostSpan<Real> dx, core::HostSpan<Real> dy) override {
    std::vector<Real> rhs(k_);
    for (std::size_t j = 0; j < n_; ++j) rhs[j] = rhs_x[j];
    for (std::size_t i = 0; i < m_; ++i) rhs[n_ + i] = rhs_y[i];
    std::vector<Real> sol;
    if (!dense_solve(matrix_, rhs, sol)) {
      return core::make_error(core::ErrorCode::NumericalError, "singular K");
    }
    for (std::size_t j = 0; j < n_; ++j) dx[j] = sol[j];
    for (std::size_t i = 0; i < m_; ++i) dy[i] = sol[n_ + i];
    ++solves_;
    return core::Status::Ok();
  }

 private:
  std::size_t n_, m_, k_;
  Dense matrix_{0};
};

// -------------------------------------------------------------------------
// Fixtures
// -------------------------------------------------------------------------

bool parse_into(const char* text, model::CanonicalResult& out) {
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

/// Every bound finite, equality AND inequality rows. Shiftable, so oracle 2
/// applies.
bool build_boxed(model::CanonicalResult& out) {
  return parse_into(R"(Minimize
 obj: 2 x + 3 y - z + 4 w
Subject To
 e1: x + y + z = 6
 c1: x - y + w <= 3
 c2: y + w - z <= 8
Bounds
 0 <= x <= 5
 1 <= y <= 9
 -2 <= z <= 4
 2 <= w <= 7
End
)",
                    out);
}

/// Mixed: bounded below only, above only, both, and free. Oracle 1 applies;
/// oracle 2 does not, which is the point of having both fixtures.
bool build_mixed(model::CanonicalResult& out) {
  return parse_into(R"(Minimize
 obj: 2 x + 3 y - z + 4 w
Subject To
 e1: x + y + z = 6
 c1: x - y + w <= 3
Bounds
 0 <= x <= 5
 y <= 9
 z free
 w >= 2
End
)",
                    out);
}

SolverState make_state(const model::CanonicalProblem& p, Real tau, Real kappa) {
  SolverState s;
  const std::size_t n = p.num_cols();
  const std::size_t m = p.num_rows();
  const std::size_t m_i = p.num_inequality_rows();
  s.x = core::RealVector(n);
  s.z = core::RealVector(n, 0.0);
  s.v = core::RealVector(n, 0.0);
  s.y = core::RealVector(m, 0.0);
  s.s = core::RealVector(m_i);
  for (std::size_t j = 0; j < n; ++j) {
    const Real l = p.col_lower[j];
    const Real u = p.col_upper[j];
    const bool has_l = core::is_finite_bound(l);
    const bool has_u = core::is_finite_bound(u);
    const Real frac = 0.35 + 0.1 * static_cast<Real>(j % 3);
    if (has_l && has_u) {
      s.x[j] = (1.0 - frac) * l * tau + frac * u * tau;
    } else if (has_l) {
      s.x[j] = (l + 1.0 + 0.3 * static_cast<Real>(j % 4)) * tau;
    } else if (has_u) {
      s.x[j] = (u - 1.0 - 0.3 * static_cast<Real>(j % 4)) * tau;
    } else {
      s.x[j] = 0.4 + 0.2 * static_cast<Real>(j % 5);
    }
    if (has_l) s.z[j] = 0.4 + 0.2 * static_cast<Real>(j % 3);
    if (has_u) s.v[j] = 0.3 + 0.25 * static_cast<Real>(j % 4);
  }
  for (std::size_t i = 0; i < p.num_equality; ++i) {
    s.y[i] = -0.3 - 0.2 * static_cast<Real>(i % 3);
  }
  for (std::size_t k = 0; k < m_i; ++k) {
    s.s[k] = 0.8 + 0.3 * static_cast<Real>(k % 2);
    s.y[p.num_equality + k] = -(0.6 + 0.2 * static_cast<Real>(k % 3));
  }
  s.tau = tau;
  s.kappa = kappa;
  return s;
}

/// A deterministic pseudo-random right-hand side. Deterministic so a failure
/// reproduces; varied so the test is not accidentally satisfied by a structured
/// vector such as all-ones.
HomogeneousNewtonRhs make_rhs(const model::CanonicalProblem& p, unsigned seed) {
  auto next = [&seed]() {
    seed = seed * 1664525u + 1013904223u;
    return static_cast<Real>((seed >> 8) % 2000u) / 1000.0 - 1.0;
  };
  HomogeneousNewtonRhs r;
  const std::size_t n = p.num_cols();
  r.rp = core::RealVector(p.num_rows());
  r.rd = core::RealVector(n);
  r.rxz = core::RealVector(n, 0.0);
  r.ruv = core::RealVector(n, 0.0);
  r.rsy = core::RealVector(p.num_inequality_rows());
  for (std::size_t i = 0; i < p.num_rows(); ++i) r.rp[i] = next();
  for (std::size_t j = 0; j < n; ++j) r.rd[j] = next();
  r.rg = next();
  for (std::size_t j = 0; j < n; ++j) {
    if (core::is_finite_bound(p.col_lower[j])) r.rxz[j] = next();
    if (core::is_finite_bound(p.col_upper[j])) r.ruv[j] = next();
  }
  for (std::size_t k = 0; k < p.num_inequality_rows(); ++k) r.rsy[k] = next();
  r.rtk = next();
  return r;
}

// -------------------------------------------------------------------------
// ORACLE 1 -- FORMULATION.md section 13.1, transcribed and factorized whole.
// -------------------------------------------------------------------------

/// Layout: dx(n) dy(m) dz(n) dv(n) ds(m_I) dtau dkappa.
bool full_newton_solve(const model::CanonicalProblem& p, const SolverState& st,
                       const HomogeneousNewtonRhs& r, std::vector<Real>& out) {
  const std::size_t n = p.num_cols();
  const std::size_t m = p.num_rows();
  const std::size_t m_i = p.num_inequality_rows();
  const std::size_t m_e = p.num_equality;
  const Real tau = st.tau;
  const auto a = densify(p);

  const std::size_t ix = 0, iy = n, iz = n + m, iv = 2 * n + m, is = 3 * n + m;
  const std::size_t it = 3 * n + m + m_i, ik = it + 1;
  const std::size_t N = ik + 1;

  Dense j(N);
  std::vector<Real> rhs(N, 0.0);
  std::size_t row = 0;

  // R1: A dx + E_I ds - b dtau
  for (std::size_t i = 0; i < m; ++i, ++row) {
    for (std::size_t c = 0; c < n; ++c) j(row, ix + c) = a[i][c];
    if (i >= m_e) j(row, is + (i - m_e)) = 1.0;
    j(row, it) = -p.b[i];
    rhs[row] = r.rp[i];
  }
  // R2: A'dy + dz - dv - c dtau
  for (std::size_t c = 0; c < n; ++c, ++row) {
    for (std::size_t i = 0; i < m; ++i) j(row, iy + i) = a[i][c];
    j(row, iz + c) = 1.0;
    j(row, iv + c) = -1.0;
    j(row, it) = -p.c[c];
    rhs[row] = r.rd[c];
  }
  // R3: c'dx - b'dy - l'dz + u'dv + dkappa
  for (std::size_t c = 0; c < n; ++c) {
    j(row, ix + c) = p.c[c];
    if (core::is_finite_bound(p.col_lower[c])) j(row, iz + c) = -p.col_lower[c];
    if (core::is_finite_bound(p.col_upper[c])) j(row, iv + c) = p.col_upper[c];
  }
  for (std::size_t i = 0; i < m; ++i) j(row, iy + i) = -p.b[i];
  j(row, ik) = 1.0;
  rhs[row] = r.rg;
  ++row;
  // R4: Z(dx - l dtau) + (x - l tau) dz      [absent where l is infinite]
  for (std::size_t c = 0; c < n; ++c, ++row) {
    if (!core::is_finite_bound(p.col_lower[c])) {
      j(row, iz + c) = 1.0;  // pins the unused dz to 0
      continue;
    }
    const Real l = p.col_lower[c];
    j(row, ix + c) = st.z[c];
    j(row, it) = -st.z[c] * l;
    j(row, iz + c) = st.x[c] - l * tau;
    rhs[row] = r.rxz[c];
  }
  // R5: V(u dtau - dx) + (u tau - x) dv
  for (std::size_t c = 0; c < n; ++c, ++row) {
    if (!core::is_finite_bound(p.col_upper[c])) {
      j(row, iv + c) = 1.0;
      continue;
    }
    const Real u = p.col_upper[c];
    j(row, ix + c) = -st.v[c];
    j(row, it) = st.v[c] * u;
    j(row, iv + c) = u * tau - st.x[c];
    rhs[row] = r.ruv[c];
  }
  // R7: Sigma_I ds - S dy_I
  for (std::size_t k = 0; k < m_i; ++k, ++row) {
    j(row, is + k) = solver::slack_dual(st.y[m_e + k]);
    j(row, iy + m_e + k) = -st.s[k];
    rhs[row] = r.rsy[k];
  }
  // R6: kappa dtau + tau dkappa
  j(row, it) = st.kappa;
  j(row, ik) = tau;
  rhs[row] = r.rtk;

  return dense_solve(j, rhs, out);
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

void test_reduced_solve_matches_the_full_system() {
  for (int which = 0; which < 2; ++which) {
    model::CanonicalResult canon;
    const bool built = (which == 0) ? build_boxed(canon) : build_mixed(canon);
    if (!built) return;
    const model::CanonicalProblem& p = canon.problem;
    const std::size_t n = p.num_cols();
    const std::size_t m = p.num_rows();
    const std::size_t m_i = p.num_inequality_rows();
    const std::size_t m_e = p.num_equality;

    SolverState st = make_state(p, 1.3, 0.45);
    HomogeneousBorder border;
    CHECK(solver::compute_homogeneous_border(p, st, border).ok());

    DenseKktSolver kkt(p, border);
    HomogeneousNewtonWorkspace work;
    CHECK(solver::refresh_border_solve(p, border, kkt, work).ok());

    const HomogeneousNewtonRhs rhs = make_rhs(p, 7u + static_cast<unsigned>(which));
    SolverState out = make_state(p, 1.3, 0.45);
    CHECK(solver::solve_homogeneous_newton(p, st, border, rhs, kkt, work,
                                           /*affine=*/false, out)
              .ok());

    std::vector<Real> full;
    CHECK(full_newton_solve(p, st, rhs, full));
    if (full.empty()) continue;

    const std::size_t ix = 0, iy = n, iz = n + m, iv = 2 * n + m, is = 3 * n + m;
    const std::size_t it = 3 * n + m + m_i, ik = it + 1;
    for (std::size_t j = 0; j < n; ++j) CHECK_NEAR(out.dx[j], full[ix + j], 1e-9);
    for (std::size_t i = 0; i < m; ++i) CHECK_NEAR(out.dy[i], full[iy + i], 1e-9);
    for (std::size_t j = 0; j < n; ++j) {
      if (core::is_finite_bound(p.col_lower[j])) CHECK_NEAR(out.dz[j], full[iz + j], 1e-9);
      if (core::is_finite_bound(p.col_upper[j])) CHECK_NEAR(out.dv[j], full[iv + j], 1e-9);
    }
    for (std::size_t k = 0; k < m_i; ++k) CHECK_NEAR(out.ds[k], full[is + k], 1e-9);
    CHECK_NEAR(out.dtau, full[it], 1e-9);
    CHECK_NEAR(out.dkappa, full[ik], 1e-9);
    (void)m_e;
  }
}

/// ORACLE 2. Build the shifted standard-form instance, write [AA]'s system down
/// with NO bound terms in it, and require our direction to satisfy it.
void test_direction_satisfies_the_published_standard_form_system() {
  model::CanonicalResult canon;
  if (!build_boxed(canon)) return;
  const model::CanonicalProblem& p = canon.problem;
  const std::size_t n = p.num_cols();
  const std::size_t m = p.num_rows();
  const std::size_t m_i = p.num_inequality_rows();
  const std::size_t m_e = p.num_equality;
  const auto a = densify(p);

  // Every bound must be finite for the shift to exist.
  for (std::size_t j = 0; j < n; ++j) {
    CHECK(core::is_finite_bound(p.col_lower[j]));
    CHECK(core::is_finite_bound(p.col_upper[j]));
  }

  const Real tau = 1.3, kappa = 0.45;
  SolverState st = make_state(p, tau, kappa);
  HomogeneousBorder border;
  CHECK(solver::compute_homogeneous_border(p, st, border).ok());
  DenseKktSolver kkt(p, border);
  HomogeneousNewtonWorkspace work;
  CHECK(solver::refresh_border_solve(p, border, kkt, work).ok());

  const HomogeneousNewtonRhs r = make_rhs(p, 99u);
  SolverState d = make_state(p, tau, kappa);
  CHECK(solver::solve_homogeneous_newton(p, st, border, r, kkt, work, false, d).ok());

  // --- the shifted instance: xi = (s1, s2, sl) >= 0 ------------------------
  const std::size_t nt = 2 * n + m_i;
  const std::size_t mt = m + n;

  // At: rows [A | 0 | E_I] then [I | I | 0]
  std::vector<std::vector<Real>> At(mt, std::vector<Real>(nt, 0.0));
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) At[i][j] = a[i][j];
    if (i >= m_e) At[i][2 * n + (i - m_e)] = 1.0;
  }
  for (std::size_t j = 0; j < n; ++j) {
    At[m + j][j] = 1.0;
    At[m + j][n + j] = 1.0;
  }
  // bt = (b - A l ; u - l),  ct = (c ; 0 ; 0)
  std::vector<Real> bt(mt, 0.0), ct(nt, 0.0);
  for (std::size_t i = 0; i < m; ++i) {
    Real al = 0.0;
    for (std::size_t j = 0; j < n; ++j) al += a[i][j] * p.col_lower[j];
    bt[i] = p.b[i] - al;
  }
  for (std::size_t j = 0; j < n; ++j) bt[m + j] = p.col_upper[j] - p.col_lower[j];
  for (std::size_t j = 0; j < n; ++j) ct[j] = p.c[j];

  // the iterate, mapped
  std::vector<Real> xi(nt, 0.0), sig(nt, 0.0), yt(mt, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    xi[j] = st.x[j] - p.col_lower[j] * tau;
    xi[n + j] = p.col_upper[j] * tau - st.x[j];
    sig[j] = st.z[j];
    sig[n + j] = st.v[j];
    yt[m + j] = -st.v[j];
  }
  for (std::size_t i = 0; i < m; ++i) yt[i] = st.y[i];
  for (std::size_t k = 0; k < m_i; ++k) {
    xi[2 * n + k] = st.s[k];
    sig[2 * n + k] = solver::slack_dual(st.y[m_e + k]);
  }

  // the direction, mapped
  std::vector<Real> dxi(nt, 0.0), dsig(nt, 0.0), dyt(mt, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    dxi[j] = d.dx[j] - p.col_lower[j] * d.dtau;
    dxi[n + j] = p.col_upper[j] * d.dtau - d.dx[j];
    dsig[j] = d.dz[j];
    dsig[n + j] = d.dv[j];
    dyt[m + j] = -d.dv[j];
  }
  for (std::size_t i = 0; i < m; ++i) dyt[i] = d.dy[i];
  for (std::size_t k = 0; k < m_i; ++k) {
    dxi[2 * n + k] = d.ds[k];
    dsig[2 * n + k] = -d.dy[m_e + k];
  }

  // --- [AA]'s system (1.6)/(1.7), verbatim, on the shifted instance --------
  // Row 1: At dxi - bt dtau  ==  (r.rp ; 0)
  for (std::size_t i = 0; i < mt; ++i) {
    Real acc = -bt[i] * d.dtau;
    for (std::size_t j = 0; j < nt; ++j) acc += At[i][j] * dxi[j];
    CHECK_NEAR(acc, i < m ? r.rp[i] : 0.0, 1e-9);
  }
  // Row 2: At' dyt + dsig - ct dtau  ==  (r.rd ; 0 ; 0)
  for (std::size_t j = 0; j < nt; ++j) {
    Real acc = dsig[j] - ct[j] * d.dtau;
    for (std::size_t i = 0; i < mt; ++i) acc += At[i][j] * dyt[i];
    CHECK_NEAR(acc, j < n ? r.rd[j] : 0.0, 1e-9);
  }
  // Row 3: -ct'dxi + bt'dyt - dkappa  ==  -rg - l'rd
  //
  // The transformed gap row is ours plus a multiple of the dual row, not ours
  // exactly -- an identity, checked here rather than assumed.
  {
    Real acc = -d.dkappa;
    for (std::size_t j = 0; j < nt; ++j) acc -= ct[j] * dxi[j];
    for (std::size_t i = 0; i < mt; ++i) acc += bt[i] * dyt[i];
    Real l_rd = 0.0;
    for (std::size_t j = 0; j < n; ++j) l_rd += p.col_lower[j] * r.rd[j];
    CHECK_NEAR(acc, -r.rg - l_rd, 1e-9);
  }
  // Row 4: Sigma dxi + Xi dsig  ==  (rxz ; ruv ; rsy)
  for (std::size_t j = 0; j < nt; ++j) {
    const Real acc = sig[j] * dxi[j] + xi[j] * dsig[j];
    Real want = 0.0;
    if (j < n) {
      want = r.rxz[j];
    } else if (j < 2 * n) {
      want = r.ruv[j - n];
    } else {
      want = r.rsy[j - 2 * n];
    }
    CHECK_NEAR(acc, want, 1e-9);
  }
  // Row 5: kappa dtau + tau dkappa  ==  rtk
  CHECK_NEAR(kappa * d.dtau + tau * d.dkappa, r.rtk, 1e-9);
}

/// At `l = 0, u = inf` the border must be [AA] (1.26)'s exactly.
void test_border_collapses_to_the_published_one() {
  model::CanonicalResult canon;
  if (!parse_into(R"(Minimize
 obj: 2 x + 3 y + z
Subject To
 e1: x + y = 4
 e2: y + z = 3
End
)",
                  canon)) {
    return;
  }
  const model::CanonicalProblem& p = canon.problem;
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    CHECK_NEAR(p.col_lower[j], 0.0, 0.0);
    CHECK(!core::is_finite_bound(p.col_upper[j]));
  }

  SolverState st = make_state(p, 1.3, 0.45);
  HomogeneousBorder border;
  CHECK(solver::compute_homogeneous_border(p, st, border).ok());

  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    CHECK_NEAR(border.h_x[j], -p.c[j], 1e-14);  // [AA]'s border column
    CHECK_NEAR(border.g_x[j], p.c[j], 1e-14);   // [AA]'s border row
    // and Theta^-1 is [AA]'s X^-1 S, since only the lower pair exists
    CHECK_NEAR(border.theta_inv[j], st.z[j] / st.x[j], 1e-14);
  }
  CHECK_NEAR(border.w, 0.0, 1e-14);
  CHECK_NEAR(border.trailing, -st.kappa / st.tau, 1e-14);
}

/// With finite bounds the border row is NOT the negated border column, which
/// is what FORMULATION.md section 13.3 used to claim. If this ever passes by
/// coincidence the two oracles above would still catch it, but the claim is
/// worth asserting directly because it is the one a reader is most likely to
/// assume.
void test_border_row_is_not_the_negated_column() {
  model::CanonicalResult canon;
  if (!build_boxed(canon)) return;
  const model::CanonicalProblem& p = canon.problem;
  SolverState st = make_state(p, 1.3, 0.45);
  HomogeneousBorder border;
  CHECK(solver::compute_homogeneous_border(p, st, border).ok());

  Real separation = 0.0;
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    separation = std::fmax(separation, std::fabs(border.g_x[j] + border.h_x[j]));
    // They differ by exactly 2c -- both carry the same bound term.
    CHECK_NEAR(border.g_x[j] - border.h_x[j], 2.0 * p.c[j], 1e-12);
  }
  CHECK(separation > 1e-3);
  CHECK(border.w > 0.0);  // zero only in standard form
}

/// [AA] section 1.5's cost claim: the border solve is once per ITERATION and
/// shared by the predictor and the corrector, so two directions cost three
/// solves and not four.
void test_border_solve_is_shared_across_directions() {
  model::CanonicalResult canon;
  if (!build_boxed(canon)) return;
  const model::CanonicalProblem& p = canon.problem;
  SolverState st = make_state(p, 1.3, 0.45);
  HomogeneousBorder border;
  CHECK(solver::compute_homogeneous_border(p, st, border).ok());

  DenseKktSolver kkt(p, border);
  HomogeneousNewtonWorkspace work;
  CHECK(solver::refresh_border_solve(p, border, kkt, work).ok());
  CHECK_EQ(kkt.solves(), std::size_t{1});

  SolverState out = make_state(p, 1.3, 0.45);
  CHECK(solver::solve_homogeneous_newton(p, st, border, make_rhs(p, 1u), kkt, work,
                                         /*affine=*/true, out)
            .ok());
  CHECK_EQ(kkt.solves(), std::size_t{2});
  CHECK(solver::solve_homogeneous_newton(p, st, border, make_rhs(p, 2u), kkt, work,
                                         /*affine=*/false, out)
            .ok());
  CHECK_EQ(kkt.solves(), std::size_t{3});

  // The affine and corrected directions land in different slots and differ.
  bool differs = false;
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    if (std::fabs(out.dx_aff[j] - out.dx[j]) > 1e-9) differs = true;
  }
  CHECK(differs);
  CHECK(std::fabs(out.dtau_aff - out.dtau) > 1e-12);
}

/// A stale `(p; q)` is a wrong direction that still looks plausible, so it must
/// be refused rather than used.
void test_stale_border_solve_is_refused() {
  model::CanonicalResult canon;
  if (!build_boxed(canon)) return;
  const model::CanonicalProblem& p = canon.problem;
  SolverState st = make_state(p, 1.3, 0.45);
  HomogeneousBorder border;
  CHECK(solver::compute_homogeneous_border(p, st, border).ok());

  DenseKktSolver kkt(p, border);
  HomogeneousNewtonWorkspace work;  // never refreshed
  SolverState out = make_state(p, 1.3, 0.45);
  CHECK(!solver::solve_homogeneous_newton(p, st, border, make_rhs(p, 3u), kkt, work,
                                          false, out)
             .ok());
  CHECK_EQ(kkt.solves(), std::size_t{0});
}

/// Leaving the interior is an error with a name, not a NaN four modules later.
void test_non_interior_iterates_are_rejected() {
  model::CanonicalResult canon;
  if (!build_boxed(canon)) return;
  const model::CanonicalProblem& p = canon.problem;
  HomogeneousBorder border;

  SolverState zero_tau = make_state(p, 1.3, 0.45);
  zero_tau.tau = 0.0;
  CHECK(!solver::compute_homogeneous_border(p, zero_tau, border).ok());

  SolverState on_bound = make_state(p, 1.3, 0.45);
  on_bound.x[0] = p.col_lower[0] * on_bound.tau;  // exactly on the lower bound
  CHECK(!solver::compute_homogeneous_border(p, on_bound, border).ok());

  SolverState past_upper = make_state(p, 1.3, 0.45);
  past_upper.x[0] = p.col_upper[0] * past_upper.tau + 1.0;
  CHECK(!solver::compute_homogeneous_border(p, past_upper, border).ok());

  SolverState bad_slack = make_state(p, 1.3, 0.45);
  if (p.num_inequality_rows() > 0) {
    bad_slack.y[p.num_equality] = 0.0;  // the (s, -y_I) pair collapses
    CHECK(!solver::compute_homogeneous_border(p, bad_slack, border).ok());
  }
}

}  // namespace

int main() {
  test_reduced_solve_matches_the_full_system();
  test_direction_satisfies_the_published_standard_form_system();
  test_border_collapses_to_the_published_one();
  test_border_row_is_not_the_negated_column();
  test_border_solve_is_shared_across_directions();
  test_stale_border_solve_is_refused();
  test_non_interior_iterates_are_rejected();
  return ::sovsolve::test::report("homogeneous_newton_test");
}
