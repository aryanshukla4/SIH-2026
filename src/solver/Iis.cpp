#include "sovsolve/solver/Iis.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Transform.hpp"
#include "sovsolve/solver/LpSolve.hpp"
#include "sovsolve/solver/Scaler.hpp"
#include "sovsolve/solver/simplex/SolveSimplex.hpp"

namespace sovsolve::solver {

namespace {

using core::ErrorCode;
using core::Real;

/// Support threshold. A multiplier this small is rounding in the
/// factorization, not a row participating in the contradiction -- and
/// including it would report a subsystem that is not irreducible.
constexpr Real kSupportTolerance = 1e-9;

/// Inverse of the canonicalizer's `MapRow` records: canonical row -> original.
///
/// Every original row pushes exactly one of `MapRow`, `RemoveEmptyRow` or
/// `DropFreeRow` (Transform.hpp), and only `MapRow` produces a canonical row,
/// so reading those records gives an exact inverse with nothing to infer.
std::vector<std::size_t> canonical_to_original_rows(
    const model::TransformStack& transforms, std::size_t canonical_rows) {
  std::vector<std::size_t> map(canonical_rows, static_cast<std::size_t>(-1));
  for (const model::TransformRecord& record : transforms.records()) {
    if (record.kind != model::TransformKind::MapRow) continue;
    const auto canonical = static_cast<std::size_t>(record.secondary);
    if (canonical < canonical_rows) {
      map[canonical] = static_cast<std::size_t>(record.primary);
    }
  }
  return map;
}

}  // namespace

core::Expected<Iis> compute_iis(const model::Problem& problem,
                                const model::Options& options) {
  if (problem.has_quadratic()) {
    return core::make_error(ErrorCode::UnsupportedFeature,
                            "compute_iis: an IIS is defined for a linear "
                            "constraint system; this model has a quadratic "
                            "objective");
  }

  // PRESOLVE IS DISABLED for this solve, deliberately.
  //
  // Presolve removes, merges and substitutes rows, so a certificate computed
  // on the reduced model has a support in the REDUCED row space -- and the
  // rows it names are not the rows the user wrote. Mapping back through the
  // full transform stack would be possible but would also answer a different
  // question: a contradiction presolve already folded away is still a
  // contradiction the user needs to see. Running on the unreduced model keeps
  // `MapRow` an exact inverse and keeps the answer in the user's own terms.
  model::Options iis_options = options;
  iis_options.presolve.enabled = false;
  iis_options.simplex.method = model::Method::DualSimplex;
  // The dual simplex is required rather than preferred: its certificate is a
  // row of `B^-1`, hence a vertex of the alternative polyhedron, which is
  // exactly the hypothesis Gleeson and Ryan's theorem needs for the support to
  // be IRREDUCIBLE. The primal simplex proves infeasibility by a different
  // route (phase 1 ending with positive infeasibility) that yields no such
  // vector, and PDLP's is a limit of interior points with no vertex guarantee.
  iis_options.simplex.primal_cleanup = false;

  auto canon = model::canonicalize(problem, iis_options);
  if (!canon.has_value()) {
    if (canon.error().code == ErrorCode::PrimalInfeasible) {
      // Infeasible by inspection during canonicalization -- a bound pair with
      // lower above upper, or an all-zero row excluding zero. There is no
      // certificate to take a support of, and there does not need to be: the
      // contradiction is one row or one column, and the canonicalizer already
      // knows which. Reported as an error rather than guessed at.
      return core::make_error(ErrorCode::PrimalInfeasible,
                              "compute_iis: the model is infeasible by "
                              "inspection during canonicalization, so no "
                              "iterative certificate exists: " +
                                  canon.error().message);
    }
    return canon.error();
  }

  // Scaling is applied because the simplex is numerically much better behaved
  // for it, and it cannot change which rows appear in the support: a row scale
  // is a positive multiplier, so it rescales a certificate entry without
  // moving it on or off zero.
  core::Status status = scale(canon->problem, iis_options, canon->transforms);
  if (!status.ok()) return status.error();

  auto result = simplex::solve_simplex(canon->problem, iis_options);
  if (!result.has_value()) return result.error();

  if (result->status != core::SolverStatus::Infeasible) {
    return core::make_error(
        ErrorCode::InconsistentBounds,
        "compute_iis: the model is not infeasible, so it has no irreducible "
        "infeasible subsystem");
  }
  if (result->infeasibility_certificate.empty()) {
    return core::make_error(ErrorCode::NotImplemented,
                            "compute_iis: infeasibility was proved without "
                            "producing a Farkas certificate, so no subsystem "
                            "can be extracted");
  }

  const std::vector<Real>& y = result->infeasibility_certificate;
  const model::CanonicalProblem& canonical = canon->problem;
  const std::size_t m = canonical.num_rows();
  const std::size_t n = canonical.num_cols();

  Real largest = 0.0;
  for (std::size_t i = 0; i < m && i < y.size(); ++i) {
    largest = std::fmax(largest, std::fabs(y[i]));
  }
  if (!(largest > 0.0)) {
    return core::make_error(ErrorCode::NumericalError,
                            "compute_iis: the certificate is identically zero");
  }

  Iis iis;
  iis.quality = IisQuality::Irreducible;

  // ---- rows: the support of the certificate ------------------------------
  const std::vector<std::size_t> to_original =
      canonical_to_original_rows(canon->transforms, m);
  for (std::size_t i = 0; i < m && i < y.size(); ++i) {
    // Equality rows carry a free-signed multiplier, inequality rows a signed
    // one, but membership of the SUPPORT is a magnitude test either way.
    if (std::fabs(y[i]) <= kSupportTolerance * largest) continue;
    const std::size_t original = to_original[i];
    if (original == static_cast<std::size_t>(-1)) continue;
    iis.rows.push_back(original);
  }

  // ---- bounds: a finite bound can be part of the contradiction ------------
  //
  // `x <= 3` with `x >= 5` is infeasible with no rows at all, so a bound the
  // certificate leans on belongs in the reported subsystem. WHICH side of a
  // column's box it leans on cannot be read off the sign of `A'y` alone: the
  // dual simplex hands back `sigma * rho`, and nothing fixes its overall sign.
  // Reading it that way reported the wrong side -- usually an infinite one,
  // which was then dropped -- and left subsystems that HiGHS found FEASIBLE
  // (Netlib GRAN, PANG, PILOT4I and others).
  //
  // So the side is decided the way the proof itself works. The engine solves
  // `A x + s = b` with one logical `s_i` per row (coefficient +1, fixed at 0 on
  // an equality row, `>= 0` on an inequality row). Every solution satisfies
  // `y'b = sum_j g_j x_j + sum_i y_i s_i` with `g = A'y`; the certificate is a
  // proof because `y'b` lies above the largest value the right side can take
  // over the box, or below the smallest. Whichever of the two holds names the
  // participating bounds: on the "above" side a column contributes its upper
  // bound when `g_j > 0` and its lower when `g_j < 0`; on the "below" side the
  // reverse.
  const auto& csc = canonical.A.csc;
  std::vector<Real> g(n, 0.0);
  Real largest_g = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    Real acc = 0.0;
    for (std::size_t k = csc.slice_begin(j); k < csc.slice_end(j); ++k) {
      const auto row = static_cast<std::size_t>(csc.indices()[k]);
      if (row < y.size()) acc += csc.values()[k] * y[row];
    }
    g[j] = acc;
    largest_g = std::fmax(largest_g, std::fabs(acc));
  }
  const Real g_zero = kSupportTolerance * std::fmax(largest_g, largest);

  Real yb = 0.0;
  Real high = 0.0;
  Real low = 0.0;
  bool high_finite = true;
  bool low_finite = true;
  Real magnitude = 0.0;
  for (std::size_t i = 0; i < m && i < y.size(); ++i) {
    yb += y[i] * canonical.b[i];
    magnitude += std::fabs(y[i] * canonical.b[i]);
    // The row's logical: [0, 0] on an equality row contributes nothing;
    // [0, inf) on an inequality row is unbounded on the side y_i points to.
    if (i >= canonical.num_equality) {
      if (y[i] > g_zero) high_finite = false;
      if (y[i] < -g_zero) low_finite = false;
    }
  }
  for (std::size_t j = 0; j < n; ++j) {
    if (std::fabs(g[j]) <= g_zero) continue;
    const Real at_max = g[j] > 0.0 ? canonical.col_upper[j] : canonical.col_lower[j];
    const Real at_min = g[j] > 0.0 ? canonical.col_lower[j] : canonical.col_upper[j];
    if (core::is_finite_bound(at_max)) {
      high += g[j] * at_max;
      magnitude += std::fabs(g[j] * at_max);
    } else {
      high_finite = false;
    }
    if (core::is_finite_bound(at_min)) {
      low += g[j] * at_min;
      magnitude += std::fabs(g[j] * at_min);
    } else {
      low_finite = false;
    }
  }
  const Real proof_tol =
      iis_options.simplex.primal_feasibility_tolerance * largest + 1e-9 * magnitude;
  const bool above = high_finite && yb > high + proof_tol;
  const bool below = low_finite && yb < low - proof_tol;
  // An AMBIGUOUS support is refused. The support is the set of multipliers
  // above `kSupportTolerance`; a multiplier far below it but still well above
  // rounding (1e-14 relative) is one the threshold drops even though the
  // contradiction may need it. Measured on the 29 Netlib infeasible models:
  // the two whose extracted subsystems HiGHS and SoPlex found FEASIBLE (QUAL,
  // VOL1) are exactly the ones with such entries, and every other model but
  // one (GOSH, whose answer was right) has none. No threshold choice fixes
  // that -- moving it to 1e-13 left HiGHS unable to decide either way -- so a
  // certificate with any multiplier in that band yields no IIS at all.
  constexpr Real kRoundingFloor = 1e-14;
  for (std::size_t i = 0; i < m && i < y.size(); ++i) {
    const Real r = std::fabs(y[i]) / largest;
    if (r > kRoundingFloor && r <= kSupportTolerance) {
      return core::make_error(ErrorCode::NumericalError,
                              "compute_iis: the certificate's support is "
                              "ambiguous (a row multiplier sits between "
                              "rounding and the support threshold)");
    }
  }
  for (std::size_t j = 0; j < n; ++j) {
    const Real r = std::fabs(g[j]) / std::fmax(largest_g, largest);
    if (r > kRoundingFloor && r <= kSupportTolerance) {
      return core::make_error(ErrorCode::NumericalError,
                              "compute_iis: the certificate's support is "
                              "ambiguous (a bound multiplier sits between "
                              "rounding and the support threshold)");
    }
  }
  if (!above && !below) {
    // The certificate does not verify against the box it is supposed to
    // contradict. Reporting its support anyway would name a subsystem that
    // may well be feasible, so this is an error, not a guess.
    return core::make_error(ErrorCode::NumericalError,
                            "compute_iis: the infeasibility certificate does "
                            "not verify against the model's bounds");
  }
  // On the "above" side a column leans on its UPPER bound when g > 0.
  const auto leans_on_upper = [above](Real gj) { return above ? gj > 0.0 : gj < 0.0; };

  // Canonical columns map back through `KeepColumn`; range columns
  // (index >= the kept count) stand for a row's range, and that row is
  // already in the subsystem.
  std::vector<std::size_t> column_to_original(n, static_cast<std::size_t>(-1));
  // Per ORIGINAL row: its canonical row, and whether the canonicalizer
  // negated it (`a'x >= b` -> `-a'x <= -b`).
  std::vector<std::size_t> row_to_canonical(problem.num_rows(), static_cast<std::size_t>(-1));
  std::vector<char> row_negated(problem.num_rows(), 0);
  // Per canonical row: the scale the Scaler multiplied it by.
  std::vector<Real> row_scale(m, 1.0);
  std::vector<std::size_t> fixed_columns;
  for (const model::TransformRecord& record : canon->transforms.records()) {
    const auto primary = static_cast<std::size_t>(record.primary);
    switch (record.kind) {
      case model::TransformKind::KeepColumn:
        if (static_cast<std::size_t>(record.secondary) < n) {
          column_to_original[static_cast<std::size_t>(record.secondary)] = primary;
        }
        break;
      case model::TransformKind::RemoveFixedVariable:
        fixed_columns.push_back(primary);
        break;
      case model::TransformKind::MapRow:
        if (primary < row_to_canonical.size()) {
          row_to_canonical[primary] = static_cast<std::size_t>(record.secondary);
        }
        break;
      case model::TransformKind::NegateRow:
        if (primary < row_negated.size()) row_negated[primary] = 1;
        break;
      case model::TransformKind::RowScaling:
        if (primary < m) row_scale[primary] = record.value;
        break;
      default:
        break;
    }
  }

  for (std::size_t j = 0; j < n; ++j) {
    if (std::fabs(g[j]) <= g_zero) continue;
    const std::size_t original = column_to_original[j];
    if (original == static_cast<std::size_t>(-1)) continue;
    if (leans_on_upper(g[j])) {
      if (core::is_finite_bound(canonical.col_upper[j])) {
        iis.upper_bound_columns.push_back(original);
      }
    } else if (core::is_finite_bound(canonical.col_lower[j])) {
      iis.lower_bound_columns.push_back(original);
    }
  }

  // FIXED columns were substituted out: their value is folded into `b`, so
  // they are in no canonical column, yet the contradiction may rest on them
  // as surely as on any bound. Their `g` is recomputed from the ORIGINAL
  // matrix, in the scaled canonical row space the certificate lives in:
  // canonical row = row_scale * (+/-1) * original row.
  const auto& ocsc = problem.A.csc;
  for (const std::size_t f : fixed_columns) {
    Real gf = 0.0;
    for (std::size_t k = ocsc.slice_begin(f); k < ocsc.slice_end(f); ++k) {
      const auto orow = static_cast<std::size_t>(ocsc.indices()[k]);
      if (orow >= row_to_canonical.size()) continue;
      const std::size_t crow = row_to_canonical[orow];
      if (crow >= m || crow >= y.size()) continue;
      const Real sign = row_negated[orow] ? -1.0 : 1.0;
      gf += y[crow] * row_scale[crow] * sign * ocsc.values()[k];
    }
    if (std::fabs(gf) <= g_zero) continue;
    // A fixed column's two bounds coincide; the side is still the one the
    // proof pushes against, so it is reported exactly as a kept column is.
    if (leans_on_upper(gf)) {
      iis.upper_bound_columns.push_back(f);
    } else {
      iis.lower_bound_columns.push_back(f);
    }
  }

  std::sort(iis.rows.begin(), iis.rows.end());
  iis.rows.erase(std::unique(iis.rows.begin(), iis.rows.end()), iis.rows.end());
  std::sort(iis.lower_bound_columns.begin(), iis.lower_bound_columns.end());
  std::sort(iis.upper_bound_columns.begin(), iis.upper_bound_columns.end());

  // ---- self-check: the reported subsystem must itself be infeasible -------
  //
  // The support above is only as good as the certificate's numerics. On a
  // nearly-feasible model (Netlib QUAL, VOL1) a multiplier at the edge of the
  // support threshold can drop a row or bound the contradiction needs, and
  // the "subsystem" left is FEASIBLE -- a wrong answer that looks like a
  // right one. So it is solved once more, alone: every row and bound outside
  // it relaxed to infinity, no objective. Anything but an Infeasible verdict
  // means no IIS is reported. A missing answer is recoverable; a wrong
  // explanation of an infeasible plan is not.
  {
    model::Problem sub = problem.clone();
    std::vector<char> keep_row(problem.num_rows(), 0);
    for (const std::size_t i : iis.rows) keep_row[i] = 1;
    for (std::size_t i = 0; i < problem.num_rows(); ++i) {
      if (keep_row[i] != 0) continue;
      sub.row_lower[i] = -core::INF;
      sub.row_upper[i] = core::INF;
    }
    std::vector<char> keep_lower(problem.num_cols(), 0);
    std::vector<char> keep_upper(problem.num_cols(), 0);
    for (const std::size_t j : iis.lower_bound_columns) keep_lower[j] = 1;
    for (const std::size_t j : iis.upper_bound_columns) keep_upper[j] = 1;
    for (std::size_t j = 0; j < problem.num_cols(); ++j) {
      sub.c[j] = 0.0;
      if (keep_lower[j] == 0) sub.col_lower[j] = -core::INF;
      if (keep_upper[j] == 0) sub.col_upper[j] = core::INF;
    }
    sub.obj_constant = 0.0;

    model::Options check_options = options;
    check_options.simplex.method = model::Method::DualSimplex;
    check_options.presolve.enabled = true;
    check_options.simplex.primal_cleanup = true;
    auto check = solve_lp(sub, check_options);
    if (!check.has_value() || check->status != core::SolverStatus::Infeasible) {
      return core::make_error(ErrorCode::NumericalError,
                              "compute_iis: the extracted subsystem did not "
                              "re-solve as infeasible, so it is not reported");
    }
  }

  return iis;
}

}  // namespace sovsolve::solver
