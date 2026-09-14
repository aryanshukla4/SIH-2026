#include "sovsolve/solver/Iis.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Transform.hpp"
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
  // `x <= 3` with `x >= 5` is infeasible with no rows at all, so a bound
  // carrying a nonzero multiplier belongs in the reported subsystem. The
  // multiplier on column `j` is the reduced cost the certificate induces,
  // `-(A' y)_j`, and its SIGN says which side participates: a positive value
  // pushes against the lower bound, a negative one against the upper.
  const auto& csc = canonical.A.csc;
  Real largest_column = 0.0;
  std::vector<Real> column_multiplier(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    Real acc = 0.0;
    for (std::size_t k = csc.slice_begin(j); k < csc.slice_end(j); ++k) {
      const auto row = static_cast<std::size_t>(csc.indices()[k]);
      if (row < y.size()) acc += csc.values()[k] * y[row];
    }
    column_multiplier[j] = -acc;
    largest_column = std::fmax(largest_column, std::fabs(acc));
  }

  // Canonical columns map back through `KeepColumn`.
  std::vector<std::size_t> column_to_original(n, static_cast<std::size_t>(-1));
  for (const model::TransformRecord& record : canon->transforms.records()) {
    if (record.kind != model::TransformKind::KeepColumn) continue;
    const auto canonical_index = static_cast<std::size_t>(record.secondary);
    if (canonical_index < n) {
      column_to_original[canonical_index] = static_cast<std::size_t>(record.primary);
    }
  }

  for (std::size_t j = 0; j < n; ++j) {
    if (std::fabs(column_multiplier[j]) <= kSupportTolerance * largest_column) {
      continue;
    }
    const std::size_t original = column_to_original[j];
    if (original == static_cast<std::size_t>(-1)) continue;
    if (column_multiplier[j] > 0.0) {
      if (core::is_finite_bound(canonical.col_lower[j])) {
        iis.lower_bound_columns.push_back(original);
      }
    } else if (core::is_finite_bound(canonical.col_upper[j])) {
      iis.upper_bound_columns.push_back(original);
    }
  }

  std::sort(iis.rows.begin(), iis.rows.end());
  iis.rows.erase(std::unique(iis.rows.begin(), iis.rows.end()), iis.rows.end());
  std::sort(iis.lower_bound_columns.begin(), iis.lower_bound_columns.end());
  std::sort(iis.upper_bound_columns.begin(), iis.upper_bound_columns.end());

  return iis;
}

}  // namespace sovsolve::solver
