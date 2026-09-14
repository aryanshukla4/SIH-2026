// Module 26: Irreducible Infeasible Subsystem.
//
// "This model is infeasible" is a verdict. "These four rows contradict each
// other, and removing any one of them makes the rest satisfiable" is something
// a person can act on. That second answer is an IIS -- an infeasible subsystem
// every proper subsystem of which is feasible -- and it is what a planner
// actually wants when a schedule will not solve.
//
// --------------------------------------------------------------------------
// Why this is nearly free here
// --------------------------------------------------------------------------
//
// Gleeson and Ryan (1990), as stated in Kellner, Pfetsch and Theobald,
// "Irreducible Infeasible Subsystems of Semidefinite Systems" (arXiv
// 1804.01327) Theorem 3.4:
//
//     For an infeasible system  Ax <= b,  the index sets of the IISs are
//     exactly the SUPPORT SETS of the VERTICES of the alternative polyhedron
//
//         P = { y : y'A = 0,  y'b = -1,  y >= 0 }
//
// A point of `P` is precisely a Farkas certificate of infeasibility, scaled so
// that `y'b = -1`. This project already computes one, twice over and by two
// unrelated routes:
//
//   * the dual simplex, when `collect_candidates()` finds no column able to
//     absorb the dual step (the certificate is a row of `B^-1`), and
//   * PDLP, from the infimal displacement vector (pdlp/Infeasibility.hpp).
//
// So the IIS is not a new computation. It is the SUPPORT of a certificate we
// already have -- the rows whose multiplier is nonzero.
//
// The word "vertex" in the theorem is what separates an IIS from a merely
// infeasible subsystem, and it is why the SIMPLEX certificate is the better
// input of the two: a row of `B^-1` is a basic solution, hence already a
// vertex of the alternative polyhedron, so its support is irreducible by
// construction. PDLP's certificate is a limit of interior iterates with no
// such guarantee -- it may have extra rows in its support. Both are accepted
// here, and the result says which guarantee applies.
//
// --------------------------------------------------------------------------
// Our form, not the theorem's
// --------------------------------------------------------------------------
//
// Theorem 3.4 is stated for pure `Ax <= b` with no variable bounds. Our
// canonical form has equality rows and a box, and both matter:
//
//   * An EQUALITY row's multiplier is free in sign, so its support test is
//     `|y_i| > tol` rather than `y_i > tol`.
//   * A finite variable BOUND can participate in the contradiction just as a
//     row can -- `x <= 3` together with `x >= 5` is infeasible with no rows at
//     all. So a bound that carries a nonzero reduced-cost multiplier in the
//     certificate belongs in the reported subsystem, and is reported as a
//     bound rather than silently dropped.
//
// Indices are reported in the ORIGINAL model's space, not the canonical one --
// a user asked "which of my constraints" means the ones they wrote, not the
// ones the canonicalizer produced.

#ifndef SOVSOLVE_SOLVER_IIS_HPP
#define SOVSOLVE_SOLVER_IIS_HPP

#include <cstddef>
#include <vector>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Problem.hpp"

namespace sovsolve::solver {

using core::Real;

/// How much the result can be trusted, which depends on where the certificate
/// came from -- see the header comment.
enum class IisQuality : std::uint8_t {
  /// The certificate was a basic solution (a vertex of the alternative
  /// polyhedron), so by Theorem 3.4 the support is a genuine IIS: irreducible.
  Irreducible,
  /// The certificate was not known to be a vertex, so the support is an
  /// infeasible subsystem that may not be minimal. Still far more useful than
  /// the whole model, and honestly labelled.
  InfeasibleSubsystem,
};

struct Iis {
  IisQuality quality = IisQuality::InfeasibleSubsystem;

  /// Row indices in the ORIGINAL problem.
  std::vector<std::size_t> rows;
  /// Columns whose finite bound participates in the contradiction, with which
  /// side participates.
  std::vector<std::size_t> lower_bound_columns;
  std::vector<std::size_t> upper_bound_columns;

  [[nodiscard]] bool empty() const noexcept {
    return rows.empty() && lower_bound_columns.empty() &&
           upper_bound_columns.empty();
  }
  [[nodiscard]] std::size_t size() const noexcept {
    return rows.size() + lower_bound_columns.size() + upper_bound_columns.size();
  }
};

/// Computes an IIS for a model already known to be infeasible.
///
/// Re-solves internally to obtain the certificate, so the caller does not have
/// to have kept one. Returns `ErrorCode::InvalidArgument`-shaped failure if
/// the model turns out NOT to be infeasible -- asking for the contradiction in
/// a satisfiable model is a caller error worth reporting rather than answering
/// with an empty set.
[[nodiscard]] core::Expected<Iis> compute_iis(const model::Problem& problem,
                                              const model::Options& options);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_IIS_HPP
