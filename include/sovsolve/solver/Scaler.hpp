// Module 5: scale A, Q, c, and both bound pairs consistently, before
// regularization parameters (Module 11) are chosen against the result.
//
// Must run before delta_p/delta_d are picked: those floors are relative to
// SCALED data magnitudes, and gas11's pre-scaling dynamic range (2.6e11)
// makes an absolute floor meaningless (Options::IpmOptions).

#ifndef SOVSOLVE_SOLVER_SCALER_HPP
#define SOVSOLVE_SOLVER_SCALER_HPP

#include "sovsolve/core/Status.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Transform.hpp"

namespace sovsolve::solver {

using core::Real;
using core::Status;
using model::CanonicalProblem;
using model::Options;
using model::TransformRecord;
using model::TransformStack;

/// Geometric-mean row/column scaling, alternated a few passes:
///
///     row_scale[i]  <- 1 / sqrt(min_j |a_ij * col_scale[j]| * max_j |...|)
///     col_scale[j]  <- 1 / sqrt(min_i |a_ij * row_scale[i]| * max_i |...|)
///
/// each recomputed from the OTHER side's current value, so the two
/// converge toward each other rather than each chasing a stale snapshot.
///
/// Applied to A (both orientations, kept consistent), Q (Q'_jk = col_scale[j]
/// * Q_jk * col_scale[k] -- both indices are columns), b (row_scale[i] * b_i),
/// c (col_scale[j] * c_j), and both bound vectors (divided by col_scale[j],
/// finite entries only). Pushes ColumnScaling/RowScaling records onto
/// `transforms` -- the SAME stack the canonicalizer used -- keyed by
/// CANONICAL index (not original index, unlike every other record kind:
/// scaling runs after canonicalization, on the canonicalized problem).
/// model::recover_solution() inverts them: x = col_scale*x', y = row_scale*y',
/// z = z'/col_scale, v = v'/col_scale (Canonicalizer.cpp).
///
/// Must run before delta_p/delta_d are picked: those floors are relative to
/// SCALED data magnitudes, and gas11's pre-scaling dynamic range (2.6e11)
/// makes an absolute floor meaningless (Options::IpmOptions).
[[nodiscard]] Status scale(CanonicalProblem& problem, const Options& options,
                            TransformStack& transforms);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_SCALER_HPP
