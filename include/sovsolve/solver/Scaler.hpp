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

using core::Status;
using model::CanonicalProblem;
using model::Options;
using model::TransformStack;

/// Scales `problem` in place and pushes RowScaling/ColumnScaling records onto
/// `transforms` -- the SAME stack the canonicalizer used, so Solution
/// Reconstructor (Module 17) inverts canonicalization and scaling in one
/// reverse pass.
///
/// STUB: identity scaling. No records pushed yet. Real geometric-mean /
/// Curtis-Reid scaling lands once this skeleton compiles end-to-end.
[[nodiscard]] Status scale(CanonicalProblem& problem, const Options& options,
                            TransformStack& transforms);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_SCALER_HPP
