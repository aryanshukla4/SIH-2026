// Module 23: the host-only LP entry point.
//
// This is the dual simplex's counterpart to `gpu::solve_problem`
// (solver/gpu/Solve.hpp), and it runs the SAME pipeline around a different
// engine:
//
//     canonicalize -> presolve -> scale -> [engine] -> reconstruct_solution
//
// Three of those four stages are shared verbatim. `presolve()` and `scale()`
// operate on a `CanonicalProblem` and know nothing about how it will be
// solved, and `reconstruct_solution()` consumes values rather than a basis --
// so the only new code on this path is the engine itself.
//
// It lives in `sovsolve_solver`, not `sovsolve_solver_gpu`, and needs no CUDA.
// That is a deliberate consequence of putting the simplex on the host: before
// Module 23 the entire solver -- and `tools/solve` with it -- could only be
// built under WSL2 with the CUDA toolkit present, so nothing downstream of the
// canonicalizer was reachable from the default `release` preset at all.

#ifndef SOVSOLVE_SOLVER_LP_SOLVE_HPP
#define SOVSOLVE_SOLVER_LP_SOLVE_HPP

#include <functional>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Problem.hpp"
#include "sovsolve/model/Solution.hpp"
#include "sovsolve/solver/pdlp/MatVec.hpp"

namespace sovsolve::solver {

/// Solve `problem` as a continuous LP with the dual simplex, returning a
/// solution in the ORIGINAL model's variable space.
///
/// Integrality is ignored, exactly as `gpu::solve_problem` ignores it: this is
/// the relaxation engine, and the search that uses it is Module 22's job.
///
/// A quadratic objective is refused with `UnsupportedFeature` rather than
/// silently solved as an LP -- a simplex moves between vertices of the
/// feasible polytope, and a QP's optimum is generally not at one, so dropping
/// `Q` would return a confidently wrong answer instead of an error.
/// Supplies the `K`/`K'` implementation for `Method::Pdlp`, given the problem
/// AFTER canonicalize -> presolve -> scale.
///
/// A provider rather than an object, because the backend has to be built
/// against the canonical matrix and the caller does not have that until this
/// function is already running. It returns a NON-OWNING pointer and the caller
/// keeps the object alive across the call -- which is what lets a benchmark
/// read the backend's own counters afterwards instead of having them destroyed
/// with it.
///
/// The GPU implementation lives in `sovsolve_solver_gpu` and is injected here
/// rather than selected here, because the `gpu -> solver` library edge is
/// one-way (src/solver/CMakeLists.txt) and must stay that way.
using MatVecProvider =
    std::function<core::Expected<pdlp::MatVec*>(const model::CanonicalProblem&)>;

[[nodiscard]] core::Expected<model::Solution> solve_lp(
    const model::Problem& problem, const model::Options& options = {},
    const MatVecProvider& matvec_provider = {});

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_LP_SOLVE_HPP
