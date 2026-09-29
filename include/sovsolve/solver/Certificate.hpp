// LINEAR PROGRAMS ONLY: this is LP weak duality, with no Qx term.
//
// An optimality certificate measured on the ORIGINAL model, from the reported
// x and y alone -- independent of any engine's presolved, scaled working form
// and of the engine's own residuals.
//
//   primal   worst row violation, relative to 1 + |row bound|
//   dual     g(y), the weak-duality bound: each row dual times the finite row
//            bound it pushes on (a component pushing on an infinite bound is
//            clipped to 0, still a valid dual point), plus what the variable
//            bounds absorb of d = sigma c - A'y. Whatever of d no finite bound
//            can absorb is the dual residual, measured PER COLUMN against
//            1 + |c_j|: where it is zero, g is a true lower bound.
//   gap      |sigma c'x - g| / (1 + |sigma c'x| + |g|)
//
// The per-column measure is the point. A norm-relative residual (||.|| over
// 1 + ||c||) passed a wrong answer: Netlib greenbea on the GPU interior point,
// objective 1.3e-3 from the optimum, had norm-relative dual residual 6.8e-9
// but 2.0e-4 on individual columns. Standard weak duality; the reduced-cost
// projection is the one PDLP's dual objective uses (NeurIPS 2021, section 2).

#ifndef SOVSOLVE_SOLVER_CERTIFICATE_HPP
#define SOVSOLVE_SOLVER_CERTIFICATE_HPP

#include "sovsolve/core/Types.hpp"
#include "sovsolve/model/Problem.hpp"
#include "sovsolve/model/Solution.hpp"

namespace sovsolve::solver {

struct Certificate {
  bool valid = false;           ///< x and y had the model's sizes
  core::Real primal = 0.0;      ///< worst row violation, relative
  core::Real dual = 0.0;        ///< worst unabsorbed reduced cost, relative to 1 + |c_j|
  core::Real gap = 0.0;         ///< relative gap between c'x and g(y)
  core::Real dual_bound = 0.0;  ///< g(y) in the model's own sense, with its constant

  [[nodiscard]] core::Real worst() const noexcept {
    return primal > dual ? (primal > gap ? primal : gap) : (dual > gap ? dual : gap);
  }
};

[[nodiscard]] Certificate certify(const model::Problem& problem, const model::Solution& solution);

/// An Optimal answer must certify to this. OURS: a safety net for gross
/// errors (greenbea's 2e-4), two orders above the ~1e-8 of a sound solve.
inline constexpr core::Real kCertificateTolerance = 1e-6;

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_CERTIFICATE_HPP
