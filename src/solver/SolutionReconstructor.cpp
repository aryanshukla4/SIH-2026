#include "sovsolve/solver/SolutionReconstructor.hpp"

namespace sovsolve::solver {

Expected<Solution> reconstruct_solution(const Problem& original,
                                         const CanonicalProblem& canonical,
                                         const TransformStack& transforms,
                                         const Solution& canonical_solution) {
  return model::recover_solution(original, canonical, transforms, canonical_solution);
}

}  // namespace sovsolve::solver
