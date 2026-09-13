// CLI entry point: load a model file, solve it, and print the result in a form
// a benchmark script can parse.
//
// `--method` chooses the engine. The interior-point path (gpu::solve, which
// also dispatches branch-and-bound for a model with discrete columns -- see
// BranchAndBound.hpp) lives in sovsolve_solver_gpu and is compiled in only
// under SOVSOLVE_ENABLE_CUDA. The two simplex paths (Module 23) are host-only,
// so this tool now builds and runs without a CUDA toolkit at all -- before
// Module 23 it was skipped entirely on a host-only build, which left
// everything downstream of the canonicalizer unreachable from the default
// `release` preset.
//
// Every `--flag=value` below is a direct field on Options (Options.hpp) --
// this file does not invent new parameters, it just exposes the existing
// ones so tuning doesn't require an edit-rebuild cycle per attempt.

#include <cstdio>
#include <cstdlib>
#include <string>

#include "sovsolve/io/Load.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/solver/LpSolve.hpp"
#ifdef SOVSOLVE_ENABLE_CUDA
#include "sovsolve/solver/gpu/BranchAndBound.hpp"
#endif

namespace {

const char* status_name(sovsolve::core::SolverStatus status) {
  using sovsolve::core::SolverStatus;
  switch (status) {
    case SolverStatus::Optimal: return "Optimal";
    case SolverStatus::Infeasible: return "Infeasible";
    case SolverStatus::Unbounded: return "Unbounded";
    case SolverStatus::Nonconvex: return "Nonconvex";
    case SolverStatus::MaxIterations: return "MaxIterations";
    case SolverStatus::TimeLimit: return "TimeLimit";
    case SolverStatus::NumericalError: return "NumericalError";
    case SolverStatus::NotConverged: return "NotConverged";
  }
  return "Unknown";
}

void print_usage(const char* argv0) {
  std::fprintf(stderr,
      "usage: %s <model-file> [max_iterations] [--flag=value ...]\n"
      "\n"
      "tuning flags (each is a field on model::Options -- see Options.hpp\n"
      "for the full doc comment on why each default is what it is):\n"
      "\n"
      "  --method=ipm|dual-simplex|primal-simplex   SimplexOptions::method\n"
      "                        (default ipm; the two simplex paths are host-only,\n"
      "                        and are the only engines a non-CUDA build has)\n"
      "  --simplex-max-iter=N  SimplexOptions::max_iterations   (0 = auto)\n"
      "  --pivot-tolerance=X   SimplexOptions::pivot_tolerance  (default 0.1)\n"
      "  --pivot-floor=X       SimplexOptions::pivot_floor      (default 1e-9)\n"
      "  --refactor-interval=N SimplexOptions::refactor_interval (default 100)\n"
      "  --artificial-bound=X  SimplexOptions::artificial_bound  (default 1e7)\n"
      "  --primal-cleanup=0|1  SimplexOptions::primal_cleanup    (default 1 --\n"
      "                        finish a dual run that ended without a verdict\n"
      "                        by handing its basis to the primal simplex)\n"
      "  --bound-flipping=0|1  SimplexOptions::bound_flipping    (default 1)\n"
      "  --simplex-tol-primal=X  SimplexOptions::primal_feasibility_tolerance\n"
      "  --simplex-tol-dual=X    SimplexOptions::dual_feasibility_tolerance\n"
      "\n"
      "  --max-iter=N          Limits::max_iterations       (default 200,\n"
      "                        interior-point only -- see --simplex-max-iter)\n"
      "  --time-limit=S        Limits::time_limit_seconds   (default 3600)\n"
      "  --stall=N             Limits::stall_iterations     (default 10)\n"
      "  --tol-primal=X        Tolerances::primal_feasibility (default 1e-8)\n"
      "  --tol-dual=X          Tolerances::dual_feasibility   (default 1e-8)\n"
      "  --tol-gap=X           Tolerances::relative_gap       (default 1e-8)\n"
      "  --tol-abs-gap=X       Tolerances::absolute_gap       (default 1e-8)\n"
      "  --eta=X               IpmOptions::eta               (default 0.995)\n"
      "  --sigma=X             IpmOptions::sigma (fixed-sigma path only, default 0.1)\n"
      "  --predictor-corrector=0|1  IpmOptions::predictor_corrector (default 1)\n"
      "  --pfloor=X            IpmOptions::primal_regularization_floor (default 1e-8)\n"
      "  --dfloor=X            IpmOptions::dual_regularization_floor   (default 1e-8)\n"
      "  --escalation=X        IpmOptions::regularization_escalation  (default 100)\n"
      "  --decay=X             IpmOptions::regularization_decay       (default 10)\n"
      "  --delta-max=X         IpmOptions::delta_max                  (default 1e-2)\n"
      "  --max-pivot-ratio=X   IpmOptions::max_pivot_ratio            (default 1e10)\n"
      "  --refine=N            IpmOptions::max_refinement_steps       (default 3,\n"
      "                        currently unused -- refinement isn't implemented yet)\n"
      "  --normal-eq=0|1       IpmOptions::use_normal_equations       (default 0,\n"
      "                        LP only -- ignored for QP, see Options.hpp)\n"
      "  --cg-tol=X            IpmOptions::cg_tolerance               (default 1e-10)\n"
      "  --cg-max-iter=N       IpmOptions::cg_max_iterations          (default 500)\n"
      "  --minres-tol=X        IpmOptions::minres_tolerance           (default 1e-10)\n"
      "  --minres-max-iter=N   IpmOptions::minres_max_iterations      (default 5000 --\n"
      "                        measured necessary, see Options.hpp)\n"
      "  --presolve=0|1        PresolveOptions::enabled               (default 1)\n"
      "  --mip-int-tol=X       MilpOptions::integer_tolerance         (default 1e-6,\n"
      "                        MILP only -- ignored for a pure LP/QP model)\n"
      "  --mip-node-limit=N    MilpOptions::node_limit                (default 100000)\n"
      "  --mip-time-limit=S    MilpOptions::time_limit_seconds        (default 3600)\n"
      "  --mip-gap=X           MilpOptions::gap_tolerance             (default 1e-9)\n"
      "\n"
      "output is one `key=value` line per metric, ending with\n"
      "`solve_time_seconds=...` -- that's the number to optimize against.\n",
      argv0);
}

/// Parses `--flag=value` into `options`. Returns false (and prints why) on an
/// unrecognized flag or a value that doesn't parse -- fail loud rather than
/// silently run with a typo'd flag ignored.
bool apply_flag(const std::string& flag, sovsolve::model::Options& options) {
  const auto eq = flag.find('=');
  if (flag.rfind("--", 0) != 0 || eq == std::string::npos) return false;
  const std::string key = flag.substr(2, eq - 2);
  const std::string val = flag.substr(eq + 1);

  try {
    if (key == "method") {
      if (val == "ipm" || val == "interior-point") {
        options.simplex.method = sovsolve::model::Method::InteriorPoint;
      } else if (val == "dual-simplex" || val == "simplex") {
        options.simplex.method = sovsolve::model::Method::DualSimplex;
      } else if (val == "primal-simplex" || val == "primal") {
        options.simplex.method = sovsolve::model::Method::PrimalSimplex;
      } else {
        std::fprintf(stderr, "unknown --method: %s (ipm, dual-simplex or primal-simplex)\n",
                     val.c_str());
        return false;
      }
    } else if (key == "simplex-max-iter") {
      options.simplex.max_iterations = static_cast<std::size_t>(std::stoul(val));
    } else if (key == "pivot-tolerance") {
      options.simplex.pivot_tolerance = std::stod(val);
    } else if (key == "pivot-floor") {
      options.simplex.pivot_floor = std::stod(val);
    } else if (key == "refactor-interval") {
      options.simplex.refactor_interval = static_cast<std::size_t>(std::stoul(val));
    } else if (key == "artificial-bound") {
      options.simplex.artificial_bound = std::stod(val);
    } else if (key == "primal-cleanup") {
      options.simplex.primal_cleanup = std::stoul(val) != 0;
    } else if (key == "bound-flipping") {
      options.simplex.bound_flipping = std::stoul(val) != 0;
    } else if (key == "simplex-tol-primal") {
      options.simplex.primal_feasibility_tolerance = std::stod(val);
    } else if (key == "simplex-tol-dual") {
      options.simplex.dual_feasibility_tolerance = std::stod(val);
    } else if (key == "max-iter") {
      options.limits.max_iterations = static_cast<std::size_t>(std::stoul(val));
    } else if (key == "time-limit") {
      options.limits.time_limit_seconds = std::stod(val);
    } else if (key == "stall") {
      options.limits.stall_iterations = static_cast<std::size_t>(std::stoul(val));
    } else if (key == "tol-primal") {
      options.tolerances.primal_feasibility = std::stod(val);
    } else if (key == "tol-dual") {
      options.tolerances.dual_feasibility = std::stod(val);
    } else if (key == "tol-gap") {
      options.tolerances.relative_gap = std::stod(val);
    } else if (key == "tol-abs-gap") {
      options.tolerances.absolute_gap = std::stod(val);
    } else if (key == "eta") {
      options.ipm.eta = std::stod(val);
    } else if (key == "sigma") {
      options.ipm.sigma = std::stod(val);
    } else if (key == "predictor-corrector") {
      options.ipm.predictor_corrector = std::stoi(val) != 0;
    } else if (key == "pfloor") {
      options.ipm.primal_regularization_floor = std::stod(val);
    } else if (key == "dfloor") {
      options.ipm.dual_regularization_floor = std::stod(val);
    } else if (key == "escalation") {
      options.ipm.regularization_escalation = std::stod(val);
    } else if (key == "decay") {
      options.ipm.regularization_decay = std::stod(val);
    } else if (key == "delta-max") {
      options.ipm.delta_max = std::stod(val);
    } else if (key == "max-pivot-ratio") {
      options.ipm.max_pivot_ratio = std::stod(val);
    } else if (key == "refine") {
      options.ipm.max_refinement_steps = std::stoi(val);
    } else if (key == "normal-eq") {
      options.ipm.use_normal_equations = std::stoi(val) != 0;
    } else if (key == "cg-tol") {
      options.ipm.cg_tolerance = std::stod(val);
    } else if (key == "cg-max-iter") {
      options.ipm.cg_max_iterations = std::stoi(val);
    } else if (key == "minres-tol") {
      options.ipm.minres_tolerance = std::stod(val);
    } else if (key == "minres-max-iter") {
      options.ipm.minres_max_iterations = std::stoi(val);
    } else if (key == "presolve") {
      options.presolve.enabled = std::stoi(val) != 0;
    } else if (key == "mip-int-tol") {
      options.milp.integer_tolerance = std::stod(val);
    } else if (key == "mip-node-limit") {
      options.milp.node_limit = static_cast<std::size_t>(std::stoul(val));
    } else if (key == "mip-time-limit") {
      options.milp.time_limit_seconds = std::stod(val);
    } else if (key == "mip-gap") {
      options.milp.gap_tolerance = std::stod(val);
    } else {
      std::fprintf(stderr, "unknown flag: --%s\n", key.c_str());
      return false;
    }
  } catch (const std::exception&) {
    std::fprintf(stderr, "bad value for --%s: %s\n", key.c_str(), val.c_str());
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  // Line-buffer stdout even when it's not a TTY (piped/redirected, the usual
  // case for a benchmark sweep): libc otherwise fully buffers non-interactive
  // stdout, so per-iteration diagnostics (Logging.cpp) accumulate silently
  // and are LOST if the process is killed (e.g. by `timeout`) before it
  // exits normally -- exactly the failure mode that made an earlier
  // diagnostic run of this tool look like several instances produced no
  // output at all, when they were actually still running.
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  if (argc < 2) {
    print_usage(argv[0]);
    return 2;
  }
  const std::string path = argv[1];

  sovsolve::model::Options options;

  int next_arg = 2;
  // Positional max_iterations stays for backward compatibility with existing
  // scripts/muscle-memory; everything else is --flag=value.
  if (next_arg < argc && std::string(argv[next_arg]).rfind("--", 0) != 0) {
    options.limits.max_iterations = static_cast<std::size_t>(std::stoul(argv[next_arg]));
    ++next_arg;
  }
  for (; next_arg < argc; ++next_arg) {
    if (std::string(argv[next_arg]) == "--help" || std::string(argv[next_arg]) == "-h") {
      print_usage(argv[0]);
      return 0;
    }
    if (!apply_flag(argv[next_arg], options)) {
      print_usage(argv[0]);
      return 2;
    }
  }

  auto problem = sovsolve::io::loadProblem(path);
  if (!problem.has_value()) {
    std::fprintf(stderr, "load failed: %s\n", problem.error().format().c_str());
    return 1;
  }

  const bool is_milp = problem->has_discrete();
  // Anything that is not the interior-point method is a simplex, and both of
  // them enter through solver::solve_lp. Testing for DualSimplex alone silently
  // routed --method=primal-simplex into the IPM branch.
  const bool use_simplex =
      options.simplex.method != sovsolve::model::Method::InteriorPoint;

  // The dual simplex is host-only (src/solver/CMakeLists.txt), so this tool
  // now builds and runs without the CUDA toolkit -- it previously could not be
  // built at all outside WSL2, which left every stage downstream of the
  // canonicalizer unreachable from the default `release` preset.
  auto solution = use_simplex
                      ? sovsolve::solver::solve_lp(*problem, options)
#ifdef SOVSOLVE_ENABLE_CUDA
                      : sovsolve::solver::gpu::solve(*problem, options);
#else
                      : sovsolve::core::Expected<sovsolve::model::Solution>(
                            sovsolve::core::make_error(
                                sovsolve::core::ErrorCode::NotImplemented,
                                "this build has no CUDA, so the interior-point path is "
                                "absent; pass --method=dual-simplex"));
#endif
  if (!solution.has_value()) {
    std::fprintf(stderr, "solve failed: %s\n", solution.error().format().c_str());
    return 1;
  }

  std::printf("status=%s\n", status_name(solution->status));
  std::printf("objective=%.10e\n", solution->objective);
  std::printf("iterations=%zu\n", solution->iterations);
  std::printf("primal_infeasibility=%.6e\n", solution->quality.primal_infeasibility);
  std::printf("dual_infeasibility=%.6e\n", solution->quality.dual_infeasibility);
  std::printf("relative_gap=%.6e\n", solution->quality.relative_gap);
  std::printf("complementarity=%.6e\n", solution->quality.complementarity);
  std::printf("max_bound_violation=%.6e\n", solution->quality.max_bound_violation);
  std::printf("from_best_iterate=%s\n", solution->from_best_iterate ? "true" : "false");
  std::printf("solve_time_seconds=%.6f\n", solution->solve_time_seconds);
  if (is_milp) {
    std::printf("nodes_explored=%zu\n", solution->nodes_explored);
    std::printf("best_bound=%.10e\n", solution->best_bound);
  }

  return solution->status == sovsolve::core::SolverStatus::Optimal ? 0 : 1;
}
