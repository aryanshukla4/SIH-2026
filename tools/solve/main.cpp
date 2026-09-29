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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "sovsolve/io/Load.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/solver/Certificate.hpp"
#include "sovsolve/solver/Iis.hpp"
#include "sovsolve/solver/LpSolve.hpp"
#include "sovsolve/solver/MilpSolve.hpp"
#ifdef SOVSOLVE_ENABLE_CUDA
#include "sovsolve/solver/gpu/BranchAndBound.hpp"
#include "sovsolve/solver/gpu/PdlpDevice.hpp"
#include "sovsolve/solver/gpu/PdlpMatVec.hpp"
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
      "  --method=ipm|dual-simplex|primal-simplex|pdlp|pdlpx|hsd|concurrent\n"
      "                        `concurrent` (Module 30) races dual simplex, cuPDLPx,\n"
      "                        primal simplex and HSD on separate cores and keeps\n"
      "                        the first verdict. SimplexOptions::method\n"
      "                        (default concurrent; ipm is the GPU interior point,\n"
      "                        the rest are host engines and run on every build)\n"
      "  --concurrent-threads=N  ConcurrentOptions::max_threads (0 = use the\n"
      "                        machine's own core count; the engine line-up is\n"
      "                        truncated to this)\n"
      "  --concurrent-gpu-ipm=0|1  race the GPU interior-point engine too\n"
      "                        (default 0 -- worth turning on for a datacentre\n"
      "                        card, where FP64 is not 1/64 rate)\n"
      "  --scaling=geometric|ruiz  ScalingOptions::mode (ruiz is implied by\n"
      "                        --method=pdlp; pass this AFTER it to override)\n"
      "  --pdlp-tol=X          PdlpOptions::termination_tolerance (default 1e-8)\n"
      "  --pdlp-max-iter=N     PdlpOptions::max_iterations   (0 = auto, 1000000)\n"
      "  --pdlp-check-interval=N  PdlpOptions::check_interval (default 40)\n"
      "  --pdlp-resident-check=0|1  termination check on the GPU (default 1)\n"
      "  --pdlp-certificate-every=N  infeasibility test every N-th check (default 10)\n"
      "  --pdlp-adaptive=0|1   PdlpOptions::adaptive_step_size (default 1)\n"
      "  --pdlp-restart=0|1    PdlpOptions::adaptive_restart   (default 1)\n"
      "  --pdlp-primal-weight=0|1  PdlpOptions::primal_weight_update (default 1)\n"
      "  Module 31 (--method=pdlpx), cuPDLPx arXiv 2507.14051:\n"
      "  --pdlp-original-termination=0|1  check (6a)-(6c) on the ORIGINAL LP, not\n"
      "                        the preconditioned one, as cuPDLPx does (default 1)\n"
      "  --pdlp-reflection=X   reflection gamma in [0,1]      (default 1)\n"
      "  --pdlp-step-fraction=X  eta = X/||A||_2              (default 0.998)\n"
      "  --pdlp-restart-sufficient=X  fixed-point decay, hard (default 0.2, HPR-LP)\n"
      "  --pdlp-restart-necessary=X   fixed-point decay, weak (default 0.6, HPR-LP)\n"
      "  --pdlp-restart-artificial=X  epoch length cap        (default 0.2, HPR-LP)\n"
      "  --pdlp-restart-check-every=N test restarts every N iterations (default 1;\n"
      "                        HPR-LP uses 150)\n"
      "  --pdlp-weight-rule=hpr|pid  primal weight: HPR-LP Alg. 3 (default) or PID\n"
      "  --pdlp-pid=Kp,Ki,Kd   PID coefficients, with --pdlp-weight-rule=pid\n"
      "  --gpu-resident=0|1    --method=pdlp with the WHOLE iterate on the GPU\n"
      "                        (CUDA builds only). The fast path: per trial only\n"
      "                        24 bytes cross the bus. Use this one.\n"
      "  --gpu-graphs=0|1      with --gpu-resident and --method=pdlpx, replay each\n"
      "                        check interval as ONE captured CUDA graph (default 1)\n"
      "  --gpu-spmv=0|1        --method=pdlp with only K and K' on the GPU (CUDA\n"
      "                        builds only). Kept for comparison -- it copies\n"
      "                        vectors across the bus on every product.\n"
      "  --gpu-spmv-timing=0|1 with --gpu-spmv, attribute time to kernel vs\n"
      "                        transfer. Adds two syncs per product, so it\n"
      "                        INFLATES the wall time it reports on.\n"
      "  --iis[=0|1]           on an Infeasible verdict, print an irreducible\n"
      "                        infeasible subsystem: the rows and bounds that\n"
      "                        conflict (Gleeson & Ryan 1990)\n"
      "  --hsd-max-iter=N      HsdOptions::max_iterations    (default 200)\n"
      "  --hsd-cg-max-iter=N   HsdOptions::cg_max_iterations (default 5000)\n"
      "  --hsd-cg-tol=X        HsdOptions::cg_tolerance      (default 1e-10)\n"
      "  --hsd-direct=0|1      HsdOptions::direct: factor the normal equations and\n"
      "                        precondition CG with it (default 1)\n"
      "  --hsd-delta-d=X       HsdOptions::delta_d, dual regularization (default 1.49e-8)\n"
      "  --ipm-mehrotra-start=0|1  Mehrotra (1992) starting point (default 0)\n"
      "  --ipm-direct=0|1|2    GPU CG preconditioner: 0 IC(0)/Jacobi, 1 cuDSS\n"
      "                        Cholesky, 2 in-house SparseLdl on the host (default)\n"
      "  --simplex-max-iter=N  SimplexOptions::max_iterations   (0 = auto)\n"
      "  --pivot-tolerance=X   SimplexOptions::pivot_tolerance  (default 0.1)\n"
      "  --pivot-floor=X       SimplexOptions::pivot_floor      (default 1e-9)\n"
      "  --refactor-interval=N SimplexOptions::refactor_interval (default 100)\n"
      "  --artificial-bound=X  SimplexOptions::artificial_bound  (default 1e7)\n"
      "  --primal-cleanup=0|1  SimplexOptions::primal_cleanup    (default 1 --\n"
      "                        finish a dual run that ended without a verdict\n"
      "                        by handing its basis to the primal simplex)\n"
      "  --bound-flipping=0|1  SimplexOptions::bound_flipping    (default 1)\n"
      "  --dse=0|1             SimplexOptions::dual_steepest_edge (default 1):\n"
      "                        Koberstein 3.3; 0 = Dantzig pricing\n"
      "  --perturb=0|1         SimplexOptions::cost_perturbation (default 1):\n"
      "                        Koberstein 6.3.1, against dual degeneracy\n"
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
      "  --presolve=0|1|2      0 off, 1 Module 4 only, 2 LP presolve  (default 2)\n"
      "  --node-selection=best-first|interleaved\n"
      "                        MilpOptions::node_selection (default interleaved):\n"
      "                        CIP ch. 6 best estimate + plunging, best-bound\n"
      "                        every 10th plunge\n"
      "  --branching=most-fractional|pseudocost|reliability\n"
      "                        MilpOptions::branching (default reliability). A\n"
      "                        MILP with --method=dual-simplex or primal-simplex\n"
      "                        runs the host branch-and-bound (docs/spec/module.txt 28).\n"
      "  --mip-conflicts=0|1   MilpOptions::conflict_analysis (default 1): CIP ch. 11\n"
      "  (OURS -- chosen here, not taken from a paper; tune freely:)\n"
      "  --mip-conflict-age=N  MilpOptions::conflict_max_age (default 1000)\n"
      "  --mip-dive-allowance=N  MilpOptions::dive_allowance (default 1000)\n"
      "  --mip-prop-row-visits=N  MilpOptions::propagation_row_visits (default 20)\n"
      "  --mip-cut-margin=X    MilpOptions::cut_violation_margin (default 1e-6)\n"
      "  --mip-presolve=0|1    MilpOptions::presolve (default 1): CIP ch. 10\n"
      "  --mip-presolve-rounds=N  MilpOptions::presolve_rounds (default 20, OURS)\n"
      "  --mip-presolve-columns=0|1  MilpOptions::presolve_columns (default 1):\n"
      "                        the reductions that REMOVE a column (AGH 4.5, 6.3)\n"
      "  --mip-gomory=0|1      MilpOptions::gomory_cuts (default 1)\n"
      "  --mip-cmir=0|1        MilpOptions::cmir_cuts (default 1)\n"
      "  --mip-cut-rounds=N    MilpOptions::cut_rounds (default 15)\n"
      "  --mip-propagation=0|1 MilpOptions::propagation (default 1): CIP ch. 7\n"
      "  --mip-pump=0|1        MilpOptions::feasibility_pump (default 1)\n"
      "  --mip-rens=0|1        MilpOptions::rens (default 1)\n"
      "  --mip-heuristics=0|1  MilpOptions::heuristics (default 1): CIP ch. 9\n"
      "                        simple rounding and diving\n"
      "  --mip-cuts=0|1        MilpOptions::root_cuts (default 1): root cover/GCD\n"
      "                        cuts before the host branch-and-bound\n"
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
/// Not a field on `model::Options`, deliberately: which BACKEND applies `K` is
/// a property of this build and this invocation, not of the model or the
/// algorithm. `Options` describes the problem and the method; putting a CUDA
/// switch in it would push a GPU concept down into the `model` layer, which
/// sits below `solver` and must not know about it.
bool gpu_spmv = false;
bool gpu_spmv_timing = false;
/// The whole PDLP iterate on the device (docs/spec/module.txt 24F), not just `K`.
bool gpu_resident = false;
/// Module 31: CUDA graphs over the resident Halpern loop. A flag so the gain
/// is measured against the identical launch sequence issued directly.
bool gpu_graphs = true;
/// On an Infeasible verdict, also compute and print an IIS (solver/Iis.hpp):
/// WHICH rows and bounds contradict each other, not just that some do.
bool want_iis = false;

bool apply_flag(const std::string& flag, sovsolve::model::Options& options) {
  const auto eq = flag.find('=');
  if (flag.rfind("--", 0) != 0 || eq == std::string::npos) return false;
  const std::string key = flag.substr(2, eq - 2);
  const std::string val = flag.substr(eq + 1);

  try {
    if (key == "scaling") {
      if (val == "geometric") {
        options.scaling.mode = sovsolve::model::ScalingMode::GeometricMean;
      } else if (val == "ruiz") {
        options.scaling.mode = sovsolve::model::ScalingMode::RuizPockChambolle;
      } else {
        std::fprintf(stderr, "unknown --scaling: %s (geometric or ruiz)\n", val.c_str());
        return false;
      }
      return true;
    }
    if (key == "pdlp-tol") {
      options.pdlp.termination_tolerance = std::stod(val);
      return true;
    }
    if (key == "pdlp-max-iter") {
      options.pdlp.max_iterations = static_cast<std::size_t>(std::stoull(val));
      return true;
    }
    if (key == "pdlp-cert-tol") {
      options.pdlp.certificate_tolerance = std::stod(val);
      return true;
    }
    if (key == "pdlp-infeasibility") {
      options.pdlp.infeasibility_detection = (val != "0");
      return true;
    }
    if (key == "pdlp-primal-weight") {
      options.pdlp.primal_weight_update = (val != "0");
      return true;
    }
    if (key == "pdlp-restart") {
      options.pdlp.adaptive_restart = (val != "0");
      return true;
    }
    if (key == "pdlp-adaptive") {
      options.pdlp.adaptive_step_size = (val != "0");
      return true;
    }
    if (key == "pdlp-original-termination") {
      options.pdlp.terminate_on_original = (val != "0");
      return true;
    }
    if (key == "pdlp-reflection") {
      options.pdlp.reflection = std::stod(val);
      return true;
    }
    if (key == "pdlp-step-fraction") {
      options.pdlp.halpern_step_fraction = std::stod(val);
      return true;
    }
    if (key == "pdlp-restart-sufficient") {
      options.pdlp.halpern_restart_sufficient = std::stod(val);
      return true;
    }
    if (key == "pdlp-restart-necessary") {
      options.pdlp.halpern_restart_necessary = std::stod(val);
      return true;
    }
    if (key == "pdlp-restart-artificial") {
      options.pdlp.halpern_restart_artificial = std::stod(val);
      return true;
    }
    if (key == "pdlp-weight-rule") {
      if (val == "hpr" || val == "hpr-lp") {
        options.pdlp.weight_rule = sovsolve::model::PdlpOptions::WeightRule::HprLp;
      } else if (val == "pid") {
        options.pdlp.weight_rule = sovsolve::model::PdlpOptions::WeightRule::Pid;
      } else {
        std::fprintf(stderr, "--pdlp-weight-rule must be hpr or pid\n");
        return false;
      }
      return true;
    }
    if (key == "pdlp-restart-check-every") {
      options.pdlp.halpern_restart_check_every =
          static_cast<std::size_t>(std::stoull(val));
      return true;
    }
    if (key == "pdlp-pid") {
      // Three coefficients in one flag because they are tuned together: the
      // integral and derivative terms are corrections on the proportional
      // one, and a sweep that moves one without the others is measuring the
      // wrong thing.
      const std::size_t first = val.find(',');
      const std::size_t second = first == std::string::npos
                                     ? std::string::npos
                                     : val.find(',', first + 1);
      if (second == std::string::npos) {
        std::fprintf(stderr, "--pdlp-pid needs three values: Kp,Ki,Kd\n");
        return false;
      }
      options.pdlp.pid_kp = std::stod(val.substr(0, first));
      options.pdlp.pid_ki = std::stod(val.substr(first + 1, second - first - 1));
      options.pdlp.pid_kd = std::stod(val.substr(second + 1));
      return true;
    }
    if (key == "pdlp-check-interval") {
      options.pdlp.check_interval = static_cast<std::size_t>(std::stoull(val));
      return true;
    }
    if (key == "pdlp-resident-check") {
      options.pdlp.resident_check = val != "0";
      return true;
    }
    if (key == "pdlp-certificate-every") {
      options.pdlp.certificate_check_every = static_cast<std::size_t>(std::stoull(val));
      return true;
    }
    if (key == "method") {
      if (val == "ipm" || val == "interior-point") {
        options.simplex.method = sovsolve::model::Method::InteriorPoint;
      } else if (val == "dual-simplex" || val == "simplex") {
        options.simplex.method = sovsolve::model::Method::DualSimplex;
      } else if (val == "primal-simplex" || val == "primal") {
        options.simplex.method = sovsolve::model::Method::PrimalSimplex;
      } else if (val == "pdlp") {
        options.simplex.method = sovsolve::model::Method::Pdlp;
        // PDLP's convergence depends on the scaling far more directly than a
        // factorization-based method's does, so selecting it also selects the
        // preconditioning its paper specifies -- overridable with an explicit
        // later --scaling=.
        options.scaling.mode = sovsolve::model::ScalingMode::RuizPockChambolle;
      } else if (val == "pdlpx" || val == "pdlp-x") {
        // Module 31: the same engine, cuPDLPx's reflected-Halpern scheme.
        options.simplex.method = sovsolve::model::Method::PdlpX;
        options.scaling.mode = sovsolve::model::ScalingMode::RuizPockChambolle;
      } else if (val == "hsd" || val == "homogeneous") {
        options.simplex.method = sovsolve::model::Method::Hsd;
      } else if (val == "concurrent") {
        options.simplex.method = sovsolve::model::Method::Concurrent;
        // The race includes cuPDLPx, which wants its paper's preconditioning
        // (see above). The other entrants are indifferent to it.
        options.scaling.mode = sovsolve::model::ScalingMode::RuizPockChambolle;
      } else {
        std::fprintf(
            stderr,
            "unknown --method: %s (ipm, dual-simplex, primal-simplex, pdlp, "
            "pdlpx, hsd, concurrent)\n",
            val.c_str());
        return false;
      }
    } else if (key == "iis") {
      want_iis = (val != "0");
    } else if (key == "gpu-spmv") {
      gpu_spmv = (val != "0");
    } else if (key == "gpu-resident") {
      gpu_resident = (val != "0");
    } else if (key == "gpu-graphs") {
      gpu_graphs = (val != "0");
    } else if (key == "gpu-spmv-timing") {
      gpu_spmv_timing = (val != "0");
    } else if (key == "dse") {
      options.simplex.dual_steepest_edge = (val != "0");
    } else if (key == "perturb") {
      options.simplex.cost_perturbation = (val != "0");
    } else if (key == "mip-prop-row-visits") {
      options.milp.propagation_row_visits = static_cast<std::size_t>(std::stoul(val));
    } else if (key == "mip-cut-margin") {
      options.milp.cut_violation_margin = std::stod(val);
    } else if (key == "mip-conflict-age") {
      options.milp.conflict_max_age = static_cast<std::size_t>(std::stoul(val));
    } else if (key == "mip-dive-allowance") {
      options.milp.dive_allowance = static_cast<std::size_t>(std::stoul(val));
    } else if (key == "mip-conflicts") {
      options.milp.conflict_analysis = (val != "0");
    } else if (key == "mip-presolve") {
      options.milp.presolve = (val != "0");
    } else if (key == "mip-presolve-rounds") {
      options.milp.presolve_rounds = static_cast<std::size_t>(std::stoul(val));
    } else if (key == "concurrent-threads") {
      options.concurrent.max_threads = static_cast<std::size_t>(std::stoul(val));
    } else if (key == "concurrent-gpu-ipm") {
      options.concurrent.include_gpu_interior_point = (val != "0");
    } else if (key == "mip-presolve-columns") {
      options.milp.presolve_columns = (val != "0");
    } else if (key == "mip-gomory") {
      options.milp.gomory_cuts = (val != "0");
    } else if (key == "mip-cmir") {
      options.milp.cmir_cuts = (val != "0");
    } else if (key == "mip-cut-rounds") {
      options.milp.cut_rounds = static_cast<std::size_t>(std::stoul(val));
    } else if (key == "mip-propagation") {
      options.milp.propagation = (val != "0");
    } else if (key == "mip-pump") {
      options.milp.feasibility_pump = (val != "0");
    } else if (key == "mip-rens") {
      options.milp.rens = (val != "0");
    } else if (key == "mip-heuristics") {
      options.milp.heuristics = (val != "0");
    } else if (key == "mip-cuts") {
      options.milp.root_cuts = (val != "0");
    } else if (key == "node-selection") {
      if (val == "best-first") {
        options.milp.node_selection = sovsolve::model::NodeSelection::BestFirst;
      } else if (val == "interleaved") {
        options.milp.node_selection = sovsolve::model::NodeSelection::Interleaved;
      } else {
        std::fprintf(stderr, "unknown --node-selection: %s (best-first, interleaved)\n",
                     val.c_str());
        return false;
      }
    } else if (key == "branching") {
      if (val == "most-fractional") {
        options.milp.branching = sovsolve::model::BranchingRule::MostFractional;
      } else if (val == "pseudocost") {
        options.milp.branching = sovsolve::model::BranchingRule::Pseudocost;
      } else if (val == "reliability") {
        options.milp.branching = sovsolve::model::BranchingRule::Reliability;
      } else {
        std::fprintf(stderr,
                     "unknown --branching: %s (most-fractional, pseudocost, reliability)\n",
                     val.c_str());
        return false;
      }
    } else if (key == "hsd-max-iter") {
      options.hsd.max_iterations = static_cast<std::size_t>(std::stoul(val));
    } else if (key == "hsd-cg-max-iter") {
      options.hsd.cg_max_iterations = static_cast<std::size_t>(std::stoul(val));
    } else if (key == "hsd-cg-tol") {
      options.hsd.cg_tolerance = std::stod(val);
    } else if (key == "ipm-mehrotra-start") {
      options.ipm.mehrotra_start = val != "0";
    } else if (key == "ipm-direct") {
      options.ipm.direct = std::stoi(val);
    } else if (key == "hsd-direct") {
      options.hsd.direct = val != "0";
    } else if (key == "hsd-delta-d") {
      options.hsd.delta_d = std::stod(val);
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
      const int level = std::stoi(val);
      options.presolve.enabled = level != 0;
      options.presolve.lp_reductions = level >= 2;
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
    if (std::string(argv[next_arg]) == "--iis") {
      want_iis = true;
      continue;
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
  const bool use_host_engine =
      options.simplex.method != sovsolve::model::Method::InteriorPoint;

  // The simplex and PDLP engines are host-only (src/solver/CMakeLists.txt), so
  // this tool
  // now builds and runs without the CUDA toolkit -- it previously could not be
  // built at all outside WSL2, which left every stage downstream of the
  // canonicalizer unreachable from the default `release` preset.
  // Owned HERE, not inside solve_lp, so its counters survive the call and can
  // be printed below. solve_lp takes a provider precisely to allow this.
  sovsolve::solver::PdlpBackendProvider provider;
#ifdef SOVSOLVE_ENABLE_CUDA
  std::unique_ptr<sovsolve::solver::gpu::CusparseMatVec> gpu_matvec;
  std::unique_ptr<sovsolve::solver::gpu::DevicePdlpBackend> gpu_backend;
  const bool is_pdlp = options.simplex.method == sovsolve::model::Method::Pdlp ||
                       options.simplex.method == sovsolve::model::Method::PdlpX;
  if (gpu_resident && is_pdlp) {
    provider = [&gpu_backend](const sovsolve::model::CanonicalProblem& canonical)
        -> sovsolve::core::Expected<sovsolve::solver::PdlpBackends> {
      auto created = sovsolve::solver::gpu::DevicePdlpBackend::create(canonical);
      if (!created.has_value()) return created.error();
      gpu_backend = std::move(created.value());
      gpu_backend->set_use_graphs(gpu_graphs);
      return sovsolve::solver::PdlpBackends{&gpu_backend->cold_matvec(),
                                             gpu_backend.get()};
    };
  } else if (gpu_spmv && is_pdlp) {
    provider = [&gpu_matvec](const sovsolve::model::CanonicalProblem& canonical)
        -> sovsolve::core::Expected<sovsolve::solver::PdlpBackends> {
      auto created =
          sovsolve::solver::gpu::CusparseMatVec::create(canonical, gpu_spmv_timing);
      if (!created.has_value()) return created.error();
      gpu_matvec = std::move(created.value());
      return sovsolve::solver::PdlpBackends{gpu_matvec.get(), nullptr};
    };
  }
#else
  if (gpu_spmv || gpu_resident) {
    std::fprintf(stderr, "--gpu-spmv/--gpu-resident need a CUDA build; ignoring\n");
  }
#endif

  // A MILP on a simplex method goes to Module 28's branch-and-bound. Before it
  // existed this called solve_lp, which ignores integrality -- so a MILP asked
  // for with --method=dual-simplex silently got its LP RELAXATION back.
  const auto method = options.simplex.method;
  const bool simplex_method = method == sovsolve::model::Method::DualSimplex ||
                              method == sovsolve::model::Method::PrimalSimplex;
  const bool host_milp = is_milp && simplex_method;
  if (is_milp && use_host_engine && !simplex_method) {
    std::fprintf(stderr,
                 "note: this --method ignores integrality; solving the LP RELAXATION. "
                 "Use --method=dual-simplex for branch-and-bound.\n");
  }
  sovsolve::solver::MilpStatistics milp_stats;

  auto solution = host_milp
                      ? sovsolve::solver::solve_milp(*problem, options, &milp_stats)
                  : use_host_engine
                      ? sovsolve::solver::solve_lp(*problem, options, provider)
#ifdef SOVSOLVE_ENABLE_CUDA
                      : sovsolve::solver::gpu::solve(*problem, options);
#else
                      : sovsolve::core::Expected<sovsolve::model::Solution>(
                            sovsolve::core::make_error(
                                sovsolve::core::ErrorCode::NotImplemented,
                                "this build has no CUDA, so the interior-point path is "
                                "absent; pass --method=dual-simplex, "
                                "--method=pdlp or --method=hsd"));
#endif
  if (!solution.has_value()) {
    std::fprintf(stderr, "solve failed: %s\n", solution.error().format().c_str());
    return 1;
  }

  std::printf("status=%s\n", status_name(solution->status));
  // An Infeasible or Unbounded verdict has no objective value: whatever the
  // engine left in `objective` is the last iterate's, which is meaningless.
  // A TimeLimit/MaxIterations run keeps its number -- a real best-so-far.
  const bool no_objective =
      solution->status == sovsolve::core::SolverStatus::Infeasible ||
      solution->status == sovsolve::core::SolverStatus::Unbounded;
  if (no_objective) {
    std::printf("objective=nan\n");
  } else {
    std::printf("objective=%.10e\n", solution->objective);
  }
  std::printf("iterations=%zu\n", solution->iterations);
  if (solution->matrix_products != 0) {
    // The cost model for a matrix-free method. One "KKT pass" in PDLP's sense
    // is one K plus one K', so half the product count -- printed because the
    // iteration count alone understates the adaptive step size's real cost.
    std::printf("matrix_products=%zu\n", solution->matrix_products);
    std::printf("kkt_passes=%.1f\n",
                static_cast<double>(solution->matrix_products) / 2.0);
  }
  std::printf("primal_infeasibility=%.6e\n", solution->quality.primal_infeasibility);
  std::printf("dual_infeasibility=%.6e\n", solution->quality.dual_infeasibility);
  std::printf("relative_gap=%.6e\n", solution->quality.relative_gap);
  std::printf("complementarity=%.6e\n", solution->quality.complementarity);
  std::printf("max_bound_violation=%.6e\n", solution->quality.max_bound_violation);
  std::printf("from_best_iterate=%s\n", solution->from_best_iterate ? "true" : "false");
  // Measured HERE, on the model as read, from the reported x alone -- not the
  // engine's own figures, which describe its presolved, scaled working model.
  // Worst row violation relative to 1 + |row bound|.
  if (solution->x.size() == problem->num_cols()) {
    double worst = 0.0;
    const auto& csr = problem->A.csr;
    for (std::size_t i = 0; i < problem->num_rows(); ++i) {
      double act = 0.0;
      for (auto k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
        act += csr.values()[k] * solution->x[static_cast<std::size_t>(csr.indices()[k])];
      }
      const double lo = problem->row_lower[i];
      const double hi = problem->row_upper[i];
      if (sovsolve::core::is_finite_bound(lo)) worst = std::max(worst, (lo - act) / (1.0 + std::fabs(lo)));
      if (sovsolve::core::is_finite_bound(hi)) worst = std::max(worst, (act - hi) / (1.0 + std::fabs(hi)));
    }
    std::printf("original_primal_residual=%.6e\n", worst);
  }
  // The certificate on the model as read (Certificate.hpp): the weak-duality
  // bound from the reported y, and per-column dual residual -- the same
  // measure the GPU interior point must pass before it may report Optimal.
  {
    const auto cert = sovsolve::solver::certify(*problem, *solution);
    if (cert.valid && !problem->has_quadratic()) {
      std::printf("original_dual_bound=%.10e\n", cert.dual_bound);
      std::printf("original_dual_residual=%.6e\n", cert.dual);
      std::printf("original_gap=%.6e\n", cert.gap);
    }
  }
#ifdef SOVSOLVE_ENABLE_CUDA
  if (gpu_backend) {
    std::printf("gpu_resident=1\n");
    std::printf("gpu_device_bytes=%zu\n", gpu_backend->device_bytes());
    // Whether graphs actually ENGAGED, not whether they were asked for: a
    // capture that failed falls back silently to direct launches, and a
    // benchmark row must not credit graphs with a run that never used one.
    std::printf("gpu_graphs=%d\n", gpu_backend->graphs_active() ? 1 : 0);
    std::printf("gpu_graph_launches=%zu\n", gpu_backend->graph_launches());
    if (!gpu_backend->graph_note().empty()) {
      std::printf("gpu_graph_note=%s\n", gpu_backend->graph_note().c_str());
    }
  }
  if (gpu_matvec) {
    // Kernel and transfer are reported SEPARATELY on purpose. The roofline
    // argument in PdlpMatVec.hpp predicts the KERNEL time; the copies are an
    // artefact of entering through a host-span interface. One combined number
    // would let a good kernel hide behind bad transfers, and the ratio between
    // them is exactly what says whether a device-resident iterate is the next
    // thing worth building.
    if (gpu_spmv_timing) {
      std::printf("spmv_kernel_seconds=%.6f\n", gpu_matvec->kernel_seconds());
      std::printf("spmv_transfer_seconds=%.6f\n", gpu_matvec->transfer_seconds());
    }
    std::printf("gpu_device_bytes=%zu\n", gpu_matvec->device_bytes());
  }
#endif
  std::printf("solve_time_seconds=%.6f\n", solution->solve_time_seconds);
  // Where that time went. Printed unconditionally so the benchmark CSV always
  // carries the split -- a stage fraction is only useful if it was recorded on
  // the same run as the total.
  std::printf("canonicalize_seconds=%.6f\n", solution->canonicalize_seconds);
  std::printf("lp_presolve_seconds=%.6f\n", solution->presolve_seconds);
  if (solution->presolved_rows + solution->presolved_cols > 0) {
    std::printf("presolved_rows=%zu\n", solution->presolved_rows);
    std::printf("presolved_cols=%zu\n", solution->presolved_cols);
    std::printf("presolved_nnz=%zu\n", solution->presolved_nnz);
  }
  std::printf("scale_seconds=%.6f\n", solution->scale_seconds);
  std::printf("engine_seconds=%.6f\n", solution->engine_seconds);
  if (is_milp) {
    std::printf("nodes_explored=%zu\n", solution->nodes_explored);
    std::printf("best_bound=%.10e\n", solution->best_bound);
    if (host_milp) {
      // The branching rules trade nodes against work per node, so nodes alone
      // would flatter strong branching. These make the trade visible.
      std::printf("node_lp_iterations=%zu\n", milp_stats.node_lp_iterations);
      std::printf("strong_branching_probes=%zu\n", milp_stats.strong_branching_probes);
      std::printf("strong_branching_iterations=%zu\n",
                  milp_stats.strong_branching_iterations);
      std::printf("strong_branching_fixings=%zu\n", milp_stats.strong_branching_fixings);
      std::printf("unreliable_nodes=%zu\n", milp_stats.unreliable_nodes);
      std::printf("root_cuts=%zu\n", milp_stats.root_cuts);
      std::printf("incumbents=%zu\n", milp_stats.incumbents);
      std::printf("first_incumbent_node=%zu\n", milp_stats.first_incumbent_node);
      std::printf("plunge_steps=%zu\n", milp_stats.plunge_steps);
      std::printf("dives=%zu\n", milp_stats.dives);
      std::printf("dive_lp_iterations=%zu\n", milp_stats.dive_lp_iterations);
      std::printf("dive_solutions=%zu\n", milp_stats.dive_solutions);
      std::printf("rounding_solutions=%zu\n", milp_stats.rounding_solutions);
      std::printf("pump_rounds=%zu\n", milp_stats.pump_rounds);
      std::printf("pump_lp_iterations=%zu\n", milp_stats.pump_lp_iterations);
      std::printf("pump_solutions=%zu\n", milp_stats.pump_solutions);
      std::printf("rens_nodes=%zu\n", milp_stats.rens_nodes);
      std::printf("rens_solutions=%zu\n", milp_stats.rens_solutions);
      std::printf("propagation_tightenings=%zu\n", milp_stats.propagation_tightenings);
      std::printf("propagation_cutoffs=%zu\n", milp_stats.propagation_cutoffs);
      std::printf("redcost_tightenings=%zu\n", milp_stats.redcost_tightenings);
      std::printf("local_redcost_tightenings=%zu\n", milp_stats.local_redcost_tightenings);
      std::printf("presolve_rounds=%zu\n", milp_stats.presolve_rounds);
      std::printf("presolve_bounds=%zu\n", milp_stats.presolve_bounds);
      std::printf("presolve_coefficients=%zu\n", milp_stats.presolve_coefficients);
      std::printf("presolve_rows_removed=%zu\n", milp_stats.presolve_rows_removed);
      std::printf("presolve_fixed=%zu\n", milp_stats.presolve_fixed);
      std::printf("presolve_substituted=%zu\n", milp_stats.presolve_substituted);
      std::printf("presolve_merged=%zu\n", milp_stats.presolve_merged);
      std::printf("presolve_seconds=%.6f\n", milp_stats.presolve_seconds);
      std::printf("cut_rounds=%zu\n", milp_stats.cut_rounds);
      std::printf("gomory_cuts=%zu\n", milp_stats.gomory_cuts);
      std::printf("cmir_cuts=%zu\n", milp_stats.cmir_cuts);
      std::printf("cuts_added=%zu\n", milp_stats.cuts_added);
      std::printf("root_bound_before_cuts=%.10e\n", milp_stats.root_bound_before_cuts);
      std::printf("root_bound_after_cuts=%.10e\n", milp_stats.root_bound_after_cuts);
      std::printf("conflicts_analyzed=%zu\n", milp_stats.conflicts_analyzed);
      std::printf("conflict_constraints=%zu\n", milp_stats.conflict_constraints);
      std::printf("conflict_deductions=%zu\n", milp_stats.conflict_deductions);
      std::printf("conflict_cutoffs=%zu\n", milp_stats.conflict_cutoffs);
      std::printf("conflict_checks=%zu\n", milp_stats.conflict_checks);
    }
  }

  if (want_iis && solution->status == sovsolve::core::SolverStatus::Infeasible) {
    // Recomputed from the ORIGINAL problem, whatever engine gave the verdict:
    // compute_iis re-solves for its own certificate, so indices below are
    // rows and columns of the model file as written.
    auto iis = sovsolve::solver::compute_iis(*problem, options);
    if (!iis.has_value()) {
      std::fprintf(stderr, "iis failed: %s\n", iis.error().format().c_str());
    } else {
      const auto row_name = [&](std::size_t i) {
        return i < problem->row_names.size() ? std::string(problem->row_names[i])
                                             : "R" + std::to_string(i);
      };
      const auto col_name = [&](std::size_t j) {
        return j < problem->col_names.size() ? std::string(problem->col_names[j])
                                             : "C" + std::to_string(j);
      };
      std::printf("iis_quality=%s\n",
                  iis->quality == sovsolve::solver::IisQuality::Irreducible
                      ? "Irreducible"
                      : "InfeasibleSubsystem");
      std::printf("iis_size=%zu\n", iis->size());
      for (const std::size_t i : iis->rows) {
        std::printf("iis_row=%s\n", row_name(i).c_str());
      }
      for (const std::size_t j : iis->lower_bound_columns) {
        std::printf("iis_lower_bound=%s\n", col_name(j).c_str());
      }
      for (const std::size_t j : iis->upper_bound_columns) {
        std::printf("iis_upper_bound=%s\n", col_name(j).c_str());
      }
    }
  }

  return solution->status == sovsolve::core::SolverStatus::Optimal ? 0 : 1;
}
