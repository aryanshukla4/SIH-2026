#include "sovsolve/solver/Diagnostics.hpp"

#include <sstream>

namespace sovsolve::solver {

namespace {
const char* const kCsvHeader =
    "iteration,objective,primal_residual_inf,dual_residual_inf,"
    "aggregate_complementarity,mu,mu_aff,sigma,alpha_primal,alpha_dual,"
    "factorization_time,solve_time,refinement_time,regularization_events,"
    "theta_floor_activations,refinement_passes,iteration_time,status";
}  // namespace

std::string Diagnostics::to_csv() const {
  std::ostringstream out;
  out << kCsvHeader << '\n';
  for (const auto& r : history_) {
    out << r.iteration << ',' << r.objective << ',' << r.primal_residual_inf << ','
        << r.dual_residual_inf << ',' << r.aggregate_complementarity << ',' << r.mu
        << ',' << r.mu_aff << ',' << r.sigma << ',' << r.alpha_primal << ','
        << r.alpha_dual << ',' << r.factorization_time << ',' << r.solve_time << ','
        << r.refinement_time << ',' << r.regularization_events << ','
        << r.theta_floor_activations << ',' << r.refinement_passes << ','
        << r.iteration_time << ',' << static_cast<int>(r.status) << '\n';
  }
  return out.str();
}

std::string Diagnostics::to_json() const {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < history_.size(); ++i) {
    const auto& r = history_[i];
    if (i > 0) out << ",";
    out << "{\"iteration\":" << r.iteration << ",\"objective\":" << r.objective
        << ",\"primal_residual_inf\":" << r.primal_residual_inf
        << ",\"dual_residual_inf\":" << r.dual_residual_inf << ",\"mu\":" << r.mu
        << ",\"sigma\":" << r.sigma << ",\"alpha_primal\":" << r.alpha_primal
        << ",\"alpha_dual\":" << r.alpha_dual
        << ",\"theta_floor_activations\":" << r.theta_floor_activations
        << ",\"refinement_passes\":" << r.refinement_passes
        << ",\"status\":" << static_cast<int>(r.status) << "}";
  }
  out << "]";
  return out.str();
}

}  // namespace sovsolve::solver
