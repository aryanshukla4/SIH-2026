// mpsinfo -- load a model file and print the structure report.
//
// The manual smoke test: run it on an instance whose dimensions are known
// (Netlib afiro is 27 x 32 with 83 nonzeros) and check the numbers by hand.
// It also exercises the memory-mapped path, which the in-memory unit tests do
// not reach.

#include <chrono>
#include <cstdio>
#include <string>

#include "sovsolve/analysis/MatrixAnalysis.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"

namespace {

using sovsolve::core::INF;

const char* type_name(sovsolve::core::ProblemType t) {
  switch (t) {
    case sovsolve::core::ProblemType::LP: return "LP";
    case sovsolve::core::ProblemType::QP: return "QP";
    case sovsolve::core::ProblemType::MILP: return "MILP";
    case sovsolve::core::ProblemType::MIQP: return "MIQP";
  }
  return "?";
}

void print_distribution(const char* label,
                        const sovsolve::analysis::NnzDistribution& d) {
  std::printf("  %-9s min %-7zu max %-7zu mean %-8.2f empty %-6zu singleton %zu\n",
              label, d.min, d.max, d.mean, d.empty_count, d.singleton_count);
}

/// Count rows by the shape of their bounds. This is the view the canonicalizer
/// cares about, and it is not visible from the matrix alone.
void print_row_shapes(const sovsolve::model::Problem& p) {
  std::size_t eq = 0, le = 0, ge = 0, ranged = 0, free_rows = 0;
  for (std::size_t i = 0; i < p.num_rows(); ++i) {
    const bool lo = sovsolve::core::is_finite_bound(p.row_lower[i]);
    const bool hi = sovsolve::core::is_finite_bound(p.row_upper[i]);
    if (lo && hi) {
      if (p.row_lower[i] == p.row_upper[i]) ++eq;
      else ++ranged;
    } else if (hi) ++le;
    else if (lo) ++ge;
    else ++free_rows;
  }
  std::printf("  equality %zu   <= %zu   >= %zu   ranged %zu   free %zu\n",
              eq, le, ge, ranged, free_rows);
}

void print_col_shapes(const sovsolve::model::Problem& p) {
  std::size_t boxed = 0, lower_only = 0, upper_only = 0, free_cols = 0, fixed = 0;
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    const bool lo = sovsolve::core::is_finite_bound(p.col_lower[j]);
    const bool hi = sovsolve::core::is_finite_bound(p.col_upper[j]);
    if (lo && hi) {
      if (p.col_lower[j] == p.col_upper[j]) ++fixed;
      else ++boxed;
    } else if (lo) ++lower_only;
    else if (hi) ++upper_only;
    else ++free_cols;
  }
  std::printf("  boxed %zu   lower-only %zu   upper-only %zu   fixed %zu   free %zu\n",
              boxed, lower_only, upper_only, fixed, free_cols);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: mpsinfo <model file>\n");
    return 2;
  }

  const std::string path = argv[1];
  const auto t0 = std::chrono::steady_clock::now();
  auto result = sovsolve::io::loadProblem(path);
  const auto t1 = std::chrono::steady_clock::now();

  if (!result.has_value()) {
    std::fprintf(stderr, "error: %s\n", result.error().format().c_str());
    return 1;
  }
  const auto& p = result.value();

  const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::printf("%s\n", p.problem_name.empty() ? path.c_str()
                                             : p.problem_name.c_str());
  std::printf("  type %s   %s   objective row '%s'\n", type_name(p.type()),
              p.sense == sovsolve::core::ObjSense::Maximize ? "maximize"
                                                            : "minimize",
              p.objective_row_name.c_str());
  std::printf("  rows %zu   cols %zu   nnz %zu   discrete %zu\n", p.num_rows(),
              p.num_cols(), p.nnz(), p.num_discrete());
  if (p.obj_constant != 0.0) {
    std::printf("  objective constant %.10g\n", p.obj_constant);
  }

  std::printf("\nrows\n");
  print_row_shapes(p);
  std::printf("\ncolumns\n");
  print_col_shapes(p);

  const auto a = sovsolve::analysis::analyze(p.A);
  std::printf("\nstructure\n");
  std::printf("  density %.6g\n", a.density);
  print_distribution("per row", a.by_row);
  print_distribution("per col", a.by_column);
  std::printf("  |a| in [%.6g, %.6g]   dynamic range %.6g\n", a.values.min_abs,
              a.values.max_abs, a.values.dynamic_range);

  // The number the KKT builder needs: one dense column makes A*Theta*A' dense.
  std::printf("  dense columns %zu (threshold %zu)\n", a.dense_columns.size(),
              a.dense_column_threshold);
  std::printf("  empty rows %zu   empty cols %zu   singleton rows %zu   "
              "singleton cols %zu\n",
              a.empty_rows.size(), a.empty_columns.size(),
              a.singleton_rows.size(), a.singleton_columns.size());
  std::printf("  duplicate candidates: rows %zu   cols %zu\n",
              a.duplicate_row_candidates.size(),
              a.duplicate_column_candidates.size());
  if (a.has_invalid_values) {
    std::printf("  WARNING: matrix contains NaN or infinite entries\n");
  }

  std::printf("\nstorage\n");
  std::printf("  sparse %zu bytes   dense %zu bytes   prefer %s\n",
              a.recommendation.sparse_bytes, a.recommendation.dense_bytes,
              a.recommendation.sparse_preferred ? "sparse" : "dense");
  if (p.nnz() > 0) {
    const double per_nnz = static_cast<double>(a.recommendation.sparse_bytes) /
                           static_cast<double>(p.nnz());
    std::printf("  %.2f bytes/nnz (one orientation; both is ~2x)\n", per_nnz);
  }

  if (p.has_quadratic()) {
    std::printf("\nquadratic objective\n");
    const auto q = sovsolve::analysis::analyze(p.Q);
    std::printf("  nnz %zu   density %.6g   symmetric %s\n", q.nnz, q.density,
                q.numerically_symmetric ? "yes" : "NO");
  }

  std::printf("\nvalidate %s   load %.2f ms\n", p.validate() ? "ok" : "FAILED",
              ms);
  return p.validate() ? 0 : 1;
}
