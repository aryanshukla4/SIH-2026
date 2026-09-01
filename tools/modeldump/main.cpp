// Dump a model in a neutral form, so an EXTERNAL solver can check it.
//
// Everything else in this repository validates the ingestion layer against
// itself: the LP reader is checked against the MPS reader, the readers against
// the writer, the canonicalizer against a forward map written from the same
// specification. Those catch a great deal, but they share a blind spot -- if
// two of my own components hold the same misunderstanding, they agree and the
// test passes.
//
// This tool exists to break that. It writes both the parsed model and the
// canonicalized model in a form `scripts/oracle_check.py` can hand to HiGHS
// (through SciPy), which then answers two questions nothing here can:
//
//   1. Does the PARSED model have the published optimal objective? That is
//      external ground truth for the whole read path -- RANGES, BOUNDS, the
//      negated objective constant, integer markers, all of it at once.
//
//   2. Does the CANONICAL model have the SAME optimum as the parsed one? The
//      canonicalizer substitutes columns out, drops rows, negates rows and
//      permutes them. Every property test so far checks that a point maps
//      correctly; none checks that the OPTIMUM is preserved, because that needs
//      a solver.
//
// Output is line-oriented and exact: numbers go through `std::to_chars`, whose
// shortest round-trip form reads back bit-identical, so the oracle solves the
// same model this process holds rather than a rounded copy.

#include <array>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <system_error>
#include <vector>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"
#include "sovsolve/model/Canonical.hpp"

namespace {

using sovsolve::core::Index;
using sovsolve::core::INF;
using sovsolve::core::Real;

void emit(Real v) {
  std::array<char, 40> buf{};
  const auto r = std::to_chars(buf.data(), buf.data() + buf.size(), v);
  if (r.ec == std::errc{}) {
    std::fwrite(buf.data(), 1, static_cast<std::size_t>(r.ptr - buf.data()),
                stdout);
  } else {
    std::fputs("0", stdout);
  }
}

void emit_vector(const char* name, const sovsolve::core::RealVector& v) {
  std::printf("%s %zu", name, v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::fputc(' ', stdout);
    emit(v[i]);
  }
  std::fputc('\n', stdout);
}

void emit_csr(const char* prefix, const sovsolve::core::SparseMatrixPair<>& A) {
  const auto off = A.csr.offsets();
  const auto idx = A.csr.indices();
  const auto val = A.csr.values();

  std::printf("%s_offsets %zu", prefix, A.rows() + 1);
  for (std::size_t i = 0; i <= A.rows(); ++i) {
    std::printf(" %lld", static_cast<long long>(off[i]));
  }
  std::fputc('\n', stdout);

  std::printf("%s_indices %zu", prefix, A.nnz());
  for (std::size_t k = 0; k < A.nnz(); ++k) {
    std::printf(" %lld", static_cast<long long>(idx[k]));
  }
  std::fputc('\n', stdout);

  std::printf("%s_values %zu", prefix, A.nnz());
  for (std::size_t k = 0; k < A.nnz(); ++k) {
    std::fputc(' ', stdout);
    emit(val[k]);
  }
  std::fputc('\n', stdout);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: modeldump <model-file> [--up-neg-keeps-lower]\n");
    return 2;
  }

  // The MPS negative-UP quirk: `UP` with a negative value and no prior lower
  // bound. Readers disagree on whether the lower bound becomes -infinity (our
  // default, matching CPLEX/Gurobi/HiGHS) or stays at zero. The flag lets the
  // oracle test which convention reproduces a published optimum, rather than
  // leaving the choice resting on documentation alone.
  sovsolve::model::ReaderOptions opts;
  for (int i = 2; i < argc; ++i) {
    if (std::string(argv[i]) == "--up-neg-keeps-lower") {
      opts.negative_upper_implies_free_lower = false;
    }
  }

  auto loaded =
      sovsolve::io::loadProblem(argv[1], sovsolve::io::FileFormat::Auto, opts);
  if (!loaded.has_value()) {
    std::fprintf(stderr, "%s\n", loaded.error().format().c_str());
    return 1;
  }
  const auto& p = loaded.value();

  // -- the parsed model, exactly as the file described it -------------------
  std::printf("BEGIN original\n");
  std::printf("name %s\n", p.problem_name.c_str());
  std::printf("sense %s\n",
              p.sense == sovsolve::core::ObjSense::Maximize ? "max" : "min");
  std::printf("obj_constant ");
  emit(p.obj_constant);
  std::fputc('\n', stdout);
  std::printf("rows %zu\ncols %zu\n", p.num_rows(), p.num_cols());
  std::printf("discrete %zu\n", p.num_discrete());
  emit_vector("c", p.c);
  emit_vector("row_lower", p.row_lower);
  emit_vector("row_upper", p.row_upper);
  emit_vector("col_lower", p.col_lower);
  emit_vector("col_upper", p.col_upper);
  emit_csr("A", p.A);
  std::printf("END original\n");

  // -- the canonical model the solver would actually receive ----------------
  auto canon = sovsolve::model::canonicalize(p);
  if (!canon.has_value()) {
    // An infeasibility verdict is a legitimate outcome, not a failure to dump.
    std::printf("BEGIN canonical\nstatus %s\nEND canonical\n",
                canon.error().format().c_str());
    return 0;
  }
  const auto& cp = canon->problem;

  std::printf("BEGIN canonical\n");
  std::printf("status ok\n");
  std::printf("rows %zu\ncols %zu\n", cp.num_rows(), cp.num_cols());
  std::printf("num_equality %zu\n", cp.num_equality);
  std::printf("num_range %zu\n", cp.num_range);
  std::printf("objective_negated %d\n", cp.objective_negated ? 1 : 0);
  std::printf("obj_offset ");
  emit(cp.obj_offset);
  std::fputc('\n', stdout);
  emit_vector("c", cp.c);
  emit_vector("b", cp.b);
  emit_vector("col_lower", cp.col_lower);
  emit_vector("col_upper", cp.col_upper);
  emit_csr("A", cp.A);
  std::printf("END canonical\n");
  return 0;
}
