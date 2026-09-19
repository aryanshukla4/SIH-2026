#include "sovsolve/solver/MilpCuts.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "sovsolve/core/SparseBuilder.hpp"
#include "sovsolve/core/Types.hpp"

namespace sovsolve::solver {

namespace {

using core::Real;
using core::RealVector;
using core::SolverStatus;
using core::VarType;
using model::Problem;

/// One knapsack cover inequality: `sum_{(col,_) in entries} x_col <= bound`.
struct CoverCut {
  std::vector<std::pair<core::Index, Real>> entries;  // (column, coefficient == 1.0)
  Real bound;
};

/// Rebuilds `problem.A`/`row_lower`/`row_upper` with `cuts` appended as new
/// rows. `SparseMatrixPair` has no incremental "add a row" -- it is fixed-
/// shape once built (SparseBuilder.hpp's whole design is exact two-pass
/// allocation) -- so this is a full O(nnz) rebuild. Acceptable here because
/// it runs a handful of times total, at the ROOT only (see the cutting loop
/// in `solve()` below), never per node.
[[nodiscard]] core::Status append_cut_rows(Problem& problem, const std::vector<CoverCut>& cuts) {
  if (cuts.empty()) return core::Status::Ok();

  const std::size_t old_m = problem.num_rows();
  const std::size_t n = problem.num_cols();
  const std::size_t new_m = old_m + cuts.size();
  const auto& old_csr = problem.A.csr;

  core::SparseBuilder builder(new_m, n);
  for (std::size_t i = 0; i < old_m; ++i) {
    for (auto k = old_csr.slice_begin(i); k < old_csr.slice_end(i); ++k) {
      builder.count(static_cast<core::Index>(i), old_csr.indices()[k]);
    }
  }
  for (std::size_t r = 0; r < cuts.size(); ++r) {
    for (const auto& [col, coeff] : cuts[r].entries) {
      builder.count(static_cast<core::Index>(old_m + r), col);
    }
  }
  if (const auto st = builder.allocate(); !st.ok()) return st;
  for (std::size_t i = 0; i < old_m; ++i) {
    for (auto k = old_csr.slice_begin(i); k < old_csr.slice_end(i); ++k) {
      builder.insert(static_cast<core::Index>(i), old_csr.indices()[k], old_csr.values()[k]);
    }
  }
  for (std::size_t r = 0; r < cuts.size(); ++r) {
    for (const auto& [col, coeff] : cuts[r].entries) {
      builder.insert(static_cast<core::Index>(old_m + r), col, coeff);
    }
  }
  problem.A = builder.finish();

  RealVector new_lower(new_m);
  RealVector new_upper(new_m);
  for (std::size_t i = 0; i < old_m; ++i) {
    new_lower[i] = problem.row_lower[i];
    new_upper[i] = problem.row_upper[i];
  }
  for (std::size_t r = 0; r < cuts.size(); ++r) {
    new_lower[old_m + r] = -core::INF;
    new_upper[old_m + r] = cuts[r].bound;
  }
  problem.row_lower = std::move(new_lower);
  problem.row_upper = std::move(new_upper);
  return core::Status::Ok();
}

/// A capacity row's `<=` structure, extracted once and reused by both
/// single-row and cross-row (aggregated) cut separation below: `sum
/// entries[k].coeff * x[entries[k].col] <= capacity`.
struct RowView {
  std::vector<std::pair<core::Index, Real>> entries;  // (column, coefficient), coeff > 0
  Real capacity;
};

/// True for a column that behaves as 0/1 for cover/GCD-cut purposes: either
/// tagged `Binary` outright, or `Integer` with bounds exactly `[0,1]`. MPS
/// has no dedicated "binary" bound type outside the explicit `BV` marker
/// (MpsReader.cpp) -- a column written as `INTORG ... UP bnd x 1` (integer,
/// implicit lower 0, explicit upper 1 -- markshare_4_0.mps's exact
/// declaration for all 30 of its variables) is tagged `Integer`, not
/// `Binary`, even though it is one. Checking `col_type == Binary` alone was
/// a real, caught bug: it silently rejected every row of markshare_4_0.mps,
/// so cover-cut separation (both single-row and the cross-row aggregation
/// below) never fired on it at all despite the file being exactly the
/// knapsack-cut-shaped instance this was built for.
[[nodiscard]] bool is_binary_like(const Problem& problem, std::size_t j) noexcept {
  if (problem.col_type[j] == VarType::Binary) return true;
  return problem.col_type[j] == VarType::Integer && problem.col_lower[j] == 0.0 &&
         problem.col_upper[j] == 1.0;
}

/// Extracts row `row`'s entries IF it is eligible for cover/GCD-cut
/// reasoning: finite row_upper (a pure `<=` capacity constraint), every
/// column effectively 0/1 (see `is_binary_like`), every coefficient
/// POSITIVE. Rows with a negative coefficient would need variable
/// complementation (`x_j -> 1-x_j`) first, not attempted here. Returns
/// false (view left empty) otherwise.
[[nodiscard]] bool eligible_capacity_row(const Problem& problem, std::size_t row, RowView& out) {
  if (!core::is_finite_bound(problem.row_upper[row])) return false;
  out.entries.clear();
  out.capacity = problem.row_upper[row];
  const auto& csr = problem.A.csr;
  for (auto k = csr.slice_begin(row); k < csr.slice_end(row); ++k) {
    const auto j = static_cast<std::size_t>(csr.indices()[k]);
    if (!is_binary_like(problem, j)) return false;
    const Real coeff = csr.values()[k];
    if (coeff <= 0.0) return false;
    out.entries.emplace_back(csr.indices()[k], coeff);
  }
  return !out.entries.empty();
}

/// Combines several eligible capacity rows into ONE aggregated view:
/// coefficient-wise sum, bound-wise sum. A nonnegative (all-ones)
/// combination of valid `a'x <= b` inequalities is itself a valid
/// inequality, so this is sound regardless of which rows are combined --
/// the point of doing it is that the AGGREGATE can reveal a cover or a GCD
/// structure no single one of its rows shows on its own (a real, standard
/// technique for exactly the markshare/Cornuejols-Dawande instance family
/// this was built against: 4 knapsack rows over the SAME 30 binary columns,
/// each individually weak, but not necessarily weak in combination).
void aggregate_capacity_rows(const Problem& problem, const std::vector<std::size_t>& rows,
                             RowView& out) {
  const std::size_t n = problem.num_cols();
  std::vector<Real> coeff_by_col(n, 0.0);
  std::vector<bool> touched(n, false);
  out.capacity = 0.0;
  for (std::size_t row : rows) {
    RowView rv;
    // Caller already checked eligibility, but re-checking here (rather than
    // discarding the result) means a future caller that forgets to pre-
    // filter degrades to skipping that row instead of aggregating garbage.
    if (!eligible_capacity_row(problem, row, rv)) continue;
    out.capacity += rv.capacity;
    for (const auto& [col, coeff] : rv.entries) {
      const auto j = static_cast<std::size_t>(col);
      coeff_by_col[j] += coeff;
      touched[j] = true;
    }
  }
  out.entries.clear();
  for (std::size_t j = 0; j < n; ++j) {
    if (touched[j]) out.entries.emplace_back(static_cast<core::Index>(j), coeff_by_col[j]);
  }
}

/// Greedy knapsack cover separation (Wolsey, "Integer Programming", the
/// standard textbook heuristic -- not exact minimum-weight-cover separation,
/// and the resulting cut is not lifted to strengthen it against columns
/// outside the cover; both are real, known-available improvements left for
/// later).
///
/// A "cover" is a subset S of `view`'s columns with `sum_{j in S} a_j >
/// view.capacity` -- infeasible for the row/aggregate alone, so `sum_{j in
/// S} x_j <= |S|-1` is valid for every 0/1 point that satisfies it. Greedily
/// builds S by adding columns in DESCENDING order of their current LP value
/// `x*_j` (the columns closest to "fully used" in the relaxation, which is
/// what makes a violated cover likely), stopping the moment the running
/// coefficient sum exceeds the bound. Returns no cut if no cover is found,
/// or the resulting inequality is not violated by `x` (a valid but useless
/// cut).
[[nodiscard]] bool try_cover_cut_on_view(const RowView& view, const RealVector& x, CoverCut& out) {
  struct Entry {
    core::Index col;
    Real coeff;
    Real value;
  };
  std::vector<Entry> entries;
  entries.reserve(view.entries.size());
  for (const auto& [col, coeff] : view.entries) {
    entries.push_back({col, coeff, x[static_cast<std::size_t>(col)]});
  }
  std::sort(entries.begin(), entries.end(),
            [](const Entry& a, const Entry& b) { return a.value > b.value; });

  Real running = 0.0;
  std::vector<Entry> cover;
  for (const auto& e : entries) {
    if (running > view.capacity) break;
    cover.push_back(e);
    running += e.coeff;
  }
  if (running <= view.capacity) return false;  // no cover here

  Real lhs = 0.0;
  out.entries.clear();
  for (const auto& e : cover) {
    lhs += e.value;
    out.entries.emplace_back(e.col, 1.0);
  }
  out.bound = static_cast<Real>(cover.size()) - 1.0;

  constexpr Real kViolationTol = 1e-6;
  return lhs > out.bound + kViolationTol;
}

[[nodiscard]] bool try_cover_cut(const Problem& problem, const RealVector& x, std::size_t row,
                                  CoverCut& out) {
  RowView view;
  if (!eligible_capacity_row(problem, row, view)) return false;
  return try_cover_cut_on_view(view, x, out);
}

std::int64_t gcd_i64(std::int64_t a, std::int64_t b) noexcept {
  while (b != 0) {
    const auto t = b;
    b = a % b;
    a = t;
  }
  return a;
}

/// GCD (Chvatal-Gomory rounding) cut on an aggregated view: since every
/// column here is BINARY (an integer) and every coefficient is an exact
/// integer, any 0/1 combination is necessarily a multiple of
/// `g = gcd(|coeff|)` -- so the AGGREGATE's own bound can be rounded down to
/// the nearest multiple of `g` with no 0/1 point lost, exactly
/// MilpPresolve.hpp's `tighten_integer_rows` reasoning, applied here to a
/// SYNTHETIC combined row instead of a real one. Unlike that function (which
/// tightens a row's bound unconditionally, since it is always sound to do
/// so), this only reports a cut when the tightened bound is actually
/// VIOLATED by the current relaxation -- an unviolated tightening would be
/// valid but useless to add as a new row.
[[nodiscard]] bool try_gcd_cut_on_view(const RowView& view, const RealVector& x, CoverCut& out) {
  std::int64_t g = 0;
  Real lhs = 0.0;
  for (const auto& [col, coeff] : view.entries) {
    const Real rounded = std::round(coeff);
    if (std::fabs(coeff - rounded) > 1e-9) return false;  // needs exact-integer coefficients
    const auto c = static_cast<std::int64_t>(rounded);
    lhs += coeff * x[static_cast<std::size_t>(col)];
    if (c == 0) continue;
    const std::int64_t mag = c < 0 ? -c : c;
    g = g == 0 ? mag : gcd_i64(g, mag);
  }
  if (g <= 1) return false;  // no exploitable common factor

  const Real gd = static_cast<Real>(g);
  const Real tightened = gd * std::floor(view.capacity / gd + 1e-9);
  if (tightened >= view.capacity - 1e-9) return false;  // already tight; nothing to add

  constexpr Real kViolationTol = 1e-6;
  if (lhs <= tightened + kViolationTol) return false;  // valid but not violated -- useless

  out.entries = view.entries;
  out.bound = tightened;
  return true;
}

}  // namespace

core::Status add_root_cuts(Problem& working, const RelaxationSolver& solve,
                           CutStatistics* stats) {
  // Root-only knapsack cover cutting ("cut-and-branch": cuts are generated
  // once, up front, against the root relaxation, then held fixed through
  // the whole search -- not regenerated per node, which would be full
  // branch-and-cut). Every cut here is a globally valid inequality (see
  // try_cover_cut's doc comment), so this is sound regardless of where in
  // the tree it is added -- adding it once at the root, before any node is
  // solved, benefits every node for free. Capped at a small, fixed number
  // of rounds/cuts: this is a real, bounded improvement, not a full
  // separation engine.
  // Cross-row (aggregated) cutting: a SINGLE row's own cover/GCD structure
  // can be too weak to expose anything (confirmed on markshare_4_0.mps,
  // whose 4 knapsack rows over the same 30 binary columns are each
  // individually weak -- best_bound stayed pinned near zero across 10,000+
  // nodes regardless of budget), but a nonnegative COMBINATION of several
  // rows is still a valid inequality and can reveal a cover or GCD
  // structure none of its component rows shows alone. Eligible ORIGINAL
  // rows (not cut rows added below -- aggregating a cut with itself or
  // another cut is not attempted this pass) are computed once, since
  // eligibility (finite capacity, all-binary, all-positive) never changes
  // for an original row once GCD tightening above has run. Full power-set
  // enumeration is exponential in the eligible count, so it is only
  // attempted when that count is small -- exactly markshare_4_0's shape (4
  // rows), not a general-purpose separation engine for models with many
  // eligible rows.
  const std::size_t original_m = working.num_rows();
  std::vector<std::size_t> eligible_original_rows;
  for (std::size_t i = 0; i < original_m; ++i) {
    RowView rv;
    if (eligible_capacity_row(working, i, rv)) eligible_original_rows.push_back(i);
  }
  constexpr std::size_t kMaxRowsForAggregation = 12;
  const bool try_aggregation =
      eligible_original_rows.size() >= 2 && eligible_original_rows.size() <= kMaxRowsForAggregation;

  constexpr std::size_t kMaxCutRounds = 20;
  constexpr std::size_t kMaxCutsPerRound = 10;
  for (std::size_t round = 0; round < kMaxCutRounds; ++round) {
    auto root_relax = solve(working);
    if (!root_relax.has_value() || root_relax->status != SolverStatus::Optimal) {
      // An unreliable or infeasible root relaxation is exactly what the main
      // loop below already has to handle on its own first pop -- stop
      // cutting and let it do so, rather than duplicating that handling
      // here against a point cuts cannot safely be separated from anyway.
      break;
    }

    std::vector<CoverCut> cuts;
    const std::size_t old_m = working.num_rows();
    for (std::size_t i = 0; i < old_m && cuts.size() < kMaxCutsPerRound; ++i) {
      CoverCut cut;
      if (try_cover_cut(working, root_relax->x, i, cut)) cuts.push_back(std::move(cut));
    }

    if (try_aggregation && cuts.size() < kMaxCutsPerRound) {
      const std::size_t k = eligible_original_rows.size();
      const std::size_t subset_count = (std::size_t{1} << k) - 1;
      for (std::size_t mask = 1; mask <= subset_count && cuts.size() < kMaxCutsPerRound; ++mask) {
        // Popcount < 2 is a single row, already tried above.
        if ((mask & (mask - 1)) == 0) continue;
        std::vector<std::size_t> subset;
        for (std::size_t bit = 0; bit < k; ++bit) {
          if (mask & (std::size_t{1} << bit)) subset.push_back(eligible_original_rows[bit]);
        }
        RowView aggregated;
        aggregate_capacity_rows(working, subset, aggregated);

        CoverCut cut;
        if (try_cover_cut_on_view(aggregated, root_relax->x, cut)) {
          cuts.push_back(std::move(cut));
        } else if (try_gcd_cut_on_view(aggregated, root_relax->x, cut)) {
          cuts.push_back(std::move(cut));
        }
      }
    }

    if (cuts.empty()) break;  // no violated cut found -- nothing left to add
    if (const auto st = append_cut_rows(working, cuts); !st.ok()) return st;
    if (stats != nullptr) {
      ++stats->rounds;
      stats->cuts += cuts.size();
    }
  }

  return core::Status::Ok();
}

}  // namespace sovsolve::solver
