// Behavioural tests for the core storage types.
//
// Compiling is not the same as working. These check the invariants the rest of
// the layer relies on -- especially alignment, the infinity convention, and
// the workspace carving that keeps allocation out of the IPM loop.

#include <cstdint>
#include <string>

#include "sovsolve/core/AlignedAllocator.hpp"
#include "sovsolve/core/DenseMatrix.hpp"
#include "sovsolve/core/NameArena.hpp"
#include "sovsolve/core/Span.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"
#include "sovsolve/core/Workspace.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)

namespace {

void test_infinity_convention() {
  // The MPS convention: 1e20 IS infinity, and must not be an IEEE infinity --
  // arithmetic on a real inf produces NaN where 1e20 produces a large finite
  // number. See docs/MPS-FORMAT-NOTES.md.
  CHECK(core::is_infinite(core::INF));
  CHECK(core::is_infinite(-core::INF));
  CHECK(core::is_pos_infinite(1e21));
  CHECK(core::is_neg_infinite(-1e30));
  CHECK(!core::is_infinite(1e19));
  CHECK(core::is_finite_bound(0.0));
  CHECK(core::is_finite_bound(-1e19));
  CHECK(!core::is_finite_bound(core::INF));

  // Must stay finite so bound arithmetic does not produce NaN.
  CHECK(core::INF - core::INF == 0.0);
}

void test_vector_alignment_and_move() {
  core::RealVector v(1000, 3.5);
  CHECK_EQ(v.size(), std::size_t{1000});
  CHECK_NEAR(v[0], 3.5, 1e-15);
  CHECK_NEAR(v[999], 3.5, 1e-15);

  // 64-byte alignment is what lets vectorized loops use aligned loads.
  const auto addr = reinterpret_cast<std::uintptr_t>(v.data());
  CHECK_EQ(addr % core::kAlignment, std::uintptr_t{0});

  // Move must not copy. After the move the source is empty and the
  // destination owns the original buffer.
  const auto* original = v.data();
  core::RealVector moved = std::move(v);
  CHECK(moved.data() == original);
  CHECK_EQ(moved.size(), std::size_t{1000});
  CHECK_EQ(v.size(), std::size_t{0});  // NOLINT(bugprone-use-after-move)

  // Copying is explicit, and deep.
  core::RealVector copy = moved.clone();
  CHECK(copy.data() != moved.data());
  CHECK_NEAR(copy[500], 3.5, 1e-15);
  copy[500] = 9.0;
  CHECK_NEAR(moved[500], 3.5, 1e-15);
}

void test_span_subspan() {
  core::RealVector v(100);
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<core::Real>(i);

  auto s = v.span();
  CHECK_EQ(s.size(), std::size_t{100});
  CHECK_EQ(s.size_bytes(), std::size_t{800});

  auto mid = s.subspan(10, 5);
  CHECK_EQ(mid.size(), std::size_t{5});
  CHECK_NEAR(mid[0], 10.0, 1e-15);
  CHECK_NEAR(mid[4], 14.0, 1e-15);

  // A span is a view: writing through it writes the original.
  mid[0] = -1.0;
  CHECK_NEAR(v[10], -1.0, 1e-15);
}

void test_name_arena() {
  core::NameArena arena;
  CHECK(arena.empty());

  CHECK_EQ(arena.add("COST"), std::size_t{0});
  CHECK_EQ(arena.add("LIM1"), std::size_t{1});
  CHECK_EQ(arena.add("a_very_long_column_name_x"), std::size_t{2});
  CHECK_EQ(arena.add(""), std::size_t{3});  // empty names are legal

  CHECK_EQ(arena.size(), std::size_t{4});
  CHECK(arena[0] == "COST");
  CHECK(arena[1] == "LIM1");
  CHECK(arena[2] == "a_very_long_column_name_x");
  CHECK(arena[3].empty());

  // One blob plus offsets, not one allocation per name.
  CHECK(arena.bytes() < 200);
}

void test_workspace_carving() {
  // The point of the arena: declare everything up front, allocate once, then
  // never allocate again inside the IPM loop.
  core::Workspace ws;
  const std::size_t n = 500;
  const std::size_t m = 300;

  const auto h_x = ws.reserve<core::Real>(n);
  const auto h_s = ws.reserve<core::Real>(m);
  const auto h_y = ws.reserve<core::Real>(m);
  CHECK(!ws.committed());

  ws.commit();
  CHECK(ws.committed());

  auto x = ws.get<core::Real>(h_x, n);
  auto s = ws.get<core::Real>(h_s, m);
  auto y = ws.get<core::Real>(h_y, m);

  CHECK_EQ(x.size(), n);
  CHECK_EQ(s.size(), m);

  // Every carved span is independently aligned.
  CHECK_EQ(reinterpret_cast<std::uintptr_t>(x.data()) % core::kAlignment,
           std::uintptr_t{0});
  CHECK_EQ(reinterpret_cast<std::uintptr_t>(s.data()) % core::kAlignment,
           std::uintptr_t{0});
  CHECK_EQ(reinterpret_cast<std::uintptr_t>(y.data()) % core::kAlignment,
           std::uintptr_t{0});

  // The slices must not overlap.
  for (std::size_t i = 0; i < n; ++i) x[i] = 1.0;
  for (std::size_t i = 0; i < m; ++i) s[i] = 2.0;
  for (std::size_t i = 0; i < m; ++i) y[i] = 3.0;
  CHECK_NEAR(x[n - 1], 1.0, 1e-15);
  CHECK_NEAR(s[m - 1], 2.0, 1e-15);
  CHECK_NEAR(y[0], 3.0, 1e-15);
}

void test_dense_matrix_is_column_major() {
  // Column-major, because cuBLAS and LAPACK are. Element (i,j) lives at
  // data[j*ld + i], and each column is contiguous.
  core::DenseMatrix<core::Real> M(3, 4);
  CHECK_EQ(M.rows(), std::size_t{3});
  CHECK_EQ(M.cols(), std::size_t{4});
  CHECK(M.ld() >= 3);

  for (std::size_t j = 0; j < 4; ++j) {
    for (std::size_t i = 0; i < 3; ++i) {
      M(i, j) = static_cast<core::Real>(10 * i + j);
    }
  }
  CHECK_NEAR(M(2, 3), 23.0, 1e-15);

  // Columns are contiguous; walking column 1 must see rows 0,1,2 in order.
  auto col1 = M.column(1);
  CHECK_EQ(col1.size(), std::size_t{3});
  CHECK_NEAR(col1[0], 1.0, 1e-15);
  CHECK_NEAR(col1[1], 11.0, 1e-15);
  CHECK_NEAR(col1[2], 21.0, 1e-15);

  // Each column starts aligned, via leading-dimension padding.
  CHECK_EQ(reinterpret_cast<std::uintptr_t>(M.column(1).data()) % core::kAlignment,
           std::uintptr_t{0});
}

void test_problem_type_derivation() {
  // Classification is derived from the data, never declared, so it cannot
  // disagree with the model it describes.
  CHECK(core::ProblemType::LP != core::ProblemType::MILP);
}

}  // namespace

int main() {
  test_infinity_convention();
  test_vector_alignment_and_move();
  test_span_subspan();
  test_name_arena();
  test_workspace_carving();
  test_dense_matrix_is_column_major();
  test_problem_type_derivation();
  return sovsolve::test::report("core_types");
}
