# Tests link the module targets, so a dependency violation shows up as a
# link error rather than a code-review comment.
add_library(sovsolve_test_support INTERFACE)
target_include_directories(sovsolve_test_support INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>)

function(sovsolve_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE sovsolve sovsolve_test_support)
  sovsolve_set_warnings(${name})
  sovsolve_set_sanitizers(${name})
  sovsolve_static_runtime(${name})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

sovsolve_add_test(header_compile_check unit/header_compile_check.cpp)
sovsolve_add_test(core_types_test      unit/core_types_test.cpp)
sovsolve_add_test(mps_reader_test      unit/mps_reader_test.cpp)
sovsolve_add_test(lp_reader_test       unit/lp_reader_test.cpp)
sovsolve_add_test(qplib_reader_test    unit/qplib_reader_test.cpp)
sovsolve_add_test(solver_types_test    unit/solver_types_test.cpp)
sovsolve_add_test(presolver_test       property/presolver_test.cpp)
sovsolve_add_test(milp_presolve_test   property/milp_presolve_test.cpp)

# Module 23. Host-only by construction (see src/solver/CMakeLists.txt), so
# unlike the IPM's tests these run in the default `release` preset with no
# CUDA toolkit present. It reads a real instance, so it needs the data path.
sovsolve_add_test(lu_factor_test       unit/lu_factor_test.cpp)
target_compile_definitions(lu_factor_test PRIVATE
  SOVSOLVE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

sovsolve_add_test(dual_simplex_test    unit/dual_simplex_test.cpp)
target_compile_definitions(dual_simplex_test PRIVATE
  SOVSOLVE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

sovsolve_add_test(primal_simplex_test  unit/primal_simplex_test.cpp)
target_compile_definitions(primal_simplex_test PRIVATE
  SOVSOLVE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

# gpu::compute_residuals lives in sovsolve_solver_gpu, which only exists
# under SOVSOLVE_ENABLE_CUDA (see src/solver/CMakeLists.txt) -- registering
# this test unconditionally would fail to link on a host-only build.
if(SOVSOLVE_ENABLE_CUDA)
  sovsolve_add_test(solver_gpu_algorithms_test unit/solver_gpu_algorithms_test.cpp)
  target_compile_definitions(solver_gpu_algorithms_test PRIVATE
    SOVSOLVE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
  sovsolve_add_test(branch_and_bound_test      unit/branch_and_bound_test.cpp)
endif()

# Corpus tests read real files, so they need an absolute data path -- ctest
# runs them from the build tree, not the source tree.
sovsolve_add_test(netlib_corpus_test   corpus/netlib_corpus_test.cpp)
target_compile_definitions(netlib_corpus_test PRIVATE
  SOVSOLVE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(netlib_corpus_test PRIVATE sovsolve_analysis)

sovsolve_add_test(roundtrip_test       corpus/roundtrip_test.cpp)
target_compile_definitions(roundtrip_test PRIVATE
  SOVSOLVE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

sovsolve_add_test(parser_fuzz          fuzz/parser_fuzz.cpp)
target_compile_definitions(parser_fuzz PRIVATE
  SOVSOLVE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

sovsolve_add_test(canonical_test       property/canonical_test.cpp)
target_compile_definitions(canonical_test PRIVATE
  SOVSOLVE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

# Architecture enforcement. See scripts/check_layering.py -- the module
# dependency graph is checked mechanically, not by convention.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME layering_check
           COMMAND ${Python3_EXECUTABLE}
                   ${CMAKE_SOURCE_DIR}/scripts/check_layering.py)
else()
  message(WARNING "Python3 not found: layering check will not run")
endif()

# External validation. Everything else here checks the ingestion layer against
# itself; this checks it against HiGHS (via SciPy) and against Netlib's
# published optima. Skips cleanly when SciPy or Python is absent -- it is a
# dev-only oracle and never a build dependency.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND AND TARGET modeldump)
  add_test(NAME oracle_check
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/scripts/oracle_check.py
            $<TARGET_FILE:modeldump>
            ${CMAKE_CURRENT_SOURCE_DIR}/data/netlib)
endif()
