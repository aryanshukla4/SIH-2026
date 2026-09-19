add_library(sovsolve_solver STATIC
  Scaler.cpp
  Initializer.cpp
  Presolver.cpp
  MilpPresolve.cpp
  SolutionReconstructor.cpp
  SolutionQuality.cpp
  Iis.cpp
  DualBound.cpp
  Homogeneous.cpp
  HomogeneousSolve.cpp
  HostKkt.cpp
  HomogeneousStep.cpp
  HomogeneousNewton.cpp
  ConvergenceChecker.cpp
  Diagnostics.cpp
  Logging.cpp
  # Module 23, the dual simplex. Host-only by construction: LU, FTRAN/BTRAN and
  # the ratio test are sequential and latency-bound, so they belong here rather
  # than in sovsolve_solver_gpu. The pricing sweep is the one part with enough
  # parallelism to be worth offloading, and it goes behind an interface so the
  # GPU implementation can live in the gpu library without this one ever
  # linking against it -- the solver -> gpu edge must stay absent (see below).
  simplex/detail/SimplexEngine.cpp
  simplex/Basis.cpp
  simplex/LuFactor.cpp
  simplex/DualSimplex.cpp
  simplex/PrimalSimplex.cpp
  simplex/SolveSimplex.cpp
  simplex/SimplexSolution.cpp
  # Module 24, PDLP. Host-only for now for the same reason the simplex is, but
  # for the opposite eventual destination: PDLP's inner loop is nothing BUT
  # SpMV against K and K', which is the one operation this hardware's FP64
  # penalty leaves competitive (docs/ARCHITECTURE-REVIEW.md 3.5). The matrix
  # product therefore goes behind an interface so a cuSPARSE implementation can
  # live in the gpu library and be injected, without this library ever linking
  # against it.
  pdlp/TrustRegion.cpp
  pdlp/MatVec.cpp
  pdlp/IterationBackend.cpp
  pdlp/DualityGap.cpp
  pdlp/Infeasibility.cpp
  pdlp/Pdlp.cpp
  pdlp/PdlpSolution.cpp
  LpSolve.cpp)
# Regularization is header-only (see Regularization.hpp) and PredictorCorrector
# lives under gpu/ -- it orchestrates GPU module calls, so it belongs in
# sovsolve_solver_gpu, not here.

target_include_directories(sovsolve_solver PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
target_link_libraries(sovsolve_solver PUBLIC
  sovsolve_core sovsolve_model sovsolve_analysis)
sovsolve_set_warnings(sovsolve_solver)
sovsolve_set_sanitizers(sovsolve_solver)

# GPU-boundary modules (Modules 7, 9, 10, 12-16): SpMV, KKT assembly,
# factorization/solve, refinement, step length, state update -- see
# architecture.txt "## GPU boundary". Built only under WSL2/Linux with the
# CUDA toolkit; the host library above stays fully self-contained without it.
if(SOVSOLVE_ENABLE_CUDA)
  add_library(sovsolve_solver_gpu STATIC
    gpu/ResidualCalculator.cu
    gpu/KktBuilder.cu
    gpu/Ordering.cu
    gpu/VectorOps.cu
    gpu/Preconditioner.cu
    gpu/LinearSolver.cu
    gpu/NewtonRecovery.cu
    gpu/StepLength.cu
    gpu/StateUpdate.cu
    gpu/MuController.cu
    gpu/PredictorCorrector.cu
    gpu/PdlpMatVec.cu
    gpu/Solve.cu
    gpu/BranchAndBound.cu)

  target_include_directories(sovsolve_solver_gpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
  target_link_libraries(sovsolve_solver_gpu PUBLIC
    sovsolve_core sovsolve_model sovsolve_analysis sovsolve_solver
    CUDA::cublas CUDA::cusparse CUDA::cusolver)
  # This is a ONE-WAY edge: sovsolve_solver_gpu -> sovsolve_solver, needed
  # because gpu/Solve.cu (the top-level solve_problem() entry point) calls
  # host-only functions (canonicalize, initialize, ConvergenceChecker,
  # reconstruct_solution) alongside gpu::run_iteration. It is safe exactly
  # because it is one-way: sovsolve_solver does not, and must not, link
  # back (see Regularization.hpp's header comment) -- and this library
  # already requires the CUDA toolkit to exist at all, so depending on the
  # always-available host library costs it nothing it doesn't already have.
  # Not CUDA_SEPARABLE_COMPILATION: these .cu files are plain host C++ so far
  # (see the GPU module headers' STUB notes) with no __global__/__device__
  # code calling across translation units, so relocatable device code buys
  # nothing yet and its device-link step does not chain cleanly through
  # sovsolve_solver_gpu -> sovsolve_solver -> a test executable. Revisit once
  # real kernels exist.

  if(SOVSOLVE_CUDSS_LIBRARY)
    target_link_libraries(sovsolve_solver_gpu PUBLIC ${SOVSOLVE_CUDSS_LIBRARY})
    target_compile_definitions(sovsolve_solver_gpu PUBLIC SOVSOLVE_WITH_CUDSS)
  endif()

  # Still no edge the OTHER way (sovsolve_solver -> sovsolve_solver_gpu):
  # nothing in the host library calls a GPU function. Both PredictorCorrector
  # and Solve (the top-level entry point) needed to, and both live in
  # sovsolve_solver_gpu instead, for exactly this reason.
endif()
