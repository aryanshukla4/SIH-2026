add_library(sovsolve_solver STATIC
  Scaler.cpp
  Initializer.cpp
  Regularization.cpp
  PredictorCorrector.cpp
  SolutionReconstructor.cpp
  ConvergenceChecker.cpp
  Diagnostics.cpp
  Logging.cpp)

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
    gpu/LinearSolver.cu
    gpu/NewtonRecovery.cu
    gpu/StepLength.cu
    gpu/StateUpdate.cu
    gpu/MuController.cu)

  target_include_directories(sovsolve_solver_gpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
  target_link_libraries(sovsolve_solver_gpu PUBLIC
    sovsolve_core sovsolve_model sovsolve_analysis
    CUDA::cublas CUDA::cusparse CUDA::cusolver)
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

  target_link_libraries(sovsolve_solver PUBLIC sovsolve_solver_gpu)
endif()
