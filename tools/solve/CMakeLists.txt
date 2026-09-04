# solve_problem() lives in sovsolve_solver_gpu, which only exists under
# SOVSOLVE_ENABLE_CUDA -- silently skip this tool on a host-only build,
# same pattern as the oracle_check test's `if(TARGET modeldump)` guard.
if(TARGET sovsolve_solver_gpu)
  add_executable(solve main.cpp)
  target_link_libraries(solve PRIVATE sovsolve_io sovsolve_model sovsolve_solver_gpu)
  sovsolve_set_warnings(solve)
  sovsolve_static_runtime(solve)
endif()
