# Built unconditionally since Module 23.
#
# It used to be guarded by `if(TARGET sovsolve_solver_gpu)`, because the only
# solver entry point lived in the CUDA library -- which meant that on a
# host-only build (the default `release` preset, and the only one that works on
# native Windows) this tool did not exist and nothing downstream of the
# canonicalizer could be run at all. `solver::solve_lp` (the dual simplex) is
# host-only, so the tool is now always buildable; the interior-point path is
# compiled in only when the GPU library is there to provide it.
add_executable(solve main.cpp)
target_link_libraries(solve PRIVATE sovsolve_io sovsolve_model sovsolve_solver)
if(TARGET sovsolve_solver_gpu)
  target_link_libraries(solve PRIVATE sovsolve_solver_gpu)
  target_compile_definitions(solve PRIVATE SOVSOLVE_ENABLE_CUDA)
endif()
sovsolve_set_warnings(solve)
sovsolve_static_runtime(solve)
