add_executable(mpsinfo main.cpp)
target_link_libraries(mpsinfo PRIVATE sovsolve_io sovsolve_analysis)
sovsolve_set_warnings(mpsinfo)
sovsolve_static_runtime(mpsinfo)
