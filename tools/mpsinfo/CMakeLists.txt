add_executable(mpsinfo main.cpp)
target_link_libraries(mpsinfo PRIVATE sovsolve)
sovsolve_set_warnings(mpsinfo)
