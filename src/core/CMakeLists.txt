# core depends on nothing.
add_library(sovsolve_core STATIC Status.cpp)
target_include_directories(sovsolve_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
sovsolve_set_warnings(sovsolve_core)
sovsolve_set_sanitizers(sovsolve_core)
