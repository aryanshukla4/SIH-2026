add_library(sovsolve_model STATIC Problem.cpp Canonicalizer.cpp)
target_include_directories(sovsolve_model PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
target_link_libraries(sovsolve_model PUBLIC sovsolve_core)
sovsolve_set_warnings(sovsolve_model)
sovsolve_set_sanitizers(sovsolve_model)
