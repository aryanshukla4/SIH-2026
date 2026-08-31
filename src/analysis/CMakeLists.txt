add_library(sovsolve_analysis STATIC MatrixAnalysis.cpp)
target_include_directories(sovsolve_analysis PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
target_link_libraries(sovsolve_analysis PUBLIC sovsolve_core)
sovsolve_set_warnings(sovsolve_analysis)
sovsolve_set_sanitizers(sovsolve_analysis)
