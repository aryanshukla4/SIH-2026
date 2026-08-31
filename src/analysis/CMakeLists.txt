add_library(sovsolve_analysis INTERFACE)
target_include_directories(sovsolve_analysis INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
target_link_libraries(sovsolve_analysis INTERFACE sovsolve_core)
