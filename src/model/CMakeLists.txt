add_library(sovsolve_model INTERFACE)
target_include_directories(sovsolve_model INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
target_link_libraries(sovsolve_model INTERFACE sovsolve_core)
