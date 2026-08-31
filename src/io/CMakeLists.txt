add_library(sovsolve_io INTERFACE)
target_include_directories(sovsolve_io INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
target_link_libraries(sovsolve_io INTERFACE sovsolve_core sovsolve_model)
