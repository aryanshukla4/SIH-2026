# core depends on nothing. Header-only for now; becomes a static lib when
# non-template code lands.
add_library(sovsolve_core INTERFACE)
target_include_directories(sovsolve_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
