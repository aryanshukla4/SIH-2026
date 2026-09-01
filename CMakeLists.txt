cmake_minimum_required(VERSION 3.24)
project(sovsolve VERSION 0.1.0 LANGUAGES CXX)

# C++20, not 23: nvcc on Windows requires MSVC, whose C++23 support lags.
# Core headers must compile under both MinGW g++ and MSVC/nvcc.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CompilerWarnings)
include(Sanitizers)

option(SOVSOLVE_BUILD_TESTS  "Build tests"      ON)
option(SOVSOLVE_BUILD_BENCH  "Build benchmarks" ON)
option(SOVSOLVE_BUILD_TOOLS  "Build CLI tools"  ON)

# --------------------------------------------------------------------------
# Module targets.
#
# The dependency graph is ENFORCED here, not by convention:
#     core     -> (nothing)
#     model    -> core
#     analysis -> core
#     io       -> core, model
#
# The handoff docs claim the modules are independent; nothing enforced it.
#
# These targets express the graph, but cannot enforce it on their own: every
# module exposes the same include/ root, so any header can reach any other.
# scripts/check_layering.py scans the actual #include edges and runs as a
# ctest case, so a violation fails the build like a compile error.
# --------------------------------------------------------------------------

add_subdirectory(src/core)
add_subdirectory(src/model)
add_subdirectory(src/analysis)
add_subdirectory(src/io)

add_library(sovsolve INTERFACE)
target_link_libraries(sovsolve INTERFACE
  sovsolve_core sovsolve_model sovsolve_analysis sovsolve_io)

# Tools come BEFORE tests: the oracle test runs `modeldump`, and a
# `if(TARGET modeldump)` guard is evaluated at configure time, so a test
# declared first would silently not be registered at all.
if(SOVSOLVE_BUILD_TOOLS)
  add_subdirectory(tools/mpsinfo)
  add_subdirectory(tools/modeldump)
endif()

if(SOVSOLVE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(SOVSOLVE_BUILD_BENCH)
  add_subdirectory(bench)
endif()
