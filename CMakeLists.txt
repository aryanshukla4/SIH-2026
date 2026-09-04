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

# The GPU solver core (src/solver/gpu). Off by default: it needs the CUDA
# toolkit and, on Windows, nvcc requires MSVC as its host compiler -- so the
# intended path is WSL2 + the Linux CUDA toolkit, via the "cuda" preset. The
# host-only build (release/debug/asan) never needs this on.
option(SOVSOLVE_ENABLE_CUDA "Build the CUDA solver core (needs the CUDA toolkit)" OFF)

if(SOVSOLVE_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  # cuDSS is not part of the stock CUDA toolkit apt package -- degrade to a
  # message rather than a hard configure failure, same spirit as the zlib
  # message in src/io/CMakeLists.txt.
  find_library(SOVSOLVE_CUDSS_LIBRARY NAMES cudss)
  if(SOVSOLVE_CUDSS_LIBRARY)
    message(STATUS "cuDSS found: sparse-direct GPU path enabled")
  else()
    message(STATUS "cuDSS NOT found: sparse-direct GPU path unavailable "
                    "(install it separately -- it does not ship with the CUDA toolkit)")
  endif()
endif()

# --------------------------------------------------------------------------
# Module targets.
#
# The dependency graph is ENFORCED here, not by convention:
#     core     -> (nothing)
#     model    -> core
#     analysis -> core
#     solver   -> core, model, analysis
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
add_subdirectory(src/solver)
add_subdirectory(src/io)

add_library(sovsolve INTERFACE)
target_link_libraries(sovsolve INTERFACE
  sovsolve_core sovsolve_model sovsolve_analysis sovsolve_solver sovsolve_io)
# sovsolve_solver_gpu is linked here, at the umbrella level, rather than from
# sovsolve_solver itself -- src/solver/CMakeLists.txt explains why (nothing
# in the host library calls a GPU function, and linking it there would risk
# a cycle with a future caller). `sovsolve` is exactly the "third party
# depending on both libraries" that comment describes.
if(TARGET sovsolve_solver_gpu)
  target_link_libraries(sovsolve INTERFACE sovsolve_solver_gpu)
endif()

# Tools come BEFORE tests: the oracle test runs `modeldump`, and a
# `if(TARGET modeldump)` guard is evaluated at configure time, so a test
# declared first would silently not be registered at all.
if(SOVSOLVE_BUILD_TOOLS)
  add_subdirectory(tools/mpsinfo)
  add_subdirectory(tools/modeldump)
  add_subdirectory(tools/solve)
endif()

if(SOVSOLVE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(SOVSOLVE_BUILD_BENCH)
  add_subdirectory(bench)
endif()
