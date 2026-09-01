# ASan/UBSan preset, for toolchains that ship the runtimes.
#
# MinGW-w64 GCC 15 -- the toolchain this repo is developed on -- does NOT: there
# is no libasan or libubsan, so `-fsanitize=address` fails at link time. Turning
# this option on there produces a build error, not a slower build.
#
# The readers scan a memory-mapped buffer with raw pointer arithmetic, so an
# overrun detector is not optional. `tests/fuzz/parser_fuzz.cpp` therefore
# supplies its own: it places each input against a GUARD PAGE, so any read past
# the end faults deterministically, and it verifies the guard actually faults
# before trusting a clean run. That covers the overrun class without a runtime.
#
# This option remains useful on Linux/macOS CI, where the runtimes exist and
# UBSan additionally catches signed overflow and misaligned access that guard
# pages cannot see.
option(SOVSOLVE_SANITIZE "Enable address+undefined sanitizers" OFF)

function(sovsolve_set_sanitizers target)
  if(SOVSOLVE_SANITIZE AND NOT MSVC)
    target_compile_options(${target} PRIVATE
      -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endif()
endfunction()
