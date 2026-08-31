# Warnings-as-errors. A silent narrowing conversion in index arithmetic is a
# wrong answer on a million-variable problem, not a style issue.
function(sovsolve_set_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /WX /permissive- /Zc:__cplusplus)
  else()
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
      -Wconversion -Wsign-conversion
      -Wshadow -Wnon-virtual-dtor -Wold-style-cast
      -Wcast-align -Wunused -Woverloaded-virtual
      -Wnull-dereference -Wdouble-promotion -Wformat=2)
  endif()
endfunction()

# Static-link the GCC runtime on MinGW.
#
# Windows resolves DLLs from PATH, so any older libstdc++-6.dll belonging to
# some unrelated application shadows the toolchain's own and the binary dies
# with STATUS_ENTRYPOINT_NOT_FOUND (0xc0000139) before main() runs. That is
# environmental, not a code defect, but it makes the test suite unreliable and
# would make any binary we hand to a teammate unreliable too.
#
# Linking the runtime statically removes the dependency entirely and makes the
# executables self-contained.
function(sovsolve_static_runtime target)
  if(MINGW)
    target_link_options(${target} PRIVATE -static-libgcc -static-libstdc++)
  endif()
endfunction()
