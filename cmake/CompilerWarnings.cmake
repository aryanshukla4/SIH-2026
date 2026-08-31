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
