# Tests link the module targets, so a dependency violation shows up as a
# link error rather than a code-review comment.
add_library(sovsolve_test_support INTERFACE)
target_include_directories(sovsolve_test_support INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>)

function(sovsolve_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE sovsolve sovsolve_test_support)
  sovsolve_set_warnings(${name})
  sovsolve_set_sanitizers(${name})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

sovsolve_add_test(header_compile_check unit/header_compile_check.cpp)
sovsolve_add_test(core_types_test      unit/core_types_test.cpp)

# Architecture enforcement. See scripts/check_layering.py -- the module
# dependency graph is checked mechanically, not by convention.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME layering_check
           COMMAND ${Python3_EXECUTABLE}
                   ${CMAKE_SOURCE_DIR}/scripts/check_layering.py)
else()
  message(WARNING "Python3 not found: layering check will not run")
endif()
