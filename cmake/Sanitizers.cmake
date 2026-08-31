# ASan/UBSan preset. The MPS reader does raw pointer arithmetic over an mmap
# buffer; the fuzz suite is only meaningful with sanitizers on.
option(SOVSOLVE_SANITIZE "Enable address+undefined sanitizers" OFF)

function(sovsolve_set_sanitizers target)
  if(SOVSOLVE_SANITIZE AND NOT MSVC)
    target_compile_options(${target} PRIVATE
      -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endif()
endfunction()
