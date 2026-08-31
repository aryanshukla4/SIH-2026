add_library(sovsolve_io STATIC
  Load.cpp
  MpsReader.cpp
  detail/MappedFile.cpp)

target_include_directories(sovsolve_io
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  # src/ is PRIVATE so detail/ headers stay unreachable from other modules.
  PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>)

target_link_libraries(sovsolve_io PUBLIC sovsolve_core sovsolve_model)

# zlib is optional. Without it, gzip input returns a clear UnsupportedFeature
# error rather than being mis-parsed as binary garbage. MIPLIB ships .mps.gz,
# so it is worth having; Netlib MPS is uncompressed and works either way.
find_package(ZLIB QUIET)
if(ZLIB_FOUND)
  target_link_libraries(sovsolve_io PRIVATE ZLIB::ZLIB)
  target_compile_definitions(sovsolve_io PRIVATE SOVSOLVE_WITH_ZLIB)
  message(STATUS "zlib found: gzip (.mps.gz) input enabled")
else()
  message(STATUS "zlib NOT found: .mps.gz input will report UnsupportedFeature")
endif()

sovsolve_set_warnings(sovsolve_io)
sovsolve_set_sanitizers(sovsolve_io)
