// mpsinfo -- dump the analyzer report for a model file.
//
// The manual smoke test: run it on a small instance whose dimensions are
// known (Netlib afiro is 27 x 32 with 83 nonzeros) and check the numbers by
// hand. Wired up once the reader lands.

#include <cstdio>

#include "sovsolve/io/Load.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: mpsinfo <model file>\n");
    return 2;
  }
  std::fprintf(stderr, "mpsinfo: reader not yet implemented (%s)\n", argv[1]);
  return 1;
}
