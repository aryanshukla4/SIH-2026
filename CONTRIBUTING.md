# Contributing to SovSolve

Thanks for taking the time to contribute. Bug reports, test models that break
the solver, fixes and new features are all welcome.

By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting a bug

Open an issue with the **Bug report** template. The single most useful thing
you can attach is the model file (`.mps`, `.lp` or `.qplib`) together with the
exact command you ran, the output, and the answer you expected. A wrong
objective, a wrong status (for example `Infeasible` on a feasible model), or a
crash on a valid file are all bugs.

Security problems should not go in a public issue -- see
[SECURITY.md](SECURITY.md).

## Building and testing

You need a C++20 compiler (GCC 13+, MinGW-w64 or MSVC), CMake 3.24 or newer,
Ninja and Python 3.

```sh
cmake --preset release
cmake --build build
ctest --test-dir build --output-on-failure
```

Optional: zlib (`.mps.gz` input) and SuiteSparse CHOLMOD
(`sudo apt-get install zlib1g-dev libsuitesparse-dev`). The GPU engines need
the CUDA toolkit and `cmake --preset cuda`; see the [README](README.md).

The test corpus the suite reads is committed under `tests/data/`. Larger
Netlib models for benchmarking are fetched with `python3 scripts/fetch_netlib.py`.

## Making a change

1. Fork the repository and create a branch from `main`.
2. Keep each pull request to one logical change. Small PRs are reviewed faster.
3. Add or update tests for the behaviour you change.
4. Make sure the full test suite passes locally. The build uses
   `-Werror`, so new warnings fail the build.
5. Format C++ with the repository's `.clang-format`.
6. Open a pull request against `main` and fill in the template. CI must pass on
   Linux and Windows before a PR can be merged, and every PR needs one
   approving review.

## Project rules

These are what reviewers check first.

- **Never a false verdict.** A change to status or verdict logic must not
  report `Infeasible` or `Unbounded` on a model that has an optimum. Show the
  change on the feasible Netlib models, for every method it touches, in the PR.
- **Numbers come from somewhere.** Algorithms and constants should cite the
  paper and section they come from. A value chosen by measurement instead is
  labelled as such, with the measurement.
- **Tunables are registered.** Any tuning constant you add or change goes in
  [`docs/TUNABLES.md`](docs/TUNABLES.md): value, file, source (paper or
  measurement) and the command-line flag that controls it.
- **No complete third-party solvers.** Linear-algebra libraries (for example
  CHOLMOD, cuSPARSE, cuDSS) are fine. Embedding or calling a complete LP/QP/MIP
  solver (for example HiGHS or cuPDLPx) is not; the engines are written here.
- **Deterministic results.** Parallel code must give the same answer on every
  run for the same input and thread count.
- **Performance claims come with numbers.** If a PR is meant to make something
  faster, include before/after timings, the models, and the machine.

## Licensing

SovSolve is licensed under the [Apache License 2.0](LICENSE). By submitting a
pull request you agree that your contribution is licensed under the same terms
(Apache-2.0, section 5).
