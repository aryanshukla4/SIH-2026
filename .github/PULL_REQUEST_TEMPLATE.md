## What this changes

<!-- One or two sentences: what the PR does and why. Link the issue if there is one ("Fixes #123"). -->

## How it was tested

<!-- Commands you ran and what they showed. For solver changes, the models you solved. -->

- [ ] `ctest --test-dir build --output-on-failure` passes locally
- [ ] Tests added or updated for the changed behaviour

## Solver changes (skip if not applicable)

- [ ] No false `Infeasible`/`Unbounded` on the feasible Netlib models, for every method touched
- [ ] New or changed tuning constants are recorded in `docs/TUNABLES.md`
- [ ] Performance claims include before/after timings, the models and the machine
- [ ] Algorithms and constants cite their paper and section, or are labelled as measured
