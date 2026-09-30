# Security Policy

## Supported versions

Only the latest commit on `main` receives fixes.

## Reporting a vulnerability

Please do **not** open a public issue for security problems.

Report them privately through GitHub's
[private vulnerability reporting](https://github.com/aryanshukla4/SIH-2026/security/advisories/new)
(the **Security** tab, then **Report a vulnerability**). Include the input file
that triggers the problem, the command you ran, and what happened.

The most likely class of issue in this project is a malformed model file
(`.mps`, `.lp`, `.qplib`) that makes a reader crash, read out of bounds or use
unbounded memory. Those reports are very welcome.

You should get a first response within a week. Once a fix is merged, the
report will be published as a security advisory, with credit to you if you
want it.
