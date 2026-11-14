# Preserve failed runner evidence and nested script outputs

The generated Modal function now tees child-runner stdout/stderr into a log. If that child exits nonzero, it snapshots all existing partial run files and the runner log into an attempt-specific failure directory in the task volume, writes a separate identity/SHA-256 failure manifest, commits the volume, then raises. Failure evidence has no quality score and is not a valid full run.

Successful lightweight export now includes script inputs/outcomes, test equivalents and ASR cache, including per-fold copies. The previous root-only allowlist could omit precisely the evidence needed to diagnose episode behavior. Heavy weights remain outside the lightweight export; this change does not claim trained-weight export support.

Local tests verify partial nested files, initialization failure without a run directory, identity/hash rejection, and execute the generated nested export loop against fixture files. Five targeted tests, eleven existing framework Modal tests and Ruff pass. GPU/container crashes before child-runner execution, live download verification and postlaunch cleanup remain unverified; no live provider calls occurred.
