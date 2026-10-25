# Safe Clean & Hygiene

## Scope and safety boundaries

`ai-harness clean` previews an explicit inventory. No category is selected
implicitly; `--legacy`, `--probes`, and `--temp` opt into independent scopes.
Only `--apply` permits mutation, with an interactive confirmation unless `-y`.
An explicit `--dry-run` and `--apply` are mutually exclusive.

Five mandatory guards:

1. Resolve candidates strictly below their workspace or the actual system temp
   root. Reject root deletion, traversal, symbolic links, junctions and Windows
   reparse points in ancestors or descendants. Never recursively follow links.
2. Workspace eligibility is limited to `.ml-harness` and individual probe run
   directories under `.ai-harness/artifacts/<task>/probes/<experiment>/<run>`.
   Protect dataset/data, source, configuration, Git/agent metadata, all declared
   task input paths, and official experiment directories. Protect an ancestor
   of a declared input too. Refuse legacy removal if its attachment is still
   active or the current artifacts directory is inside it. Invalid task metadata
   fails closed. Root files and arbitrary clutter are advisory-only.
   Temp scans also protect the selected workspace and its declared data paths;
   the API accepts optional `project_root` (default: environment/current dir).
3. System temp candidates are immediate child directories with precisely these
   prefixes: `aih-kaggle-job_`, `aih-modal-job_`, `harness_kernel_`,
   `harness_audit_runs`. Prefixes confer eligibility, not arbitrary-path deletion.
4. Temp has a non-reducible two-hour age floor; workspace candidates use the
   requested nonnegative finite age. Use the newest timestamp in the tree,
   recheck on apply, and refuse changed inventory. Staging has a live-PID marker
   so a long running dispatch is excluded even after two hours.
5. Default is read-only. `--archive` writes legacy ZIPs exclusively under a
   checked `.ai-harness/archives` directory, validates ZIP CRCs and unchanged
   source inventory before deletion. Failure retains the source. Existing ZIPs
   are never overwritten. Archives are retained and are not cleanup candidates.

## API and execution

`scan_workspace_clutter(project_root, older_than_hours=2)` and
`scan_system_temp(older_than_hours=2)` return typed plan objects containing
candidate paths, category, sandbox root, bytes and a filesystem snapshot.
Plans can be combined for the CLI. `execute_cleanup(plan, apply=False,
archive=False)` revalidates policy and snapshot and returns per-item outcomes,
including errors. Failure of one item never broadens scope. It reports candidate
bytes separately from successful deletions; ZIP space usage is not reclaimed
space. The API supports no arbitrary filesystem glob or recursive root option.

Cloud dispatch wraps the complete staging lifetime in `try/finally`, including
packaging, push, polling, import, early dry-run return and exceptions. Fresh
staging names match the exact cleaner prefixes. `--keep-staging` (cloud runner)
and API `keep_staging=True` preserve staging for explicit debugging and print
the path. The live-PID marker is removed when the dispatch exits.

## Validation

Use disposable fixture workspaces and a patched temp root only. Cover forged
plans, traversal, root/outside paths, source and data protections, task-config
overlap, active legacy attachments, official experiments, symlinks/junctions,
recent descendants, two-hour floor, changed plans, dry-run, CLI confirmation,
successful archive round-trip, archive failure, and staging cleanup on success,
failure and dry-run plus explicit retention. Run targeted tests, lint and the
repository smoke gate. Existing dispatch tests must inspect submitted metadata
during dispatch because staging no longer persists after return.

## Limits

Age alone cannot prove that an unmarked historical directory is orphaned.
PID checks protect newly generated active staging, conservatively retaining
markers for live or inaccessible PIDs. Filesystem checks reduce accidental
deletion risk; portable path-based APIs cannot provide an absolute guarantee
against a hostile process replacing entries between a check and an OS call.
No real workspace or system-temp deletion is part of implementation validation.
