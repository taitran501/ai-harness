# Recover owned model staging before dispatch

Staging receipts now explicitly start with dispatch_started=false. The runner atomically marks an attempted launch before invoking modal run. On earlier failures it retains a local failure receipt and re-verifies all original local model SHA-256 values before deleting its confirmed owned model volume. Create collisions remain unowned, and launch attempts or changed local files block this recovery. Model receipts stay out of the generated code image.

This handles preparation/upload failures only. An interrupted launch or failed running app still requires provider terminal-state confirmation and artifact/log recovery before deletion; other data/cache/artifact volume lifecycles are unchanged. No live calls were used: thirteen related mocked-flow tests pass, including failed upload recovery and preservation after a launch attempt or local corruption.
