# Exact model transfer preview

Modal static preview now joins local and remote native resource plans by slot and compares definition, bundle, revision and file manifests before describing transfers. It lists only verified-declaration files, source paths, destination paths and declared byte totals; mismatched identities, prerequisites and escaping paths are rejected.

This is planning code only. It neither hashes files nor uploads them and explicitly leaves local_hashes_verified, remote_contents_verified and ownership_verified false. It does not authorize deleting a named volume. Actual model staging and owned-resource cleanup must be implemented and verified separately before a live run.

Track1 regression tests exercise valid planning and five invalid source/mount combinations, plus Modal preview's no-provider boundary. No provider was accessed.
