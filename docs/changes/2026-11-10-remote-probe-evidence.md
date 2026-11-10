# Preserve remote probe selection and evidence

New cloud_probe helpers normalize explicit fold/sample bounds, generate the remote runner CLI arguments and isolate output under probes/experiment/run. Download verification requires completed probe identity, every manifest file hash, the requested partition/sample bound, disjoint train/validation IDs and exact OOF ID coverage. Verification does not register a candidate or update the leaderboard.

The Modal template accepts an optional probe selection, forwards limits plus skip-leaderboard, and uses the unique probe directory for failed snapshots, remote exports, volume persistence and local download. It preserves all original hashed probe files and verifies before export and after download. Nested output parents are created and downloaded paths must stay under their output directory.

This is not yet exposed by the runner/cloud controller: the existing cloud-probe rejection remains until dispatch, local import and owned-volume cleanup all support probes. No live provider call was made. Helper regressions cover invalid identities/hashes/partitions and generated-code inspection checks argument forwarding and path consistency. Existing11Modal template tests pass.
