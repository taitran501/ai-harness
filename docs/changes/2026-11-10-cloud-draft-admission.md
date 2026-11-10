# Respect offline draft locks before cloud preparation

The offline script checked ready_for_execution only after remote launch, so a draft recipe could reach uploads before refusing model execution. Remote preparation now checks an explicitly declared ready_for_execution lock immediately after effective-config resolution, and requires its offline backend to match the requested cloud backend. Static preview still reports blockers without attempting execution; tasks without this explicit contract are unaffected.

Track1 regression invokes real _prepare_remote_run with its actual draft configuration and replaces cloud preflight with a failure, asserting the specific draft rejection. A ready-but-wrong-backend case is also rejected. No provider operation or model was run. This is an early admission check, not runtime qualification.
