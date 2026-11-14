# Use one owned volume for selected-model Modal jobs

The attached owned-model route now uploads consented declared task data beside model files in its new volume; stores original local file hashes in the receipt; and uses that volume for results too. The generated app mounts it once at /vol/models because the installed SDK rejects mounting the same volume twice. HF cache stays on ephemeral container storage, so this route creates no persistent cache volume. Existing shared resources are not adopted or deleted.

Normal local success validation therefore precedes deletion of the job-owned model/data/result volume. Predispatch recovery checks retained original data hashes as well as model hashes. Postlaunch error/interruption recovery is still pending and retains storage. Provider image/app lifecycle is not claimed verified. The static preview now reports explicit environment, draft/backend and runtime-pin blockers.

Sixteen focused mocked/static workspace checks, eleven existing framework Modal tests and Ruff pass. The generated-volume AST check verifies one mount/no persistent cache without importing Modal or running its generated module. Data upload tests verify consent/hash recording and preservation when original local data changes. No live upload/GPU/delete was performed.
