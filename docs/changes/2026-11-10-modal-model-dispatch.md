# Wire owned model storage into attached Modal dispatch

Real dispatch now resolves matching local assets and stages selected models into a new owned volume, passes that volume in the generated resource plan, and validates/synchronizes local artifacts before deleting model storage. Receipt files persist outside immutable run directories. Failures retain the model receipt and storage for recovery; aborted-job recovery is still incomplete.

Selected-model dispatch requires task.runtime.modal_environment and attached execution. Environment identity is passed consistently to model SDK operations, dataset CLI operations and modal run. Dataset staging respects HARNESS_TASK_CONFIG. Preview remains provider-free and normal tasks without selected model resources keep their existing route. These changes do not clean task-data/cache/artifact volumes or prove a live end-to-end lifecycle.

Track1 tests verify prerequisite refusal before upload, owned-volume binding without mutating the original plan, and cleanup validation before deletion using mocks. Ten targeted tests, eleven existing framework Modal tests and Ruff checks pass. No live model upload or GPU was invoked. Framework changes remain uncommitted alongside existing work.
