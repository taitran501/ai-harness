# Keep governed model weights out of Modal code images

Modal mounts declared model volumes separately, but its code-image filter only excluded task data and the local model registry. A model stored inside the task workspace could therefore be uploaded again as code.

The generator now excludes the local directory declared in task model_governance.model_locations for each selected model resource. Paths outside the workspace need no extra exclusion; declaring the entire task root as a model directory is rejected. Sibling model manifests and file-based definitions remain in the code image.

Verification uses the generated filter function without executing the generated module, importing Modal, uploading data or dispatching GPU work. The Track1 regression checks weights excluded and manifests, definition and adapter source retained. This change does not provision a model volume, fix dispatch dry-run, or establish live inference readiness. Existing unrelated template changes are preserved.
