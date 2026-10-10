# Kaggle CLI execution contract

Reviewed against official Kaggle CLI documentation/source on 2026-10-10.
Record `kaggle --version` when reporting a cloud failure. Local regression
tests do not establish that a real Kaggle GPU run succeeded.

## Installation and authentication

Current official documentation requires Python 3.11+ and recommends isolated
installation with `pipx install kaggle` or `uv tool install kaggle`. The harness
uses the executable on PATH; Kaggle need not be importable in its Python environment.

```powershell
kaggle --version
kaggle auth login
ai-harness attach --project-root . --task-config configs/task.yaml
ai-harness cloud preflight --project-root . --provider kaggle --config configs/exp001.yaml
```

Login is deliberate, outside tests. Official authentication options include
OAuth, `KAGGLE_API_TOKEN`, `~/.kaggle/access_token`, and legacy `kaggle.json`.
Kernel/dataset IDs require a real account username, not `Kaggle OAuth`.
Username resolution checks `KAGGLE_USERNAME`, credential JSON in
`KAGGLE_CONFIG_DIR` and standard directories, then `kaggle config view`.
Credential presence in preflight is a readiness hint; expired tokens can still
be rejected by the provider command.

Sources: [installation/authentication](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md),
[configuration](https://github.com/Kaggle/kaggle-cli/blob/main/docs/configuration.md).

## Data upload and internet policy

Add cloud settings to the existing task specification:

```yaml
task:
  data_config:
    train_path: dataset/train.csv
    split_file: splits/folds.csv
    license: YOUR_ACTUAL_KAGGLE_LICENSE_ID
  runtime:
    enable_internet: false
```

Select the license that actually applies; the harness does not invent one.
Consent is scoped to task/provider/data fingerprint, and changes to data invalidate it.

```powershell
ai-harness cloud consent --project-root . --provider kaggle --confirm-upload
python -m harness.cloud_runner --project-root . --backend kaggle --config configs/exp001.yaml
```

The consent command authorizes upload. Dataset create/version use
`--dir-mode zip --keep-tabular`; create omits `--public`. Version updates are
attempted only for existing-resource conflicts, not arbitrary auth/validation failures.
CLI output reporting a dataset creation error is rejected even if its exit code
is zero. Upload processing is asynchronous: the harness waits for
`kaggle datasets status owner/dataset-slug` to report `ready` before dispatching
the new dataset. Processing failure or timeout stops the dispatch.

Set `task.data_config.kaggle_dataset_slug: owner/dataset-slug` to use an existing
Kaggle dataset without uploading local data. Local input/split validation still
applies. Without a harness data manifest, the compatibility layout is `dataset/`
and `splits/`, with flat files copied into `dataset/`. Other layouts require a
matching adapter or a harness-created data bundle.

Sources: [dataset commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md),
[metadata and license IDs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md).

## Remote execution

1. Packaging emits `kernel-metadata.json` and `run_kaggle.py`, embedding the
   source/config bundle inside the script. An adjacent ZIP is not a reliable
   code upload channel; declared data uses the separately consented dataset.
2. Title and slug are generated together. Harness slugs use ASCII, are bounded
   to 50 characters with a hash suffix when needed, and include a dispatch ID.
   Polling reads the submitted metadata ID rather than reconstructing it from
   ambient task state. Private kernels and internet disabled are defaults;
   the current generator enables GPU.
3. Code and training execute in a temporary workspace outside the saved output
   directory. The selected mount is `/kaggle/input/<dataset-slug>`; a missing
   mount fails explicitly instead of selecting another dataset. Nested paths
   such as `/kaggle/input/datasets/<owner>/<dataset-slug>` are also resolved by
   the selected slug/owner; ambiguous mounts fail.
4. `mlh-data-bundle.json` lists workspace-relative files. Restoration accepts
   expanded directories or `<directory>.zip` uploads. Only listed files enter
   the run workspace. ZIPs store paths relative to the uploaded directory.
5. Dependency checks use distribution versions, not import names: `scikit-learn`
   and `PyYAML` differ from their Python modules. Pins are checked even for
   installed packages. Missing/mismatched versions with internet disabled fail
   before training. Explicit `task.runtime.enable_internet: true` permits pip;
   failures propagate and versions are rechecked. Adapter wheels install with
   `--no-deps` after dependency verification.
6. Training runs with `python -u -m harness.runner` and explicit project/task/config
   arguments. Fold progress is flushed. All files in the sealed run manifest
   are exported once into the saved output root, with job/task/config identity
   added to the manifest and sealed file hashes preserved.

Exact local pins can differ from Kaggle's image. Offline execution requires an
image that already satisfies the runtime contract and any required bundled
adapter wheel. Pins are not silently relaxed, and internet is not enabled implicitly.
Original code, input data, and nested run artifacts stay outside the saved output
root so the importer receives one manifest rather than duplicate copies.

Sources: [kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md),
[CLI implementation](https://github.com/Kaggle/kaggle-cli/blob/main/src/kaggle/api/kaggle_api_extended.py).

## Polling and recovery

```powershell
kaggle kernels status owner/submitted-kernel-slug
kaggle kernels files owner/submitted-kernel-slug
kaggle kernels output owner/submitted-kernel-slug -p downloaded-output
```

Push success means submission, not completed training. The harness waits through
queued/running states and parses the quoted status field, so words such as
`error` in the kernel name cannot change status. Nonzero CLI exits are errors.
Without a version suffix, status/output refer to the latest run: use the
submitted ID recorded in the job. Current CLI documentation also describes
explicit `owner/kernel/version` references, pagination and output filtering.

Output recovery first uses the public `KaggleApi.kernels_output` method, which
handles pagination and HTTP errors. Its filter selects root-level filenames,
including `metrics.json` and `artifact_manifest.json`; filtering by experiment
ID would exclude those files. If the SDK is absent or fails, partial files are
discarded and the CLI retries with a bounded subprocess timeout. Unit tests mock
both paths, so API authentication cannot run while testing imports.

Import requires exactly one manifest v2, matching job/task/experiment/config
identity, completed full validation, required evidence files and valid hashes.
Failed downloads, partial output, multiple manifests and identity mismatches
are rejected. Existing experiment evidence is never overwritten. Dashboard
recovery passes its attached project root explicitly to the importer.

Source: [kernel commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).

## Regressions and verification

Commits `492211c` and `a150799` added embedding/metadata fixes and corrected Python
literals. Remaining failures covered here: long experiment titles; distribution
name confusion; ignored pins/swallowed pip errors; ZIP paths treated as expanded
directories; duplicate output manifests; status polluted by kernel names;
and imports discovering the wrong workspace. The old bundle test still expected
a standalone `code_bundle.zip` and needed updating too.

`tests/test_kaggle_runtime.py` executes the generated script with the real runner
on a synthetic two-fold dataset, including archived data and one valid output
manifest. Provider operations are mocked: no upload or remote GPU job occurs.
Auth-helper tests mock browser launch and OAuth process creation, preventing
Modal/Kaggle login popups during the full suite.

```powershell
python -m unittest tests.test_kaggle_runtime
python -m unittest discover -s tests -p "test_*.py"
```

Run repository-wide integration tests in a disposable workspace because some
tests mutate configs/artifacts. Live verification still requires an authorized
account, consented data and a bounded remote smoke run.
