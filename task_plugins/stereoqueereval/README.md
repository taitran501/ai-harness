# StereoQueerEval task adapter

This optional adapter keeps StereoQueerEval loading, model construction, output normalization, and task metrics outside the ML-Harness core. It imports the attached competition workspace's src.data, src.models, and src.metrics only after this adapter is installed and selected.

## Install

Install ML-Harness and the adapter into the same Python environment:

~~~powershell
python -m pip install "ml-harness @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin"
python -m pip install "ml-harness-stereoqueereval @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin#subdirectory=task_plugins/stereoqueereval"
ml-harness plugins
~~~

From a local clone of this repository, install in editable mode from its root:

~~~powershell
python -m pip install -e .
python -m pip install -e task_plugins/stereoqueereval
~~~

## Attach to the competition workspace

The competition task code and dataset stay in the task workspace. This adapter expects that workspace to provide the referenced src modules and the task config:

~~~powershell
cd C:/work/stereoqueereval
ml-harness attach --project-root . --task-config configs/task.yaml --plugin stereoqueereval
ml-harness-agent --project-root . context
ml-harness-run --project-root . --config configs/exp001_baseline.yaml --debug --fold 0
~~~

The adapter provides local training-data loading, deterministic grouped validation, task model construction, and local evaluation. If no split file is declared, it creates deterministic GroupKFold assignments from the configured group column.

The official submission builder and validator remain unavailable until organizers publish the submission format and scorer contract. The harness blocks submission packaging for this adapter rather than guessing those requirements.

For Kaggle or Modal runs, install the optional core cloud dependencies, sign in with the provider CLI, run provider preflight, and grant task-scoped consent before uploading local data. The adapter packages itself into a versioned wheel for remote execution; adapter-specific packages belong in the task's runtime requirements. See the [cloud execution guide](../../docs/cloud-execution.md).
