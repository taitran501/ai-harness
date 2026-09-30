# Task adapter API

Use the built-in declarative adapter when a task can be described in its task config. Install a separate adapter when the competition requires custom data loading, validation, model, scoring, EDA, or submission behavior.

## Install and select an adapter

Adapters are independent Python distributions discovered through the ml_harness.task_plugins entry-point group. Install an adapter into the same environment as ML-Harness, then select the adapter's entry-point ID when attaching the task workspace.

~~~powershell
python -m pip install "ml-harness-stereoqueereval @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin#subdirectory=task_plugins/stereoqueereval"
ml-harness plugins
ml-harness attach --project-root C:/work/task --task-config configs/task.yaml --plugin stereoqueereval
~~~

See [Installation](installation.md) for core, local editable, and cloud-extra installation instructions.

## Register a new adapter

Implement harness.plugins.TaskPlugin in the adapter distribution. Register its class in that distribution's pyproject.toml:

~~~toml
[project.entry-points."ml_harness.task_plugins"]
my_competition = "my_adapter:MyCompetitionPlugin"
~~~

The entry-point name is the plugin ID accepted by ml-harness attach. A minimal adapter can override only the hooks required by its task:

~~~python
from harness.plugins import TaskPlugin, TaskPluginContext

class MyCompetitionPlugin(TaskPlugin):
    plugin_id = "my_competition"

    def load_training_data(self, context: TaskPluginContext):
        # Load data from context.project_root and return a pandas DataFrame.
        ...
~~~

Available hooks include:

- Training data loading and validation.
- Model construction and prediction normalization.
- Metric evaluation.
- Validation split construction and split checking.
- EDA report extensions.
- Remote runtime manifest, dependency list, and bundle files.
- Official submission validation and building.

The core reports each adapter's implemented capabilities in the Agent Bridge context. Missing adapters and unsupported operations fail with explicit errors. Submission readiness requires task-specific builder and validator hooks.

## Remote runtime

A locally installed adapter is not assumed to exist on Kaggle or Modal. A custom adapter must declare its remote runtime manifest and version. Add its remote packages to the task's runtime requirements and include required task code as workspace-contained bundle files. The dispatcher packages the adapter and code separately from provider credentials and task data.

Local dataset upload requires user consent for the specific provider and current data fingerprint. Remote outputs are imported only after the harness verifies the task, job, experiment, config, and artifact identity.

## StereoQueerEval adapter

The optional adapter lives in task_plugins/stereoqueereval. It loads src.data, src.models, and src.metrics from the attached competition workspace only when selected. The harness repository does not include the competition dataset or task source tree.

The adapter supports local data loading, grouped validation, task model construction, and local scoring. The official submission builder and validator remain unavailable until organizers publish their format and scorer contract; submission packaging therefore fails closed. See the [adapter guide](../task_plugins/stereoqueereval/README.md).
