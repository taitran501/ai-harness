# Agent and phase workflow

ML-Harness coordinates work with an IDE agent. It supplies current task context and phase prompts; the agent uses its own model, tools, and native web search. The dashboard records workspace state and does not host an LLM agent.

## 1. Attach and inspect

From the task workspace, attach the installed harness, then ask the IDE agent to inspect the task contract:

~~~powershell
ml-harness attach --project-root . --task-config configs/task.yaml
ml-harness-agent --project-root . context
ml-harness-agent --project-root . act --action inspect_data_readiness
~~~

If a task-specific adapter is needed, install it first and add its ID with --plugin during attachment. The selected adapter and its available capabilities appear in the context packet.

## 2. Review the prompt pack

Attach creates an editable prompt pack under prompts/ml-harness/. It defines eight ordered phases:

1. setup
2. understand
3. baseline
4. experiment
5. select
6. public_lb
7. freeze
8. finalization

Validate and render the current prompt:

~~~powershell
ml-harness phase validate --project-root .
ml-harness phase status --project-root .
ml-harness phase render --project-root .
~~~

You may request a specific phase with --phase. On the first pass, phases proceed in order. Experiment, candidate selection, and external evaluation can be revisited after new run or score evidence. Baseline recording is gated on an actual validation run and measured metric.

Edit the prompt Markdown and output contracts in the workspace to fit the competition. These are task-owned files and can be versioned with that task's Git repository.

## 3. Run EDA and establish a measured baseline

Run general-purpose EDA and ask the agent to interpret its compact reports together with the official task context:

~~~powershell
ml-harness eda run --project-root .
ml-harness-agent --project-root . context
~~~

In the dashboard's **Agent workspace**, start with the local data paths and files it finds in this workspace. Choose **Profile local data** to inspect one without copying it. If you need another source, **Prepare and copy research prompt** gives your IDE agent the task context and local inventory, then asks it to use native web search for external options only where needed. The agent returns sources and risks for you to review; it does not download or upload data from that prompt.

If you already selected a public Kaggle dataset, open **Import a Kaggle public dataset** and enter its page slug (`owner/dataset-name`). This works for datasets published outside competitions too. ML-Harness downloads it into `dataset/sources/kaggle/`, keeps its source link in the workspace, and does not change the task's train/test configuration. Choose **Profile dataset** to create an EDA report; the next phase prompt includes its workspace path and source URL for the IDE agent. Review the source license and relevance before using it. The import is local; sending data to Kaggle Kernels or Modal remains a separate action.

Use an experiment config already owned by the task workspace. For a new workspace, ml-harness-bootstrap can produce a starter task config, baseline config, and grouped splits from the provided dataset and context. Review the generated contract against official instructions before running it. Then probe and validate the generated baseline config:

~~~powershell
ml-harness-run --project-root . --config configs/exp000_baseline.yaml --debug --fold 0
ml-harness-run --project-root . --config configs/exp000_baseline.yaml
~~~

If bootstrap created the starter agent skill, finish attachment with --force as described in [Installation](installation.md).

Lock the baseline only after a completed run produced the task's primary metric. The bridge and task adapter enforce the baseline and submission capabilities they can verify.

## 4. Continue an experiment loop

Use the Agent Bridge context packet before each change. Keep each experiment hypothesis narrow, run locally or through a supported cloud backend, and compare results against the measured baseline.

~~~powershell
ml-harness-agent --project-root . context
ml-harness-agent --project-root . start-session --goal "Improve the declared validation metric" --max-experiments 3
ml-harness phase render --project-root . --phase experiment
~~~

For cloud prerequisites, account status, and private dataset upload consent, see [Cloud execution](cloud-execution.md).

## 5. Record the agent response and sources

Have the IDE agent save its phase response as Markdown or JSON in the task workspace. If it used web search, save inspected source URLs and citation metadata in a JSON list. Record those outputs:

~~~powershell
ml-harness phase record --project-root . --phase understand --response-file notes/understand.md --sources-file notes/sources.json
ml-harness phase status --project-root .
~~~

Workspace-relative artifact paths can be attached with --artifact; repeat the option for multiple files. The harness records links and metadata under .ml-harness/artifacts/<task-id>/phase_history.json.

## 6. Use the dashboard and finish

~~~powershell
ml-harness-dashboard --project-root .
~~~

Each dashboard process is bound to loopback and one task workspace. Open another process from a different workspace to work on a different task. Freeze and submission operations remain behind user approval. If an adapter lacks an official submission builder or validator, the harness reports that capability gap instead of inventing a contract.
