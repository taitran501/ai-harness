# Runtime extension contracts

## Custom model factories

The runner injects the explicit current task or plugin context when a custom
factory declares a parameter named `task_spec` or `plugin_context`:

```python
def build_model(task_spec, name, **params):
    return MyModel(task_spec=task_spec, **params)

def build_plugin_model(plugin_context, name, **params):
    return MyModel(task_spec=plugin_context.task_spec, **params)

def build_keyword_model(name, *, task_spec, **params):
    return MyModel(task_spec=task_spec, **params)
```

Legacy factories `factory(name, **params)` keep their existing signature and
receive no extra keyword arguments. For a factory using another context parameter
name, explicitly select the positional contract in the experiment config:

```yaml
model:
  name: custom
  factory: task_models.build_model
  factory_context: plugin_context  # factory(context, name, **params)
  params: {}
```

`factory_context` accepts `task_spec`, `plugin_context`, or `none` (force the legacy
contract). The runner checks signature binding before invocation. A `TypeError`
inside the factory is propagated; it never retries the factory using another
signature. Model parameters cannot override the injected context.

Factories still provide the task model's fit/predict contract. This change does not
add preprocessing or constrain architectures. Task plugins can also build models
through their existing `build_model(context, name, params)` hook.

## Windows action parameters

`ai-harness-agent act --param-file checkpoint.json` reads an absolute path or a path
relative to the selected project root. Files must contain a UTF-8 JSON object; a
UTF-8 BOM is accepted. `--params-file` is an alias. The file and inline `--params`
options are mutually exclusive. Arrays/scalars and invalid JSON are rejected.

PowerShell example:

```powershell
$checkpoint = @{
    summary = 'Weighted loss was inconclusive on full validation.'
    next_action = 'Inspect minority error slices before another loss experiment.'
    open_questions = @('Are the remaining errors annotation noise?')
}
$checkpoint | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 checkpoint.json
ai-harness-agent --project-root . act --action record_research_checkpoint --param-file checkpoint.json
```

## Progress under redirected output

Each fold emits a start message and a completion message for each partition/input
variant. LLM prompting emits fold start/completion messages. These contain row
counts and the primary score, never raw rows. Every message flushes immediately to
stdout and is appended to `run.log` before immutable evidence is sealed.

`ai-harness-run --unbuffered` additionally sets Python stdout/stderr to write through
when their streams support it. Agent Bridge, job-manager Python processes, and
generated Modal/Kaggle training commands use `python -u`. Modal's line-forwarder
also flushes output. This exposes fold boundaries without waiting for training to
finish; it cannot force a custom native training library to emit internal progress.
