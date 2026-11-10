import pandas as pd
import pytest
from harness.evaluation import validate_prediction_contract
from harness.task_spec import UniversalTaskSpec


def spec():
    return UniversalTaskSpec.from_dict({'task': {
        'id_column': 'id', 'primary_metric': 'LifeMate Score', 'direction': 'maximize',
        'subtasks': {'tool_calls': {'column': 'label', 'task_type': 'interactive_agent',
                                  'primary_metric': 'LifeMate Score', 'weight': 1}}}})


def test_ragged_episode_calls_preserve_payload():
    calls = [[], [{'name': 'x', 'arguments': {}}], [{'name': 'x'}, {'name': 'y'}]]
    truth = pd.DataFrame({'id': ['a', 'b', 'c'], 'label': [{}, {}, {}]})
    batch = validate_prediction_contract(spec(), truth, {'tool_calls': calls})
    assert batch.labels['tool_calls'] == calls


@pytest.mark.parametrize('calls', [[[]], [[], None]])
def test_interactive_missing_coverage_still_rejected(calls):
    truth = pd.DataFrame({'id': ['a', 'b'], 'label': [{}, {}]})
    with pytest.raises(ValueError, match='Missing'):
        validate_prediction_contract(spec(), truth, {'tool_calls': calls})


def test_interactive_custom_hook_is_authoritative(monkeypatch):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock
    from harness.oof_revalidation import score_predictions
    monkeypatch.setitem(sys.modules, '_interactive_test_hook', SimpleNamespace(
        score=lambda truth, predictions: {'LifeMate Score': 0.75}))
    plugin = SimpleNamespace(evaluate_predictions=Mock(side_effect=AssertionError('generic scorer invoked')))
    truth = pd.DataFrame({'id': ['a', 'b'], 'label': [{}, {}]})
    result = score_predictions(spec(), plugin, SimpleNamespace(project_root=None),
        {'paths': {'metrics_evaluator': '_interactive_test_hook:score'}},
        truth, {'tool_calls': [[], [{'name': 'x'}]]})
    assert result['primary_score'] == 0.75
    plugin.evaluate_predictions.assert_not_called()
