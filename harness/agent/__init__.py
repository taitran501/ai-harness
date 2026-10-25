"""Shim package for AgentBridge and related utilities.

Supports ergonomic imports like:
    from harness.agent import AgentBridge
    from harness.agent.actions import ActionHandlerMixin, typed_actions
    from harness.agent.cli import main
"""
from harness.agent_bridge import (
    AgentBridge,
    BridgeSession,
    ActionObservation,
    ApprovalTier,
    SessionBudget,
    TypedActionCall,
    TypedActionName,
    _TYPED_ACTION_NAMES,
    ACTION_ALIASES,
    resolve_action_name,
    TYPED_ACTION_SCHEMAS,
    get_action_schema,
    typed_actions,
    HANDLERS,
    _ACTION_REGISTRY,
)

__all__ = [
    "AgentBridge",
    "BridgeSession",
    "ActionObservation",
    "ApprovalTier",
    "SessionBudget",
    "TypedActionCall",
    "TypedActionName",
    "_TYPED_ACTION_NAMES",
    "ACTION_ALIASES",
    "resolve_action_name",
    "TYPED_ACTION_SCHEMAS",
    "get_action_schema",
    "typed_actions",
    "HANDLERS",
    "_ACTION_REGISTRY",
]
