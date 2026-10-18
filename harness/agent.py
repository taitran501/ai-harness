"""Compatibility alias module forwarding to harness.agent_bridge."""

import sys

from harness import agent_bridge

# Forward module attributes
for attr in dir(agent_bridge):
    if not attr.startswith("__"):
        globals()[attr] = getattr(agent_bridge, attr)

# Ensure sys.modules mapping
sys.modules[__name__] = agent_bridge
