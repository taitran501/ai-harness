#!/usr/bin/env python3
"""Wrapper delegating to ml-harness/tools/journal_tool.py."""
import sys
from pathlib import Path
import runpy

_HARNESS_TOOLS = Path(__file__).resolve().parent.parent / "ml-harness" / "tools"
if str(_HARNESS_TOOLS) not in sys.path:
    sys.path.insert(0, str(_HARNESS_TOOLS))

_TARGET = _HARNESS_TOOLS / "journal_tool.py"

if __name__ == "__main__":
    runpy.run_path(str(_TARGET), run_name="__main__")
else:
    _mod = runpy.run_path(str(_TARGET), run_name=__name__)
    globals().update(_mod)
