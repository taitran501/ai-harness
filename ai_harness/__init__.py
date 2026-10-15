"""Transparent alias package forwarding to harness."""

import sys
import importlib
from importlib.machinery import ModuleSpec
import harness

# Forward module attributes
for attr in dir(harness):
    if not attr.startswith("__"):
        globals()[attr] = getattr(harness, attr)

__all__ = getattr(harness, "__all__", [])
__path__ = harness.__path__


class _AliasModuleLoader:
    def __init__(self, target_module):
        self.target_module = target_module

    def create_module(self, spec):
        return self.target_module

    def exec_module(self, module):
        pass


class _AliasFinder:
    def __init__(self, alias_name: str, target_name: str = "harness"):
        self.alias_name = alias_name
        self.target_name = target_name

    def find_spec(self, fullname, path, target=None):
        if fullname.startswith(self.alias_name + "."):
            submodule = fullname[len(self.alias_name) + 1 :]
            real_name = f"{self.target_name}.{submodule}"
            try:
                real_mod = importlib.import_module(real_name)
                return ModuleSpec(fullname, _AliasModuleLoader(real_mod))
            except ImportError:
                return None
        return None


if not any(isinstance(f, _AliasFinder) and getattr(f, "alias_name", None) == "ai_harness" for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder("ai_harness", "harness"))

sys.modules[__name__] = harness
