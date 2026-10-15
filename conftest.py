"""Pytest bootstrap for the ai-harness unit/regression suite.

The test suite is CPU-only and never exercises CUDA, but the dev machine has a
GPU so ``set_seed`` would otherwise import the full PyTorch stack (~4-7s cold)
on every runner invocation. We mark torch-seeding as disabled for the whole
session here, keeping the CPU seeded values identical while removing the cost.
Production/GPU runs do not load this file, so their torch/cuDNN seeding is
unaffected (set_seed defaults to seeding when the flag is unset).
"""
import os

os.environ.setdefault("HARNESS_DISABLE_TORCH_SEED", "1")
