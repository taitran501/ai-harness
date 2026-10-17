"""Time each test module once (one-shot measurement, not a repeated test).

Writes <module> <ms> <pytest_tail> lines to stdout, slowest last.
Used to pick the --smoke tier set (target: <10s total).
"""
import subprocess
import sys
import time
from pathlib import Path

repo = Path(__file__).resolve().parents[2]  # tools/timing/ -> repo root
modules = sorted(p.name for p in (repo / "tests").glob("test_*.py"))
rows = []
for m in modules:
    t0 = time.monotonic()
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", f"tests/{m}", "-q", "--no-header"],
            cwd=repo, capture_output=True, text=True, timeout=600,
        )
        tail = (r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr) else "?"
        rows.append((m, int((time.monotonic() - t0) * 1000), tail[:90]))
    except Exception as exc:
        rows.append((m, int((time.monotonic() - t0) * 1000), f"EXC {exc}"))
rows.sort(key=lambda x: x[1])
total = 0
for m, ms, tail in rows:
    total += ms
    print(f"{ms:>7} {m}  {tail}")
print(f"{'TOTAL':>7} {total} ms  ({len(rows)} modules)")
