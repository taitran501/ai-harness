"""Time each test module once (one-shot measurement, not a repeated test).

Writes <module> <ms> <pytest_tail> lines to stdout, slowest last.
Used to pick the --smoke tier set (target: <10s total).
"""
import subprocess
import sys
import time
from pathlib import Path

def discover_modules(repo):
    modules = sorted(p.relative_to(repo).as_posix() for p in (Path(repo) / "tests").rglob("test_*.py"))
    if not modules:
        raise ValueError(f"No test modules discovered under {Path(repo) / 'tests'}")
    return modules


def main(repo=None):
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[2]
    modules = discover_modules(repo)
    rows = []
    for m in modules:
        t0 = time.monotonic()
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pytest", m, "-q", "--no-header"],
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


if __name__ == "__main__":
    main()
