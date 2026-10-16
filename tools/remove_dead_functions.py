"""AST-safe dead-function remover.

Removes complete function definitions (module-level or class-level,
decorators included) by name. Unlike regex/indent heuristics, this operates
on the parsed ``ast`` so multi-line signatures, nested blocks, and
decorator lines are handled exactly — it cannot truncate a signature or
delete a sibling function.

Usage:
    python tools/remove_dead_functions.py --check <file> name1 name2
    python tools/remove_dead_functions.py <file> name1 name2

Safety guarantees:
- A name with zero definitions -> reported, nothing changed.
- A name with >1 definition in the file -> refused (ambiguous).
- The file must still parse after removal; otherwise the original content
  is kept and the run aborts.
- ``--check`` prints the planned removals without writing.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


def _removable_ranges(tree: ast.AST, name: str) -> list[tuple[int, int]]:
    """Return (start_line, end_line) 1-indexed inclusive ranges for all defs of `name`."""
    ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            start = node.lineno
            for dec in node.decorator_list:
                start = min(start, dec.lineno)
            ranges.append((start, node.end_lineno))
    return ranges


def remove_functions(source: str, names: list[str]) -> tuple[str, list[str], list[str]]:
    """Remove the named functions from `source`.

    Returns (new_source, removed, errors). `new_source` equals `source` when
    `errors` is non-empty or nothing was removed.
    """
    tree = ast.parse(source)
    ranges: list[tuple[int, int]] = []
    errors: list[str] = []
    removed: list[str] = []
    for name in names:
        found = _removable_ranges(tree, name)
        if not found:
            errors.append(f"{name}: not found (0 definitions)")
        elif len(found) > 1:
            errors.append(f"{name}: ambiguous ({len(found)} definitions)")
        else:
            ranges.append(found[0])
            removed.append(name)
    if errors or not ranges:
        return source, removed, errors

    lines = source.splitlines(keepends=True)
    delete = set()
    for start, end in ranges:
        delete.update(range(start - 1, end))
        # absorb the single blank line directly after the block, if any
        if end < len(lines) and not lines[end].strip():
            delete.add(end)
    new_source = "".join(l for i, l in enumerate(lines) if i not in delete)
    try:
        ast.parse(new_source)
    except SyntaxError as e:
        return source, [], [f"post-removal parse failed: {e} (original kept)"]
    return new_source, removed, []


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file", help="Python file to edit")
    ap.add_argument("names", nargs="+", help="function names to remove")
    ap.add_argument("--check", action="store_true", help="dry run: report only")
    args = ap.parse_args(argv)

    path = Path(args.file)
    source = path.read_text(encoding="utf-8")
    new_source, removed, errors = remove_functions(source, args.names)

    for name in args.names:
        err = next((e for e in errors if e.startswith(f"{name}:")), None)
        if err:
            print(f"  SKIP {name}: {err.split(':', 1)[1].strip()}")
        elif name in removed:
            print(f"  REMOVE {name}" + (" (dry run)" if args.check else ""))
        else:
            print(f"  KEEP {name}: no matching definition")
    for e in errors:
        print(f"  ERROR {e}")
    if errors:
        return 1
    if not removed:
        print("Nothing to remove.")
        return 0
    if args.check:
        return 0
    path.write_text(new_source, encoding="utf-8")
    print(f"Wrote {path} ({len(removed)} function(s) removed).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
