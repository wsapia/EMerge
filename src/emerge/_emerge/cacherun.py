import os
import sys
import ast
import textwrap
from pathlib import Path

# Last Cleanup: 2026-01-04

def _is_plain_check_run(test: ast.AST) -> bool:
    """True only for a bare call to check_run() as the entire if-test."""
    if not isinstance(test, ast.Call) or test.args or test.keywords:
        return False
    f = test.func
    return (
        isinstance(f, ast.Attribute) and f.attr == "cache_run"
    ) or (
        isinstance(f, ast.Name) and f.id == "cache_run"
    )

def get_run_block_str(source: str) -> str:
    """
    Return all source text before the first `if check_run():` / `if *.check_run():`.
    Returns None if no matching if-statement exists.
    """
    tree = ast.parse(source)
    candidates = [n for n in ast.walk(tree) if isinstance(n, ast.If) and _is_plain_check_run(n.test)]
    if not candidates:
        return None
    first = min(candidates, key=lambda n: n.lineno)
    lines = source.splitlines(keepends=True)
    return "".join(lines[: first.lineno - 1])


def get_build_block_str(source: str) -> str | None:
    """Return the code string inside the first `if *.checkrun(...):` block, or None."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Call):
            f = node.test.func
            called = (isinstance(f, ast.Attribute) and f.attr == "cache_build") or \
                     (isinstance(f, ast.Name) and f.id == "cache_build")
            if called and node.body:
                start = node.body[0].lineno - 1           # 0-based start line
                end = node.body[-1].end_lineno            # 1-based end line (inclusive)
                lines = source.splitlines()
                block = "\n".join(lines[start:end])
                return textwrap.dedent(block)
    return None

def entry_script_path():
    main = sys.modules.get("__main__")
    # Most normal runs: python path/to/app.py
    if main and hasattr(main, "__file__"):
        return os.path.abspath(main.__file__)
    # Fallbacks (e.g. python -m pkg.mod)
    if sys.argv and sys.argv[0] not in ("", "-c"):
        return os.path.abspath(sys.argv[0])
    # Interactive sessions, notebooks, or embedded interpreters
    return None


def get_build_section() -> str:
    """ Returns the string section inside the check_build() if statement"""
    name = entry_script_path()

    lines = get_build_block_str(Path(name).read_text())
    return lines

def get_run_section() -> str:
    """Return sthe string section before the check_run() if statement

    Returns:
        str: _description_
    """
    name = entry_script_path()

    lines = get_run_block_str(Path(name).read_text())
    return lines