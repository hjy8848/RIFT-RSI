"""Enforce real import separation rather than directory naming alone."""

import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys


CORE = Path(__file__).resolve().parents[1] / "vast_coevo"
FORBIDDEN = {"usami", "usami_cloudops", "usami_fulfillment", "usami_stayops", "usami_harness", "integrations"}


def test_core_has_no_direct_or_constant_dynamic_domain_imports():
    violations = []
    for path in sorted(CORE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                names = [node.module]
            elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                    names = [node.args[0].value]
                elif isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                    names = [node.args[0].value]
            for name in names:
                if name.split(".")[0] in FORBIDDEN:
                    violations.append(str(path.relative_to(CORE)) + ":" + str(node.lineno) + " -> " + name)
    assert not violations, "Core domain dependencies: " + "; ".join(violations)


def test_every_core_module_imports_when_domain_packages_are_unavailable(tmp_path):
    isolated = tmp_path / "core-only"
    isolated.mkdir()
    shutil.copytree(CORE, isolated / "vast_coevo", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    program = r'''
import importlib
import importlib.abc
import pkgutil
import sys

sys.path.insert(0, sys.argv[1])
blocked = {"usami", "usami_cloudops", "usami_fulfillment", "usami_stayops", "usami_harness", "integrations"}

class DomainUnavailable(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ImportError("domain packages deliberately unavailable: " + fullname)

sys.meta_path.insert(0, DomainUnavailable())
import vast_coevo
for item in pkgutil.walk_packages(vast_coevo.__path__, vast_coevo.__name__ + "."):
    importlib.import_module(item.name)
from vast_coevo.skills import empty_snapshot
assert empty_snapshot("customer").skills == ()
assert not any(name.split(".")[0] in blocked for name in sys.modules)
print("standalone Core import passed")
'''
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run([sys.executable, "-I", "-B", "-c", program, str(isolated)], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "standalone Core import passed" in result.stdout
