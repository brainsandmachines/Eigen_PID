from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_FILES = tuple(sorted((ROOT / "examples").rglob("*.py")))


@pytest.mark.parametrize(
    "script",
    EXAMPLE_FILES,
    ids=lambda path: str(path.relative_to(ROOT)),
)
def test_ide_file_imports_without_package_context(
    script: Path, tmp_path: Path
) -> None:
    (tmp_path / "eigen_pid.py").write_text(
        "def eigen_pid(*args, **kwargs):\n    return {}\n\n"
        "def conditional_covariance(*args, **kwargs):\n    return None\n",
        encoding="utf-8",
    )
    code = (
        "import runpy, sys; "
        f"sys.path[:0] = [{str(script.parent)!r}, {str(tmp_path)!r}]; "
        f"runpy.run_path({str(script)!r}, run_name='ide_launch_check')"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
