"""Every Python example in the README must run."""

import re
from pathlib import Path

import pytest

README = Path(__file__).parents[1] / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)


def test_readme_has_a_quickstart():
    assert BLOCKS, "no python examples found in README.md"
    lines = BLOCKS[0].strip("\n").splitlines()
    assert len(lines) <= 10, "the quickstart should stay within 10 lines"


@pytest.mark.parametrize("index", range(len(BLOCKS)))
def test_readme_example_runs(index, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the quickstart saves a file
    exec(compile(BLOCKS[index], f"README.md block {index}", "exec"), {})
