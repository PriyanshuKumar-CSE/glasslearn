import re
from pathlib import Path

import glasslearn


def test_version_matches_pyproject():
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    expected = re.search(r'^version = "(.+)"', pyproject, re.M).group(1)
    assert glasslearn.__version__ == expected


def test_config_round_trip():
    assert glasslearn.get_config()["level"] == "beginner"
    with glasslearn.config_context(level="math", max_points=10):
        assert glasslearn.get_config()["level"] == "math"
        assert glasslearn.get_config()["max_points"] == 10
    assert glasslearn.get_config()["level"] == "beginner"


def test_config_rejects_bad_level():
    import pytest

    with pytest.raises(ValueError, match="level must be one of"):
        glasslearn.set_config(level="expert")


def test_import_does_not_load_sklearn():
    import subprocess
    import sys

    code = "import sys, glasslearn; print('sklearn' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stderr
