from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).parent / "narration" / "golden"


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        help="Rewrite narration golden files instead of comparing (review the diff!).",
    )


@pytest.fixture
def golden(request):
    """Compare text with golden/<name>; rewrite the file with --update-golden."""
    update = request.config.getoption("--update-golden")

    def check(name: str, text: str) -> None:
        path = GOLDEN_DIR / name
        if update:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            return
        assert path.exists(), f"missing {path}; run: uv run pytest --update-golden"
        assert text == path.read_text(encoding="utf-8")

    return check
