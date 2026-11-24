import os

import pytest

from sleepaudit.config import data_dir

_PATTERNS = {"needs_labels": "*-Hypnogram.edf", "needs_psg": "*-PSG.edf"}


def _data_present(pattern: str) -> bool:
    root = data_dir() / "sleep-cassette"
    return root.is_dir() and any(root.glob(pattern))


def pytest_collection_modifyitems(config, items):
    require = os.environ.get("SLEEPAUDIT_REQUIRE_DATA") == "1"
    for item in items:
        for marker, pattern in _PATTERNS.items():
            if marker not in item.keywords:
                continue
            if _data_present(pattern):
                continue
            if require:
                # Fail loudly instead of skipping when data is mandatory.
                item.add_marker(pytest.mark.usefixtures("_missing_data_fails"))
            else:
                item.add_marker(pytest.mark.skip(reason=f"{marker}: data not downloaded"))


@pytest.fixture
def _missing_data_fails():
    pytest.fail("required Sleep-EDF data is missing (SLEEPAUDIT_REQUIRE_DATA=1)")
