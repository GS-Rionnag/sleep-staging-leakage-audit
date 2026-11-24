import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.needs_psg
def test_ci_smoke():
    proc = subprocess.run(
        [sys.executable, "-m", "sleepaudit", "all", "--config", "configs/ci.yaml"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr[-3000:]
    summary = pd.read_csv(ROOT / "results" / "ci" / "summary.csv")
    assert set(summary["model"]) == {
        "majority", "bandpower_lr", "bandpower_ctx_lr", "yasa_lgbm", "yasa_pretrained",
    }  # fmt: skip
