"""Run-derived prose is a record of named runs, never an expectation (Kurt's ruling, 2026-10-26).

Runs: Colab T4 of blobs 59c0ed95 (2026-10-10), 198e104a and 9b56a646 (2026-10-25); Kaggle T4 of the
previous version (2026-09-25).
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "tutorials" / "yolos_detection_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_timing_promise_run_status_and_shared_pattern_are_gone():
    md = _markdown()
    for stale in ("about 3.5 minutes top to bottom", "about 105 s of it the fine-tune",
                  "each completed one pass on a Colab T4", "rose once in each", "The earlier recorded T4 run found one"):
        assert stale not in md, stale


def test_confirming_run_of_9b56a646_is_recorded():
    md = _markdown()
    assert "it went from 1.0890 to 0.3347 and rose once, at epoch 8 (0.4236 → 0.4742)" in md
    assert "`ap` 0.865, 0.8519 and 0.8643" in md
    assert "between 202.3 s and 225.9 s" in md and "102.6 s to 109.0 s" in md
