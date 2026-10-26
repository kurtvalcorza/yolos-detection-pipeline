"""Final prose pass (Kurt's ruling, 2026-10-26): the prose-rule items flagged on the confirming run are fixed.

Run-status sentences, timing promises and unlabelled expectations are removed or recast
as records of named runs.
Fails on PR #3 head 8379187.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "yolos_detection_colab.ipynb"


def _markdown(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_final_prose_pass():
    md = _markdown(NB)
    for gone in (
        'sample AP50 is already at its ceiling of 1.0, so it cannot rise',
    ):
        assert gone not in md, gone
    for kept in (
        'recorded runs the adapted AP50 on this sample was already at its ceiling of 1.0',
        'if yours is too, `ap` (tighter boxes)',
    ):
        assert kept in md, kept
