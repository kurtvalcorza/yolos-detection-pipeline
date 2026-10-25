"""Tier C review findings T115-1, T115-2, T115-S (2026-10-25)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "yolos_detection_colab.ipynb"


def _nb():
    return json.loads(NB.read_text(encoding="utf-8"))


def _markdown():
    md = "\n".join("".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "markdown")
    return " ".join(md.split())


def test_t115_1_numbers_from_the_current_blob_run():
    md = _markdown()
    for stale in ("No run of this revision is recorded yet", "no run of this revision is recorded yet"):
        assert stale not in md, stale
    assert "1.089" not in md and "`ap` 0.904." not in md
    assert "1.0921" in md and "0.865" in md and "`59c0ed95`" in md


def test_t115_2_dat9_overlap_statement():
    md = _markdown()
    assert "**Overlap with the training data:**" in md and "verified by construction" in md


def test_t115_s_spec_items():
    nb = _nb()
    dimer = nb["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    head = "\n".join("".join(c["source"]) for c in nb["cells"][:first_code])
    assert "Notice.** For research, training and evaluation only." in head
    md = _markdown()
    assert "[1] Y. Fang" in md and "`DIMER_ISOLATED_ENV`" in md


def test_prose_holds_across_both_recorded_colab_t4_runs():
    # Tier C prose-mismatch fix (2026-10-25): answers and timing name both Colab T4 runs (59c0ed95, 198e104a).
    md = _markdown()
    for gone in ("has not been timed yet", "it was not re-run for this revision"):
        assert gone not in md, gone
    for quoted in ("0.3352 and to 0.3160", "0.3653 → 0.3760 at epoch 9", "`ap` 0.865 and 0.8519",
                   "202.3 s for blob `198e104a`", "106.9 s and 102.6 s"):
        assert quoted in md, quoted
