"""Regression tests for the 2026-10-05 notebook review findings (YOS-M1..M2, YOS-m1..m4).

They need only CI's lightweight dependencies: the carried data helpers run on the real drawn dataset, and the
generated notebook's own learner cells are executed with a stand-in detector (no weights, no torch). None of this is
model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import yolos_detection_pipeline.pipeline as pipeline_module
import yolos_detection_pipeline.samples as samples_module
from yolos_detection_pipeline import SIGN_CLASSES, held_out_support, sign_dataset, split_dataset

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"yos_fix_{name}", ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook")
TEMPLATE = _load("notebook_template").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(notebook: dict) -> list[str]:
    return [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]


def _markdown(notebook: dict) -> str:
    return "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")


def _cell(notebook: dict, needle: str) -> str:
    (source,) = [s for s in _code(notebook) if needle in s]
    return source


class _StubPipe:
    """Deterministic stand-in: one detection per call, so in-memory and reloaded outputs can be compared."""

    instances: list[_StubPipe] = []

    def __init__(self, class_names=SIGN_CLASSES, shift: float = 0.0) -> None:
        self.class_names = tuple(class_names)
        self.shift = shift
        self.finetune_calls = 0
        self.device, self.source, self.adapted, self.reinitialised = "cpu", "stand-in", False, ()
        _StubPipe.instances.append(self)

    @classmethod
    def from_pretrained(cls, weights_dir=None, class_names=SIGN_CLASSES, seed=0, **_):
        return cls(class_names)

    @classmethod
    def load_artifact(cls, path, weights_dir=None):
        meta = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(meta["class_names"], shift=meta["shift"])

    def detect(self, image, threshold=0.9):
        return {"detections": [{"label": self.class_names[0], "score": 0.95, "box": [10.0 + self.shift, 10.0, 50.0, 50.0]}]}

    def evaluate(self, records):
        return {"ap": 0.5, "ap50": 0.75, "ap75": 0.5, "per_class_ap50": {name: 0.5 for name in self.class_names}, "n_references": 1}

    def finetune(self, records, **kwargs):
        self.finetune_calls += 1
        self.adapted = True
        return {"freeze_early_layers": kwargs.get("freeze_early_layers"), "trainable_encoder_layers": 4, "trainable_parameters": 1, "total_parameters": 2, "epochs": kwargs.get("epochs"), "batch_size": kwargs.get("batch_size"), "learning_rate": kwargs.get("learning_rate"), "optimizer": "AdamW", "precision": "float32", "device": "cpu"}

    def save_artifact(self, path, notes=""):
        Path(path).write_text(json.dumps({"class_names": list(self.class_names), "shift": self.shift}), encoding="utf-8")
        return {"sha256": "0" * 64, "path": str(path)}


def _namespace(tmp_path: Path, monkeypatch) -> dict:
    monkeypatch.chdir(tmp_path)
    ns = {k: v for k, v in vars(pipeline_module).items() if not k.startswith("__")}
    ns.update({k: v for k, v in vars(samples_module).items() if not k.startswith("__")})
    ns.update(json=json, np=np, Path=Path, Image=Image, OUTPUTS=tmp_path, YolosDetectionPipeline=_StubPipe, WEIGHTS_DIR=tmp_path / "w", threshold=0.9, NOTEBOOK_SOURCE={"repository_revision": "r"})
    _StubPipe.instances = []
    return ns


# ---- YOS-M1: isolated runtime, no restart --------------------------------------------------------------------------


def test_yos_M1_no_kernel_install_no_restart_and_a_hash_lock(notebook: dict) -> None:
    assert "Restart the runtime" not in NOTEBOOK.read_text(encoding="utf-8")
    sources = _code(notebook)
    assert not any("[sys.executable, '-m', 'pip'" in s for s in sources)
    kernel = [s for s in sources if "# dimer: kernel cell" in s]
    assert len(kernel) == 2
    install = next(s for s in kernel if "LOCK_TEXT = r'''" in s)
    for needed in ('"--managed-python"', '"--require-hashes"', '"--only-binary"', "UV_SHA256", "LOCK_SHA256"):
        assert needed in install
    lock = (ROOT / TEMPLATE["lock"]).read_text(encoding="utf-8")
    build.check_lock(build._pins(ROOT, TEMPLATE), lock)


# ---- YOS-M2: the guided layer --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "marker",
    ["## How to use this notebook", "**Who this notebook is for.**", "## The task: Input → Model/System → Output", "## Roadmap", "<strong>Glossary</strong>", "**Predict before running:**", "**What to notice", "<summary>Check your reasoning</summary>", "## 14. Activity: change one thing — train the whole encoder", "## Troubleshooting", "## Conclusion (your notes)"],
)
def test_yos_M2_guided_layer_marker_is_present(notebook: dict, marker: str) -> None:
    assert marker in _markdown(notebook)


def test_yos_M2_infrastructure_cells_are_collapsed(notebook: dict) -> None:
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    first_learner = next(i for i, c in enumerate(code) if c["source"].startswith("import hashlib"))
    assert first_learner >= 5
    for cell in code[:first_learner]:
        assert cell["metadata"].get("cellView") == "form", cell["source"][:60]
    assert _markdown(notebook).count("**Predict before running:**") >= 5


# ---- YOS-m1: the prose matches the recorded numbers ------------------------------------------------------------------


def test_yos_m1_no_statement_contradicts_the_record(notebook: dict) -> None:
    md = _markdown(notebook)
    assert "expected to be near zero" not in md and "Runtimes are not measured" not in md
    assert "172.1 s" in md and "105.5 s" in md
    assert "**What the baseline is, and why it is not zero.**" in md and "0.604" in md
    assert "`EVAL_DETECTION_THRESHOLD` (0.05), the setting average precision is computed at" in md and "**saturated**" in md
    assert "'found_at_eval_threshold'" in _cell(notebook, "new_records = sign_dataset(3, seed=NEW_DATA_SEED)")


# ---- YOS-m2: held-out support per class, named warnings, a BYOD minimum ----------------------------------------------


def test_yos_m2_default_held_out_support_matches_the_review_and_warns_on_speed_limit() -> None:
    records = sign_dataset(40, seed=0)
    _train, held_out = split_dataset(records, train_fraction=0.75, seed=0)
    support = held_out_support(held_out, SIGN_CLASSES)
    assert support["boxes_per_class"] == {"stop-sign": 5, "yield-sign": 10, "speed-limit-sign": 2}
    assert support["warnings"] == ["speed-limit-sign: only 2 held-out box(es); its AP moves in large steps"]
    missing = held_out_support(held_out[:1], [*SIGN_CLASSES, "no-entry-sign"])
    assert "no-entry-sign: no held-out box, so this class is left out of the mean AP" in missing["warnings"]


def _byod_dir(tmp_path: Path, n: int) -> Path:
    directory = tmp_path / "byod"
    directory.mkdir()
    entries = []
    for record in sign_dataset(n, seed=3):
        name = f"{record['id']}.png"
        record["image"].save(directory / name)
        entries.append({"file": name, "boxes": [list(map(float, b)) for b in record["boxes"]], "labels": list(record["labels"])})
    (directory / "annotations.json").write_text(json.dumps(entries), encoding="utf-8")
    return directory


def _byod_source(notebook: dict, directory: Path) -> str:
    return _cell(notebook, "USE_BYOD_DATASET = False").replace("USE_BYOD_DATASET = False", "USE_BYOD_DATASET = True").replace('BYOD_DATASET_DIR = ""', f"BYOD_DATASET_DIR = {str(directory)!r}")


def _define_reload_equivalence(notebook: dict, ns: dict) -> None:
    source = _cell(notebook, "def reload_equivalence(")
    exec(source[source.index("TOLERANCE = 1e-3") : source.index("reload_check = reload_equivalence(")], ns)  # noqa: S102


def test_yos_m2_m3_byod_dataset_branch_checks_support_reload_equivalence_and_writes_a_result(notebook: dict, tmp_path: Path, monkeypatch, capsys) -> None:
    ns = _namespace(tmp_path, monkeypatch)
    ns.update(EPOCHS=1, HOLDOUT=0.25, SEED=0, BATCH_SIZE=2, LEARNING_RATE=1e-4, FREEZE_EARLY_LAYERS=True)
    _define_reload_equivalence(notebook, ns)
    exec(_byod_source(notebook, _byod_dir(tmp_path, 12)), ns)  # noqa: S102 - the notebook's own BYOD cell
    assert ns["byod_reload_check"]["equivalent"] is True and ns["byod_reload_check"]["tolerance"] == 1e-3
    result = json.loads((tmp_path / "byod_yolos_result.json").read_text(encoding="utf-8"))
    assert {"baseline", "adapted", "artifact", "reload_check", "split"} <= set(result)
    assert result["adapted"]["ap"] == 0.5 and result["split"]["held_out_support"]["images"] == 3
    assert "held_out_boxes_per_class" in capsys.readouterr().out


def test_yos_m2_byod_dataset_below_the_minimum_is_refused(notebook: dict, tmp_path: Path, monkeypatch) -> None:
    ns = _namespace(tmp_path, monkeypatch)
    ns.update(EPOCHS=1, HOLDOUT=0.25, SEED=0, BATCH_SIZE=2, LEARNING_RATE=1e-4, FREEZE_EARLY_LAYERS=True)
    with pytest.raises(ValueError, match="at least 8 are needed"):
        exec(_byod_source(notebook, _byod_dir(tmp_path, 4)), ns)  # noqa: S102


def test_yos_m3_reload_equivalence_fails_when_the_reload_differs(notebook: dict, tmp_path: Path, monkeypatch) -> None:
    ns = _namespace(tmp_path, monkeypatch)
    _define_reload_equivalence(notebook, ns)
    image = Image.new("RGB", (64, 64))
    assert ns["reload_equivalence"](_StubPipe(), _StubPipe(), image)["equivalent"] is True
    with pytest.raises(AssertionError):
        ns["reload_equivalence"](_StubPipe(), _StubPipe(shift=0.5), image)


# ---- YOS-m4: re-running the fine-tune cell starts from the verified base ---------------------------------------------


def test_yos_m4_rerunning_the_finetune_cell_trains_a_fresh_adapter(notebook: dict, tmp_path: Path, monkeypatch) -> None:
    ns = _namespace(tmp_path, monkeypatch)
    ns.update(EPOCHS=1, SEED=0, train_records=[], SIGN_CLASSES=SIGN_CLASSES)
    source = _cell(notebook, "run = adapter.finetune(")
    exec(source, ns)  # noqa: S102
    first = ns["adapter"]
    exec(source.replace("FREEZE_EARLY_LAYERS = True", "FREEZE_EARLY_LAYERS = False"), ns)  # noqa: S102
    second = ns["adapter"]
    assert second is not first and first.finetune_calls == 1 and second.finetune_calls == 1
    assert ns["run"]["freeze_early_layers"] is False and isinstance(ns["FINETUNE_SECONDS"], float)
