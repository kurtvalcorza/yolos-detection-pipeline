"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebook:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells turn every user-input problem into `BYODRefusal` and remove the earlier run's results and
  `outputs/` files before they stop.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import types
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
from PIL import Image

from test_review_fixes import _byod_dir, _define_reload_equivalence, _namespace

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/yolos_detection_colab.ipynb"
BYOD_CELL = "yolos_detection-36"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path = NOTEBOOK) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(cell_id: str, path: Path = NOTEBOOK) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(start: str) -> dict:
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(messages):
    namespace = _router_namespace("class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


def test_router_relays_a_refusal_as_one_line_without_a_traceback(capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime([("refusal", "hostname: not readable")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(capsys):
    namespace, runtime = _kernel_runtime([("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(capsys):
    _, runtime = _kernel_runtime([("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace('_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not readable')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


def test_byod_refusal_is_carried_verbatim():
    assert CARRIED in _cell(BYOD_CELL)


def _refusal_line(exc: BaseException) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert "\n" not in line and "Traceback" not in line
    return line


NB_DATA = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
EARLIER_IMAGE = {"image_path": "old", "byod_image": "old", "byod_result": "old"}
EARLIER_DATASET = {"dataset_dir": "old", "byod_records": "old", "byod_manifest": "old", "byod_pipe": "old", "byod_adapted": "old", "byod_export": "old"}
DATASET_FILES = ["byod_yolos_adapter.safetensors", "byod_yolos_result.json"]


def _ns(tmp_path, monkeypatch) -> dict:
    ns = _namespace(tmp_path, monkeypatch)
    ns.update(EPOCHS=1, HOLDOUT=0.25, SEED=0, BATCH_SIZE=2, LEARNING_RATE=1e-4, FREEZE_EARLY_LAYERS=True, **EARLIER_IMAGE, **EARLIER_DATASET)
    _define_reload_equivalence(NB_DATA, ns)
    for name in DATASET_FILES:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    return ns


def _run(ns: dict, **fields: str) -> None:
    source = _cell(BYOD_CELL)
    for name, value in fields.items():
        old = f'{name} = ""' if not name.startswith("USE_") else f"{name} = False"
        source = source.replace(old, f"{name} = {value}", 1)
    exec(compile(source, "<byod>", "exec"), ns)


def _left(tmp_path: Path) -> list:
    return [n for n in DATASET_FILES if (tmp_path / n).exists()]


def _image_cases(tmp_path: Path) -> dict:
    (tmp_path / "hostname").write_text("runtime\n", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "tiny.png")
    return {
        "text file (the scripted /etc/hostname run)": ("hostname", "not an image file Pillow can open (UnidentifiedImageError"),
        "folder": ("folder", "not an image file Pillow can open"),
        "missing path": ("nope.png", "not an image file Pillow can open (FileNotFoundError"),
        "image too small (validate_inputs)": ("tiny.png", "MIN_IMAGE_SIDE"),
    }


@pytest.mark.parametrize("case", ["text file (the scripted /etc/hostname run)", "folder", "missing path", "image too small (validate_inputs)"])
def test_image_problems_are_one_refused_line_and_remove_earlier_results(monkeypatch, tmp_path, case):
    file_name, rule = _image_cases(tmp_path)[case]
    ns = _ns(tmp_path, monkeypatch)
    with pytest.raises(ValueError) as caught:
        _run(ns, USE_BYOD_IMAGE="True", BYOD_IMAGE_PATH=repr(str(tmp_path / file_name)))
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line, line
    assert "run this cell again" in line and "earlier BYOD results of this session and their outputs/ files were removed" in line
    assert not set(EARLIER_IMAGE) & set(ns)
    assert set(EARLIER_DATASET) <= set(ns) and _left(tmp_path) == DATASET_FILES  # the dataset branch is untouched


@pytest.mark.parametrize("field", ["USE_BYOD_IMAGE", "USE_BYOD_DATASET"])
def test_upload_outside_colab_is_one_refused_line(monkeypatch, tmp_path, field):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    ns = _ns(tmp_path, monkeypatch)
    with pytest.raises(ValueError) as caught:
        _run(ns, **{field: "True"})
    line = _refusal_line(caught.value)
    assert line.startswith("Refused: (no file): BYOD_") and "no Colab upload dialog" in line


@pytest.mark.parametrize(("uploaded", "file_name"), [({}, "(no file)"), ({"a.png": b"1", "b.png": b"2"}, "a.png, b.png")])
def test_cancelled_or_multiple_image_upload_is_one_refused_line(monkeypatch, tmp_path, uploaded, file_name):
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: uploaded)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    ns = _ns(tmp_path, monkeypatch)
    with pytest.raises(ValueError) as caught:
        _run(ns, USE_BYOD_IMAGE="True")
    assert _refusal_line(caught.value).startswith(f"Refused: {file_name}: upload exactly one image")


def _dataset_cases(tmp_path: Path) -> dict:
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc" / "hostname").write_text("runtime\n", encoding="utf-8")
    _byod_dir(tmp_path, 4).rename(tmp_path / "few")
    broken = _byod_dir(tmp_path, 9).rename(tmp_path / "broken_json")
    (broken / "annotations.json").write_text("{not json", encoding="utf-8")
    bad_box = _byod_dir(tmp_path, 9).rename(tmp_path / "bad_box")
    entries = json.loads((bad_box / "annotations.json").read_text(encoding="utf-8"))
    entries[0]["boxes"] = [5]
    (bad_box / "annotations.json").write_text(json.dumps(entries), encoding="utf-8")
    missing_image = _byod_dir(tmp_path, 9).rename(tmp_path / "missing_image")
    next(missing_image.glob("*.png")).unlink()
    return {
        "folder without annotations.json (the /etc shape)": ("etc", "annotations.json not found"),
        "fewer than 8 records": ("few", "at least 8 are needed"),
        "annotations.json is not JSON": ("broken_json", "Expecting property name"),
        "a box that is not a list (TypeError)": ("bad_box", "annotations.json has a malformed record"),
        "an image file that is missing (OSError)": ("missing_image", "No such file"),
    }


DATASET_CASES = ["folder without annotations.json (the /etc shape)", "fewer than 8 records", "annotations.json is not JSON",
                 "a box that is not a list (TypeError)", "an image file that is missing (OSError)"]


@pytest.mark.parametrize("case", DATASET_CASES)
def test_dataset_problems_are_one_refused_line_and_remove_earlier_results(monkeypatch, tmp_path, case):
    folder, rule = _dataset_cases(tmp_path)[case]
    ns = _ns(tmp_path, monkeypatch)
    with pytest.raises(ValueError) as caught:
        _run(ns, USE_BYOD_DATASET="True", BYOD_DATASET_DIR=repr(str(tmp_path / folder)))
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {folder}: ") and rule in line, line
    assert not set(EARLIER_DATASET) & set(ns) and _left(tmp_path) == []
    assert set(EARLIER_IMAGE) <= set(ns)


def test_dataset_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    directory = _byod_dir(tmp_path, 12)
    ns = _ns(tmp_path, monkeypatch)

    def broken(_directory):
        raise RuntimeError("internal bug")

    ns["read_detection_records"] = broken
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _run(ns, USE_BYOD_DATASET="True", BYOD_DATASET_DIR=repr(str(directory)))
    assert not getattr(caught.value, "dimer_refusal", False)
    assert set(EARLIER_DATASET) <= set(ns) and _left(tmp_path) == DATASET_FILES  # nothing removed


def test_fine_tune_failure_is_not_a_refusal(monkeypatch, tmp_path):
    directory = _byod_dir(tmp_path, 12)
    ns = _ns(tmp_path, monkeypatch)

    class Broken(ns["YolosDetectionPipeline"]):
        def finetune(self, records, **kwargs):
            raise ValueError("internal training bug")

    ns["YolosDetectionPipeline"] = Broken
    with pytest.raises(ValueError, match="internal training bug") as caught:
        _run(ns, USE_BYOD_DATASET="True", BYOD_DATASET_DIR=repr(str(directory)))
    assert not getattr(caught.value, "dimer_refusal", False)


def test_default_path_and_a_good_dataset_are_unchanged(monkeypatch, tmp_path, capsys):
    ns = _ns(tmp_path, monkeypatch)
    _run(ns)
    out = capsys.readouterr().out
    assert "BYOD image branch is off" in out and "BYOD dataset branch is off" in out and "Refused" not in out
    _run(ns, USE_BYOD_DATASET="True", BYOD_DATASET_DIR=repr(str(_byod_dir(tmp_path, 12))))
    assert ns["byod_reload_check"]["equivalent"] is True


@pytest.mark.skipif(not Path("/etc/hostname").is_file(), reason="needs a POSIX /etc/hostname, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_etc_hostname(monkeypatch, tmp_path):
    """The hosted refusal run sets USE_BYOD_IMAGE=True;BYOD_IMAGE_PATH='/etc/hostname': a text file present in every fresh runtime."""
    ns = _ns(tmp_path, monkeypatch)
    with pytest.raises(ValueError) as caught:
        _run(ns, USE_BYOD_IMAGE="True", BYOD_IMAGE_PATH="'/etc/hostname'")
    assert _refusal_line(caught.value).startswith("Refused: hostname: not an image file Pillow can open (UnidentifiedImageError")
