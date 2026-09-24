"""End-to-end adaptation on a tiny random-weight YOLOS: no checkpoint, no network.

The tiny model has the same classes and code paths as the pinned checkpoint (YolosForObjectDetection,
only smaller), so these tests exercise the real fine-tuning loss,
evaluation, adapter export and adapter reload. They say nothing about detection quality.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("scipy")

from safetensors.torch import save_file  # noqa: E402

from yolos_detection_pipeline import (  # noqa: E402
    ARTIFACT_FORMAT,
    MODEL_ID,
    MODEL_KEY,
    SIGN_CLASSES,
    YolosDetectionPipeline,
    sign_dataset,
    split_dataset,
)
from yolos_detection_pipeline import pipeline as pipeline_module  # noqa: E402


def _tiny_pipeline(seed: int = 0, class_names=SIGN_CLASSES, layers: int = 6) -> YolosDetectionPipeline:
    from transformers import YolosConfig, YolosForObjectDetection, YolosImageProcessor

    torch.manual_seed(seed)
    config = YolosConfig(
        hidden_size=16,
        num_hidden_layers=layers,
        num_attention_heads=2,
        intermediate_size=16,
        image_size=[64, 64],
        patch_size=16,
        num_detection_tokens=6,
        num_labels=len(class_names),
        id2label=dict(enumerate(class_names)),
        label2id={name: i for i, name in enumerate(class_names)},
    )
    model = YolosForObjectDetection(config).eval()
    processor = YolosImageProcessor(size={"shortest_edge": 64, "longest_edge": 64})
    return YolosDetectionPipeline(
        model=model, processor=processor, device="cpu", class_names=tuple(class_names), source="tiny"
    )


@pytest.fixture(scope="module")
def data():
    records = sign_dataset(6, seed=5)
    return split_dataset(records, train_fraction=0.67, seed=0)


def test_finetune_evaluate_export_and_reload(tmp_path, data):
    train, held = data
    pipe = _tiny_pipeline()
    frozen = pipeline_module.frozen_prefixes_for(6)
    assert frozen == (
        "vit.embeddings.patch_embeddings.",
        "vit.encoder.layer.0.",
        "vit.encoder.layer.1.",
    )
    backbone_before = {k: v.clone() for k, v in pipe.model.state_dict().items() if k.startswith(frozen)}
    assert backbone_before
    baseline = pipe.evaluate(held)
    assert 0.0 <= baseline["ap"] <= 1.0 and baseline["adapted"] is False

    run = pipe.finetune(train, epochs=1, batch_size=2, learning_rate=1e-3, seed=1)
    assert run["epochs"] == 1 and len(run["epoch_losses"]) == 1
    assert torch.isfinite(torch.tensor(run["final_loss"]))
    assert 0 < run["trainable_parameters"] < run["total_parameters"]
    assert run["frozen_prefixes"] == list(frozen) and run["trainable_encoder_layers"] == 4
    after = pipe.model.state_dict()
    assert all(torch.equal(value, after[key]) for key, value in backbone_before.items())

    adapted = pipe.evaluate(held)
    assert set(adapted) >= {"ap", "ap50", "ap75", "per_class_ap50", "estimation"} and adapted["adapted"]

    path = tmp_path / "adapter.safetensors"
    descriptor = pipe.save_artifact(path, notes="tiny")
    assert descriptor["format"] == ARTIFACT_FORMAT and descriptor["bytes"] == path.stat().st_size
    metadata = YolosDetectionPipeline.read_artifact_metadata(path)
    assert metadata["class_names"] == list(SIGN_CLASSES)
    assert metadata["frozen_prefixes"] == list(frozen)

    fresh = _tiny_pipeline()  # same seed: identical frozen layers, different from `pipe` elsewhere
    fresh.apply_artifact(path)
    assert fresh.adapted and fresh.source == "artifact:adapter.safetensors"
    reloaded = fresh.model.state_dict()
    assert all(torch.equal(value, reloaded[key]) for key, value in pipe.model.state_dict().items())
    image = held[0]["image"]
    first = pipe.detect(image, threshold=0.0)["detections"]
    second = fresh.detect(image, threshold=0.0)["detections"]
    assert [(d["label"], round(d["score"], 5)) for d in first] == [
        (d["label"], round(d["score"], 5)) for d in second
    ]


def test_apply_artifact_refuses_mismatched_adapters(tmp_path, data):
    train, _held = data
    pipe = _tiny_pipeline()
    pipe.finetune(train[:2], epochs=1, batch_size=2, seed=1)
    path = tmp_path / "adapter.safetensors"
    pipe.save_artifact(path)

    other_vocabulary = _tiny_pipeline(class_names=("a", "b", "c"))
    with pytest.raises(ValueError, match="class_names differ"):
        other_vocabulary.apply_artifact(path)

    forged = tmp_path / "forged.safetensors"
    tensor = {"class_labels_classifier.layers.2.bias": torch.zeros(4)}
    save_file(tensor, str(forged), metadata={"format": "other", "model_id": MODEL_ID, "model_key": MODEL_KEY})
    with pytest.raises(ValueError, match="artifact format"):
        _tiny_pipeline().apply_artifact(forged)

    partial = tmp_path / "partial.safetensors"
    metadata = {
        "format": ARTIFACT_FORMAT,
        "model_id": MODEL_ID,
        "model_revision": pipeline_module.MODEL_REVISION,
        "model_key": MODEL_KEY,
        "class_names": json.dumps(list(SIGN_CLASSES)),
        "frozen_prefixes": json.dumps(list(pipeline_module.frozen_prefixes_for(6))),
        "base_state_digest": "",
        "notes": "",
    }
    save_file(tensor, str(partial), metadata=metadata)
    with pytest.raises(ValueError, match="missing trainable tensors"):
        _tiny_pipeline().apply_artifact(partial)
