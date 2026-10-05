"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package, and the
model pin/stage/verify cells are produced by the generator from repository sources so they cannot
drift from the package.

This is an `E2E` template, so it must state `run_all` itself, and its default path really adapts:
NOTEBOOK_SPEC 2.2 RUN7/FT2 make a bounded fine-tune mandatory rather than optional for this profile.
Every value a reader can change is a `# @param` form field, and each file-reading BYOD branch has a
location field that bypasses the upload dialog when set (EXE1, EXE2).
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "yolos_detection_pipeline",
    "repo_name": "yolos-detection-pipeline",
    "stem": "yolos_detection",
    "notebook_name": "yolos_detection_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    # NOTEBOOK_SPEC 2.2 §5 (YOS-M1): the pins are installed into an isolated uv environment and every later cell runs in
    # a persistent worker there, so a hosted runtime's preloaded packages never force a restart. The lock is compiled with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "isolated_runtime": True,
    "infrastructure_labels": True,
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "pipeline_class": "YolosDetectionPipeline",
    "weights_key": "yolos-small",
    "modules": [
        "samples.py",
        "pipeline.py",
    ],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "transformers", "scipy", "numpy", "PIL"],
    "title": "YOLOS-Small (COCO) — DIMER object detection and bounded detection fine-tuning (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/yolos-detection-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/yolos-detection-pipeline/blob/main/tutorials/yolos_detection_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-hustvl%2Fyolos--small-ffcc4d?style=flat",
            "https://huggingface.co/hustvl/yolos-small",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-hustvl%2FYOLOS-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/hustvl/YOLOS",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2106.00666-b31b1b.svg", "https://arxiv.org/abs/2106.00666"),
        ("License", "https://img.shields.io/badge/License-Apache--2.0-green.svg", "https://github.com/kurtvalcorza/yolos-detection-pipeline/blob/main/LICENSE"),
    ],
    "capability": "object detection over the COCO classes with YOLOS-Small, a plain Vision Transformer trained as a detector, and a bounded detection fine-tune that re-heads YOLOS onto your own class vocabulary, evaluates it against a held-out split with COCO-style average precision, and exports a reloadable SafeTensors adapter",
    "intro": (
        "YOLOS (You Only Look at One Sequence) Small (`hustvl/yolos-small`) is a plain Vision Transformer (ViT-S/16) turned into a detector "
        "with as few changes as possible: 100 learnable **detection tokens** are appended to the image's patch tokens, the 12-layer encoder "
        "processes them together, and two small MLP heads read one box and one class distribution from each detection token. It is trained "
        "with the same set-prediction loss as DETR, so there is no convolutional backbone, no decoder, no anchor box and no non-maximum "
        "suppression (NMS). At inference the processor resizes the image so its shorter side is 800 px, and every detection token whose best "
        "class probability reaches a caller-owned threshold is returned as an xyxy box in input pixels.\n\n"
        "**The default path really adapts the model:** it re-heads YOLOS onto a three-class traffic sign vocabulary that does not exist in "
        "COCO (`stop-sign`, `yield-sign`, `speed-limit-sign`), measures a pre-adaptation baseline, runs a bounded fine-tune with the patch "
        "embedding and the first 8 encoder layers frozen, scores the result on a held-out split with COCO-style average precision "
        "(AP@[.50:.95] and AP50), runs the adapted model on unseen images, exports the changed tensors as a SafeTensors adapter, and reloads "
        "that adapter onto a fresh copy of the verified base model to check that it reproduces the same detections. Every number you see is "
        "measured in this notebook runtime."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable upstream model "
        "revision; run COCO detection on a drawn scene and score per-object `box_iou`; probe the detector with a blank and a noise image; "
        "build and validate a labelled detection dataset over a new three-class sign vocabulary; split it and measure a pre-adaptation "
        "baseline; run a bounded fine-tune with the DETR set-prediction loss YOLOS is trained with (Hungarian matching, cross-entropy, L1 and generalised IoU); "
        "score the adapted model on the held-out split with COCO-style AP; run inference on unseen images; and export, reload and verify "
        "the adapter."
    ),
    "exclusions": (
        "real-world traffic sign detection (the adaptation dataset is drawn in code, so the model learns these renderings and nothing about "
        "road photographs); COCO benchmark results (the average-precision helper here is a compact implementation without pycocotools area "
        "ranges or crowd handling, and it is run on synthetic data only); full-schedule YOLOS training (the upstream schedule is 200 epochs "
        "of ImageNet-1k pre-training and 150 epochs of COCO fine-tuning; the tutorial runs a few epochs on 30 images); segmentation; video tracking."
    ),
    "guided_opening": [
        (
            "## How to use this notebook\n\n"
            "**Who this notebook is for.** Learners who can open a hosted notebook (Google Colab, Kaggle or Jupyter), run cells in order and read "
            "short Python, and who want to see how a pretrained object detector is checked, adapted to new classes, evaluated and exported "
            "honestly. No prior experience with YOLOS or DETR-style detectors is assumed: each term is explained where it is first needed and "
            "again in the glossary below. A T4 GPU is the documented runtime; a CPU works, more slowly. The **Prerequisites** give the details.\n\n"
            "**Running it.** Choose *Runtime → Run all*. The default path needs no edit, no upload, no account, no token and no runtime restart. "
            "Section 1 builds the isolated environment (the CUDA build of `torch` is the largest download, so it is the slowest step); a second "
            "Run all in the same runtime reuses it.\n\n"
            "**Where the code runs.** The first two code cells run in the notebook kernel: they build the environment and start one Python process "
            "inside it. Every later cell is sent to that process, so the pinned `torch`, `transformers`, `numpy` and `scipy` are used without "
            "replacing anything the hosted runtime had already loaded. Printed output, images and errors come back to the notebook as usual, and "
            "variables persist from cell to cell.\n\n"
            "**Two kinds of cell.** *Infrastructure cells* (Sections 1–3: the isolated install and router, the carried package and the "
            "pinned-checkpoint staging) are collapsed and labelled **Infrastructure**; you may run them without studying their implementation. "
            "*Learner cells* (Sections 4–13) are the machine-learning workflow.\n\n"
            "**Form controls.** Learner cells start with fields Colab renders as a form: `threshold` (Section 4); `N_IMAGES`, `DATASET_SEED`, "
            "`EPOCHS` (Section 6); `HOLDOUT`, `SEED` (Section 7); `LEARNING_RATE`, `BATCH_SIZE`, `FREEZE_EARLY_LAYERS` (Section 8); "
            "`NEW_DATA_SEED` (Section 10); the BYOD switches and paths (Section 13). Leave them at their defaults for the first run. After a change, "
            "select the changed cell and choose *Runtime → Run after*.\n\n"
            "**Section tags.** Each learner heading carries one tag. **[Concept]** — what the model does and why. **[Evaluation practice]** — how "
            "the evidence is produced and how to read it. **[Engineering]** — reproducibility, provenance and packaging.\n\n"
            "**Predict, then check.** Before each principal result a **Predict before running** prompt asks you to commit to an expectation; after "
            "it, **What to notice** describes normal output and a collapsed **Check your reasoning** block gives a worked answer. Write your own "
            "answer first, then open it. The numbers the answers quote come from the recorded Kaggle T4 run of the previous version of this "
            "notebook, with the same defaults (25 September 2026); no run of this revision is recorded yet."
        ),
        (
            "## The task: Input → Model/System → Output\n\n"
            "| Stage | Input | Model / system | Output |\n"
            "|---|---|---|---|\n"
            "| **Detect (COCO)** | one RGB image, shorter side resized to 800 px | YOLOS-Small: ViT-S/16 with 100 detection tokens, COCO heads | boxes (xyxy, input pixels), labels and scores above a caller-owned threshold |\n"
            "| **Validate** | labelled records `{{image, boxes, labels}}` over a vocabulary | the carried `validate_dataset` | a dataset manifest with boxes per class |\n"
            "| **Adapt** | 30 training images over three new sign classes | re-headed class layer; last 4 encoder layers, detection tokens and heads trained with the DETR set-prediction loss | an adapted detector |\n"
            "| **Evaluate and export** | 10 held-out and 3 unseen images | COCO-style AP; then a SafeTensors adapter reloaded on a fresh base | AP before/after, unseen-image results, a verified adapter |\n\n"
            "## Roadmap\n\n"
            "| Section | Tag | What happens | What you read |\n"
            "|---|---|---|---|\n"
            "| 1. Install the pinned runtime | [Engineering] | isolated environment built; later cells routed to it | versions, CUDA |\n"
            "| 2. Package code | [Engineering] | the repository's modules, carried verbatim | nothing to run by hand |\n"
            "| 3. Pin, stage and verify the model | [Engineering] | checkpoint downloaded and digest-checked | the verified files |\n"
            "| 4. COCO detection on a drawn scene | [Concept] | the pretrained detector at threshold 0.9 | boxes, per-object IoU |\n"
            "| 5. Degenerate probes | [Evaluation practice] | blank and noise images | detections that should not exist |\n"
            "| 6. Adaptation dataset | [Evaluation practice] | 40 drawn sign images validated | boxes per class |\n"
            "| 7. Split and baseline | [Evaluation practice] | 30 / 10 split, re-headed model scored | held-out support, baseline AP |\n"
            "| 8. Fine-tune | [Concept] | a bounded DETR-loss fine-tune | losses per epoch |\n"
            "| 9. Evaluate | [Evaluation practice] | held-out AP after adaptation | the change, and why 1.0 is saturated |\n"
            "| 10. Unseen images | [Concept] | three new images at two thresholds | signs found |\n"
            "| 11. Export and reload | [Engineering] | adapter, fresh reload, equivalence check | `equivalent: True` |\n"
            "| 12. Outputs | [Engineering] | JSON, CSV and an annotated image | the file list |\n"
            "| 13. BYOD (optional) | [Engineering] | your image or labelled dataset | the same stages |\n"
            "| 14. Activity (optional) | [Concept] | train the whole encoder and compare | your comparison |\n"
            "| Troubleshooting, Conclusion | — | recovery, your notes | when needed |\n\n"
            "**Fast path.** Short on time? Run all, then read Sections 7, 9 and 10 and the conclusion."
        ),
        (
            "<details>\n<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
            "| Term | Meaning in this notebook |\n"
            "|---|---|\n"
            "| **Detection token** | One of 100 learnable tokens appended to the image patches; each one can become one detected object. |\n"
            "| **No-object class** | An extra class a detection token predicts when it has found nothing; most of the 100 tokens predict it. |\n"
            "| **Threshold** | The minimum score a detection must reach to be returned. Yours to choose; scores are not calibrated probabilities. |\n"
            "| **IoU** | Intersection over union of two boxes: 1 for identical boxes, 0 for disjoint ones. A match usually needs IoU ≥ 0.5. |\n"
            "| **Hungarian matching** | The one-to-one assignment of detection tokens to reference boxes that the training loss is computed on. |\n"
            "| **GIoU** | Generalised IoU, a box loss that still gives a useful signal when boxes do not overlap. |\n"
            "| **`eos_coefficient`** | The weight (0.1) that down-weights the no-object class in the loss, because most tokens predict it. |\n"
            "| **Re-heading** | Replacing the last class layer so the model predicts your classes; it starts random. |\n"
            "| **Frozen layers** | Layers whose weights are not updated during fine-tuning. |\n"
            "| **AP50 / AP75 / AP** | Average precision with a match at IoU ≥ 0.50, at IoU ≥ 0.75, and averaged over IoU 0.50–0.95. |\n"
            "| **Held-out support** | How many reference boxes of each class the held-out split holds; an AP on two boxes is fragile. |\n"
            "| **Saturated** | A test so easy that the score is at its ceiling (1.0), so it cannot show further improvement or rank methods. |\n"
            "| **Adapter** | The SafeTensors file of every tensor the fine-tune changed, loaded onto a fresh copy of the verified base. |\n"
            "| **Isolated environment** | A separate Python built from hash-locked pins, in which every learner cell runs. |\n\n"
            "</details>"
        ),
    ],
    "prerequisites": [
        "- **Runtime:** a fresh **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter). A CUDA GPU such as a Colab or Kaggle T4 is the documented runtime for the fine-tuning stages and is used automatically when present; the notebook also runs on CPU, more slowly. The kernel's own Python version does not matter: Section 1 builds a separate environment with CPython 3.12.12 from the hash-locked pins and every later cell runs there. Timing: the earlier version of this notebook took 172.1 s top to bottom on a Kaggle T4 (25 September 2026), 105.5 s of it the fine-tune, after a first pass that stopped in the old install cell for a restart (437.5 s in all); this version replaces that install with the isolated environment and has not been timed yet. The Linux build of `torch==2.14.0` with its CUDA libraries (several GB) and the ~123 MB checkpoint are the largest downloads.",
        "- **Knowledge:** basic Python and PIL; bounding boxes as xyxy pixel coordinates; intersection-over-union (IoU); and how to read average precision (AP50 and AP@[.50:.95]).",
        "- **Data:** the default path generates everything in code with `samples.py` and downloads no dataset: one 640×480 COCO demonstration scene and a 40-image labelled sign dataset. BYOD is optional and off by default. Expected BYOD input: one image, or a directory holding `annotations.json` — a list of `{'file': 'name.png', 'boxes': [[x0, y0, x1, y1], ...], 'labels': [name, ...]}` objects — and the image files it names. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so; uploaded inputs stay in this runtime and are not sent to any inference API.",
    ],
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the pinned checkpoint, "
        "runs COCO detection on a drawn scene, validates the 40-image sign dataset, splits it into training and held-out parts, measures the "
        "pre-adaptation baseline, **runs the bounded fine-tune**, re-evaluates on the held-out split, detects on unseen images, exports the "
        "adapter, reloads it onto a fresh base model to verify the detections, and writes machine-readable outputs with provenance. The pinned "
        "dependencies go into an isolated environment built from a hash lock, so the kernel's own packages are left alone and **no runtime "
        "restart is needed**; a second Run all reuses it. Nothing is skipped behind a default-off flag, and no clone or DIMER worker is "
        "required (NOTEBOOK_SPEC 2.2 §5, RUN7, FT2). No run of this revision is recorded yet."
    ),
    "byod": (
        "Two optional BYOD branches are included, and both are off by default (`USE_BYOD_IMAGE = False`, `USE_BYOD_DATASET = False`). "
        "`USE_BYOD_IMAGE` runs your own image through the same validation, detection and evaluation-report stages as the sample scene. "
        "`USE_BYOD_DATASET` takes your own labelled detection records through the full adaptation workflow — validate, split, baseline, "
        "fine-tune, evaluate, export, reload with the same equivalence check as the sample, and a result JSON — under NOTEBOOK_SPEC 2.2 DAT14; it "
        "needs at least 8 labelled records and warns, by class name, when the held-out split has fewer than 3 boxes of a class. Set "
        "`BYOD_IMAGE_PATH` or `BYOD_DATASET_DIR` to read from a location without an upload dialog (EXE2)."
    ),
    "cells": [
        # ---------------------------------------------------------------- 4. COCO scene
        {
            "md": (
                "## 4. What the pretrained detector does on a drawn scene · [Concept]\n\n"
                "Before adapting anything, inspect the model you start from. The carried `samples` module draws a deterministic street scene with "
                "four objects and their reference boxes: a **stop sign**, a **traffic light**, an analogue **clock**, and an orange **sports ball**. "
                "All four are COCO classes.\n\n"
                "The detection threshold is a **caller-owned request parameter**, not a pipeline constant. Each score is the query's largest class "
                "probability from a **softmax over the classes and a no-object class, not a calibrated** probability for your images. The default "
                "`0.9` is the value the pinned model README uses, and it is passed explicitly on every call.\n\n"
                "The scene is drawn, not photographed, so a miss here is a finding about renderings, not about photographs. The evaluation report "
                "records every miss with `box_iou = 0.0`. COCO mean average precision needs a labelled image set; on one scene the verdict is `sample-sanity`.\n\n"
                "**Predict before running:** at threshold 0.9, how many of the four drawn objects will the COCO detector find?"
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "OUTPUTS = Path('outputs')\n\n"
                'threshold = 0.9  # @param {{type:"number"}}\n\n'
                "print({{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_DETECTIONS': MAX_DETECTIONS,\n"
                "       'label_slots': len(LABELS), 'unannotated_slots': len(UNANNOTATED_LABEL_IDS), 'DETECTION_THRESHOLD': DETECTION_THRESHOLD}})\n\n"
                "scene, references = tutorial_scene()\n"
                "buffer = io.BytesIO()\n"
                "scene.save(buffer, format='PNG')\n"
                "print({{'sample_kind': 'synthetic', 'size': list(scene.size), 'sha256': hashlib.sha256(buffer.getvalue()).hexdigest()[:16],\n"
                "       'references': {{label: len(boxes) for label, boxes in references.items()}}}})\n\n"
                "input_manifest = validate_inputs(scene, threshold=threshold, names=['tutorial-scene'])\n"
                "try:\n"
                "    validate_inputs(scene, threshold=1.5)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'probe': 'threshold=1.5', 'rejected': str(exc)}})\n"
                "print({{'verdict': input_manifest['verdict'], 'findings': input_manifest['findings'], 'inputs': input_manifest['inputs']}})\n\n"
                "coco_result = pipe.detect(scene, threshold=threshold)\n"
                "for det in coco_result['detections']:\n"
                "    print(f\"{{det['label']:>14s}} {{det['score']:.3f}}  [{{', '.join(f'{{v:.0f}}' for v in det['box'])}}]\")\n\n"
                "coco_report = evaluation_report(coco_result, references, sample_kind='synthetic')\n"
                "print({{'verdict': coco_report['verdict'], 'n_detections': coco_report['n_detections']}})\n"
                "for metric in coco_report['metrics']:\n"
                "    print(f\"  {{metric['reference']:>18s}}  box_iou {{metric['value']:.3f}}  same-label detections {{metric['n_detected_same_label']}}\")\n"
                "hits = sum(1 for m in coco_report['metrics'] if m['value'] >= 0.5)\n"
                "print(f'{{hits}}/{{len(coco_report[\"metrics\"])}} drawn objects matched at IoU >= 0.5')\n"
                "scene"
            ),
        },
        # ---------------------------------------------------------------- 5. Degenerate inputs
        {
            "md": (
                "**What to notice (Section 4):** the detections printed with their scores, one `box_iou` per drawn object, and the count matched at IoU ≥ 0.5.\n\n"
                "<details><summary>Check your reasoning</summary>The earlier recorded T4 run found one: the stop sign, at score 0.9995 and IoU 0.912. "
                "The traffic light, clock and ball were not returned at 0.9. They are drawings, not photographs, and a detector trained on photographs "
                "can be unsure of them; lowering the threshold would return more boxes and more false positives. The threshold is yours to choose, and "
                "a miss on a drawing says nothing about photographs.</details>\n\n"
                "## 5. Degenerate input probes: blank canvas and noise · [Evaluation practice]\n\n"
                "Ask a detector about structure-free input before trusting it. A model that returns confident boxes on a blank canvas or on "
                "uniform noise will return them on empty real frames too. The cell counts detections on both images at the default threshold "
                "and at the evaluation threshold `EVAL_DETECTION_THRESHOLD` (0.05) that average precision is computed at.\n\n"
                "**What to look for:** any detection above the default threshold on these two images is a false positive by construction."
            ),
            "code": (
                "degenerate = {{}}\n"
                "for name, image in (('blank', blank_scene()), ('noise', noise_scene(0))):\n"
                "    standard = pipe.detect(image, threshold=threshold)['detections']\n"
                "    lenient = pipe.detect(image, threshold=EVAL_DETECTION_THRESHOLD)['detections']\n"
                "    degenerate[name] = {{\n"
                "        'at_default_threshold': len(standard),\n"
                "        'at_evaluation_threshold': len(lenient),\n"
                "        'top': [(d['label'], round(d['score'], 3)) for d in lenient[:3]],\n"
                "    }}\n"
                "print(json.dumps(degenerate, indent=2))"
            ),
        },
        # ---------------------------------------------------------------- 6. Dataset & Validation
        {
            "md": (
                "## 6. Labelled adaptation dataset and validation · [Evaluation practice]\n\n"
                "Suppose your task needs sign classes that COCO does not have. `sign_dataset` draws a deterministic 40-image dataset over "
                "`SIGN_CLASSES`: `stop-sign`, `yield-sign` and `speed-limit-sign`.\n\n"
                "**Keep the two vocabularies apart.** COCO has `stop sign` (with a space). The adaptation vocabulary uses `stop-sign` (hyphenated), "
                "`yield-sign` and `speed-limit-sign`. They are different class identities, and the adapted model answers in the new names only.\n\n"
                "`validate_dataset` checks every record before any model runs: the record keys, the image size ceilings, that every box is finite, "
                "non-empty and inside its image, and that every label is in the vocabulary. It returns a dataset manifest with the box count per "
                "class and a finding for any class that has no box."
            ),
            "code": (
                'N_IMAGES = 40  # @param {{type:"integer"}}\n'
                'DATASET_SEED = 0  # @param {{type:"integer"}}\n'
                'EPOCHS = 10  # @param {{type:"integer"}}\n\n'
                "records = sign_dataset(N_IMAGES, seed=DATASET_SEED)\n"
                "dataset_manifest = validate_dataset(records, SIGN_CLASSES, epochs=EPOCHS)\n"
                "print(json.dumps({{k: v for k, v in dataset_manifest.items() if k != 'schema'}}, indent=2))\n\n"
                "preview = Image.new('RGB', (480, 320))\n"
                "for index, record in enumerate(records[:6]):\n"
                "    preview.paste(record['image'].resize((160, 160)), (160 * (index % 3), 160 * (index // 3)))\n"
                "preview"
            ),
        },
        # ---------------------------------------------------------------- 7. Split & Baseline
        {
            "md": (
                "## 7. Split, re-head, and measure the pre-adaptation baseline · [Evaluation practice]\n\n"
                "The dataset is split at random into a training part (75%, 30 images) and a held-out part (25%, 10 images). A random split is "
                "valid here because every image is drawn independently; records that share a photograph or a camera session must be split by that "
                "group instead. The held-out images are never shown to the optimizer, and no hyperparameter is selected on them.\n\n"
                "`from_pretrained(class_names=SIGN_CLASSES)` loads the verified checkpoint and replaces only the last layer of the three-layer "
                "class head (`class_labels_classifier.layers.2`, now 3 classes plus no-object) with a freshly initialised layer. The encoder, the "
                "detection tokens, the first two class-head layers and the box head keep their COCO weights.\n\n"
                "**Held-out support.** `held_out_support` prints the number of held-out boxes per class and warns, by name, about any class with fewer "
                "than 3 boxes or none: `average_precision` leaves a class with no held-out box out of its mean, and the AP of a class with two boxes "
                "moves in steps of 0.5. On the default split the held-out images hold 10 yield, 5 stop and 2 speed-limit boxes, so the speed-limit "
                "sign gets that warning.\n\n"
                "**What the baseline is, and why it is not zero.** The re-headed class layer is random, so it knows nothing about the new classes; "
                "this number is the floor the fine-tune has to beat, not a property of YOLOS. It is not near zero, for two reasons. The box head and "
                "the detection tokens keep their COCO training, so boxes still land on objects; and AP is computed at the low evaluation threshold "
                "`EVAL_DETECTION_THRESHOLD` (0.05), so every box is ranked, and with very few references a near-random ranking can score well by "
                "luck.\n\n"
                "**Predict before running:** will the baseline AP50 be near 0, near 0.2, or near 1? Which class might score highest by chance, and why?"
            ),
            "code": (
                'HOLDOUT = 0.25  # @param {{type:"number"}}\n'
                'SEED = 0  # @param {{type:"integer"}}\n\n'
                "train_records, held_out = split_dataset(records, train_fraction=1.0 - HOLDOUT, seed=SEED)\n"
                "overlap = {{r['id'] for r in train_records}} & {{r['id'] for r in held_out}}\n"
                "assert not overlap, f'split leaked records: {{sorted(overlap)}}'\n"
                "print({{'train': len(train_records), 'held_out': len(held_out),\n"
                "       'train_boxes': sum(len(r['boxes']) for r in train_records),\n"
                "       'held_out_boxes': sum(len(r['boxes']) for r in held_out)}})\n"
                "support = held_out_support(held_out, SIGN_CLASSES)\n"
                "print({{'held_out_boxes_per_class': support['boxes_per_class']}})\n"
                "for warning in support['warnings']:\n"
                "    print('warning:', warning)\n\n"
                "adapter = YolosDetectionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=SIGN_CLASSES, seed=SEED)\n"
                "print({{'class_names': list(adapter.class_names), 'device': adapter.device, 'reinitialised': list(adapter.reinitialised)}})\n\n"
                "baseline = adapter.evaluate(held_out)\n"
                "print(json.dumps({{'ap': round(baseline['ap'], 4), 'ap50': round(baseline['ap50'], 4),\n"
                "                  'per_class_ap50': {{k: f\"{{v:.4f}} on {{support['boxes_per_class'][k]}} box(es)\" for k, v in baseline['per_class_ap50'].items()}},\n"
                "                  'n_references': baseline['n_references']}}, indent=2))"
            ),
        },
        # ---------------------------------------------------------------- 8. Bounded Fine-Tuning
        {
            "md": (
                "**What to notice (Section 7):** the split sizes, the held-out boxes per class with one warning, and the baseline `ap`, `ap50` and "
                "per-class AP50 with their support.\n\n"
                "<details><summary>Check your reasoning</summary>The earlier recorded run measured `ap50` 0.212 and `ap` 0.112, with per-class AP50 "
                "stop 0.0, yield 0.033 and speed-limit **0.604**. The speed-limit sign scores highest because it has only two held-out boxes: with so "
                "few references, a box that happens to rank first can lift its AP to 0.5 or more. That is luck in a tiny sample, not knowledge — "
                "which is why the support is printed next to every AP.</details>\n\n"
                "## 8. Bounded detection fine-tuning · [Concept]\n\n"
                "This cell runs the real adaptation step in this runtime. It is gradient fine-tuning with the DETR loss YOLOS was trained with: each "
                "image's 100 detection tokens are matched one-to-one to its reference boxes by the Hungarian algorithm, and the matched tokens are trained with cross-entropy "
                "over the classes plus no-object (the no-object term is down-weighted by the config's `eos_coefficient` 0.1), an L1 box term and a "
                "generalised-IoU box term.\n\n"
                "- **The early layers are frozen.** The patch embedding and encoder layers 0–7 keep their COCO values; the last 4 encoder layers, "
                "the detection tokens, the position embeddings, the final layer norm and both heads are trained. The cell prints the trainable and "
                "total parameter counts.\n"
                "- **Schedule:** `EPOCHS` epochs of AdamW at learning rate `1e-4` (weight decay `1e-4`), batch size 4, gradient-norm clipping at 0.1, "
                "float32, seed `SEED`.\n\n"
                "**Every run of this cell starts from the verified base.** The cell first rebuilds `adapter` with "
                "`from_pretrained(class_names=SIGN_CLASSES, seed=SEED)`: the same COCO weights and, under the same seed, the same re-headed class "
                "layer the Section 7 baseline measured. Re-running it after changing a field therefore trains once from the start, never on top of "
                "an already adapted model, and the comparison with the baseline stays fair.\n\n"
                "**Read the loss as optimisation evidence only.** A falling loss says the optimizer is fitting the training images; the held-out average precision in the next section is the task evidence.\n\n"
                "**Predict before running:** over 10 epochs, will the loss fall every epoch? Roughly how far?"
            ),
            "code": (
                'LEARNING_RATE = 1e-4  # @param {{type:"number"}}\n'
                'BATCH_SIZE = 4  # @param {{type:"integer"}}\n'
                'FREEZE_EARLY_LAYERS = True  # @param {{type:"boolean"}}\n\n'
                "import time\n\n"
                "# YOS-m4: start every run of this cell from the verified base and the same seeded head as the baseline.\n"
                "adapter = YolosDetectionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=SIGN_CLASSES, seed=SEED)\n"
                "finetune_started = time.perf_counter()\n"
                "run = adapter.finetune(\n"
                "    train_records,\n"
                "    epochs=EPOCHS,\n"
                "    batch_size=BATCH_SIZE,\n"
                "    learning_rate=LEARNING_RATE,\n"
                "    seed=SEED,\n"
                "    freeze_early_layers=FREEZE_EARLY_LAYERS,\n"
                "    progress=lambda row: print(f\"epoch {{row['epoch']}}/{{row['epochs']}}  loss {{row['loss']:.4f}}\"),\n"
                ")\n"
                "print(json.dumps({{key: run[key] for key in ('freeze_early_layers', 'trainable_encoder_layers', 'trainable_parameters', 'total_parameters', 'epochs',\n"
                "                                         'batch_size', 'learning_rate', 'optimizer', 'precision', 'device')}}, indent=2))\n"
                "FINETUNE_SECONDS = round(time.perf_counter() - finetune_started, 1)\n"
                "print({{'finetune_seconds': FINETUNE_SECONDS}})"
            ),
        },
        # ---------------------------------------------------------------- 9. Evaluate Held-Out
        {
            "md": (
                "**What to notice (Section 8):** one loss line per epoch, then the configuration with the trainable and total parameter counts.\n\n"
                "<details><summary>Check your reasoning</summary>Mostly, but not strictly. In the earlier recorded run the loss went from 1.089 to "
                "0.251 and rose once (0.549 → 0.556 at epoch 6): with 30 images in batches of 4, single epochs are noisy. A falling training loss "
                "says the model fits these 30 images; it does not say it finds signs it has not seen.</details>\n\n"
                "## 9. Evaluate on the held-out split · [Evaluation practice]\n\n"
                "`evaluate` re-runs on the same held-out images with the same thresholds as the baseline, so the two rows are comparable. `ap50` "
                "is average precision at IoU 0.50; `ap` averages AP over the ten IoU thresholds 0.50–0.95, so it also rewards tight boxes. Reading "
                "only `ap50` hides loose boxes; reading only `ap` hides whether objects were found at all. These are tutorial metrics from one pass over 10 synthetic images, with no dispersion estimate. Each per-class AP50 is printed with "
                "its number of held-out boxes. Every run of Sections 8–9 is appended to `run_history`, so after the activity (Section 14) the "
                "settings and results print side by side.\n\n"
                "**Predict before running:** after fine-tuning, will held-out AP50 be below 0.5, near 0.8, or 1.0? Will `ap` be as high as `ap50`?"
            ),
            "code": (
                "adapted = adapter.evaluate(held_out)\n"
                "print(f\"{{'metric':<8s}} {{'baseline':>10s}} {{'adapted':>10s}} {{'change':>10s}}\")\n"
                "for key in ('ap', 'ap50', 'ap75'):\n"
                "    print(f\"{{key:<8s}} {{baseline[key]:>10.4f}} {{adapted[key]:>10.4f}} {{adapted[key] - baseline[key]:>+10.4f}}\")\n"
                "print('per-class AP50:', {{k: f\"{{v:.4f}} on {{support['boxes_per_class'][k]}} box(es)\" for k, v in adapted['per_class_ap50'].items()}})\n"
                "run_history = globals().get('run_history', [])\n"
                "run_history.append({{'freeze_early_layers': FREEZE_EARLY_LAYERS, 'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'ap': round(adapted['ap'], 4), 'ap50': round(adapted['ap50'], 4), 'finetune_seconds': FINETUNE_SECONDS}})\n"
                "for number, row in enumerate(run_history, start=1):\n"
                "    print(f'run {{number}}:', row)"
            ),
        },
        # ---------------------------------------------------------------- 10. New-data inference
        {
            "md": (
                "**What to notice (Section 9):** the three rows (`ap`, `ap50`, `ap75`) with the change from the baseline, and the per-class AP50 "
                "with its support.\n\n"
                "<details><summary>Check your reasoning</summary>The earlier recorded run reached `ap50` 1.0, `ap75` 1.0 and `ap` 0.904. "
                "`ap` is lower because it also demands tight boxes at IoU up to 0.95. An AP50 of 1.0 here is **saturated**: on 10 drawn images from "
                "the generator the model was trained on, with every box ranked down to score 0.05, the test is too easy to show how much further the "
                "model could go, and it is not evidence that the model finds signs in general. Section 10 tests that directly.</details>\n\n"
                "## 10. Inference on unseen images · [Concept]\n\n"
                "Three new images come from a seed the dataset never used (`NEW_DATA_SEED = 99`). The adapted pipeline detects the new sign classes, and each reference box is compared with the best same-label detection by IoU. "
                "Each image is scored twice: at your detection `threshold` (0.9), the setting a user would deploy, and at "
                "`EVAL_DETECTION_THRESHOLD` (0.05), the setting average precision is computed at.\n\n"
                "**Predict before running:** the held-out AP50 was (probably) 1.0. At threshold 0.9, how many of the five signs on the three new "
                "images will be found?"
            ),
            "code": (
                'NEW_DATA_SEED = 99  # @param {{type:"integer"}}\n\n'
                "new_records = sign_dataset(3, seed=NEW_DATA_SEED)\n"
                "new_data_rows = []\n"
                "for record in new_records:\n"
                "    out = adapter.detect(record['image'], threshold=threshold)\n"
                "    lenient = adapter.detect(record['image'], threshold=EVAL_DETECTION_THRESHOLD)\n"
                "    ious, ious_lenient = [], []\n"
                "    for box, label in zip(record['boxes'], record['labels'], strict=True):\n"
                "        same_label = [d for d in out['detections'] if d['label'] == label]\n"
                "        ious.append(round(max((box_iou(d['box'], box) for d in same_label), default=0.0), 3))\n"
                "        same_label_lenient = [d for d in lenient['detections'] if d['label'] == label]\n"
                "        ious_lenient.append(round(max((box_iou(d['box'], box) for d in same_label_lenient), default=0.0), 3))\n"
                "    row = {{'id': record['id'], 'truth': record['labels'],\n"
                "           'detections': [(d['label'], round(d['score'], 3)) for d in out['detections']], 'same_label_iou': ious,\n"
                "           'same_label_iou_at_eval_threshold': ious_lenient}}\n"
                "    new_data_rows.append(row)\n"
                "    print(json.dumps(row))\n"
                "found = sum(iou >= 0.5 for row in new_data_rows for iou in row['same_label_iou'])\n"
                "found_lenient = sum(iou >= 0.5 for row in new_data_rows for iou in row['same_label_iou_at_eval_threshold'])\n"
                "n_truth = sum(len(row['truth']) for row in new_data_rows)\n"
                "print({{'found_at_threshold': f'{{found}}/{{n_truth}} at {{threshold}}', 'found_at_eval_threshold': f'{{found_lenient}}/{{n_truth}} at {{EVAL_DETECTION_THRESHOLD}}'}})"
            ),
        },
        # ---------------------------------------------------------------- 11. Export, reload & verify
        {
            "md": (
                "**What to notice (Section 10):** the two counts, at 0.9 and at 0.05.\n\n"
                "<details><summary>Check your reasoning</summary>The earlier recorded run found 3 of the 5 signs at 0.9, beside a held-out AP50 of 1.0. "
                "The two numbers measure different things. AP ranks every box above 0.05 and asks whether the right ones come first; detection at 0.9 "
                "keeps only boxes the model is very sure of. A model can rank correctly yet be unsure, so it scores perfect AP and still misses signs at "
                "a strict threshold. Compare the count at 0.05: if more signs are found there, the model sees them but with low scores. Together with "
                "a saturated 10-image test, this is why AP50 1.0 is not evidence of transfer.</details>\n\n"
                "## 11. Adapter export, fresh reload, and equivalence check · [Engineering]\n\n"
                "`save_artifact` writes `outputs/yolos_adapter.safetensors`: every tensor the fine-tune could change, plus a metadata header naming "
                "the base model, its pinned revision, the base `model.safetensors` SHA-256, the class names and the frozen prefixes. The frozen "
                "patch embedding and the frozen encoder layers are left out because they equal the verified base snapshot.\n\n"
                "`load_artifact` then builds a **fresh** pipeline from the verified base snapshot, loads the adapter tensors onto it, and refuses "
                "an adapter whose format, base identity, base digest or tensor set does not fit. The cell compares the reloaded detections with the "
                "in-memory model's on an unseen image, with a stated tolerance: loading succeeding is not the check, reproducing the detections is."
            ),
            "code": (
                "artifact_path = OUTPUTS / 'yolos_adapter.safetensors'\n"
                "descriptor = adapter.save_artifact(artifact_path, notes='YOLOS-Small sign adaptation tutorial adapter')\n"
                "print(json.dumps(descriptor, indent=2))\n\n"
                "reloaded = YolosDetectionPipeline.load_artifact(artifact_path, weights_dir=WEIGHTS_DIR)\n"
                "print({{'reloaded_source': reloaded.source, 'adapted': reloaded.adapted, 'class_names': list(reloaded.class_names)}})\n\n"
                "TOLERANCE = 1e-3\n\n\n"
                "def reload_equivalence(original, reloaded_pipe, image, tolerance=TOLERANCE):\n"
                "    # Loading succeeding is not the check; reproducing the in-memory detections is (also used by the BYOD branch).\n"
                "    det_orig = original.detect(image, threshold=EVAL_DETECTION_THRESHOLD)['detections']\n"
                "    det_reloaded = reloaded_pipe.detect(image, threshold=EVAL_DETECTION_THRESHOLD)['detections']\n"
                "    assert len(det_orig) == len(det_reloaded), f'{{len(det_orig)}} detections in memory, {{len(det_reloaded)}} after reload'\n"
                "    for d1, d2 in zip(det_orig, det_reloaded, strict=True):\n"
                "        assert d1['label'] == d2['label']\n"
                "        assert np.allclose(d1['box'], d2['box'], atol=tolerance)\n"
                "        assert abs(d1['score'] - d2['score']) <= tolerance\n"
                "    return {{'detections_compared': len(det_orig), 'tolerance': tolerance, 'equivalent': True}}\n\n\n"
                "reload_check = reload_equivalence(adapter, reloaded, new_records[0]['image'])\n"
                "print(reload_check)"
            ),
        },
        # ---------------------------------------------------------------- 12. Outputs & provenance
        {
            "md": (
                "## 12. Write machine-readable outputs and provenance · [Engineering]\n\n"
                "The cell writes:\n"
                "- `outputs/yolos_detection_input_manifest.json`\n"
                "- `outputs/yolos_detection_evaluation_report.json`\n"
                "- `outputs/yolos_detection_result.json` (identity, runtime versions, device, dataset manifest, split, baseline and adapted metrics, "
                "fine-tuning configuration, new-data rows, adapter descriptor and reload check)\n"
                "- `outputs/yolos_detection_detections.csv`\n"
                "- `outputs/yolos_detection_annotated.png`\n"
                "- `outputs/yolos_adapter.safetensors` (written in Section 11)"
            ),
            "code": (
                "import csv\n"
                "from PIL import ImageDraw\n\n"
                "with open(OUTPUTS / '{stem}_input_manifest.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(input_manifest, f, indent=2)\n\n"
                "with open(OUTPUTS / '{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(coco_report, f, indent=2)\n\n"
                "annotated = scene.copy()\n"
                "draw = ImageDraw.Draw(annotated)\n"
                "for det in coco_result['detections']:\n"
                "    x0, y0, x1, y1 = det['box']\n"
                "    draw.rectangle([x0, y0, x1, y1], outline='red', width=3)\n"
                "    draw.text((x0 + 4, y0 + 4), f\"{{det['label']}} {{det['score']:.2f}}\", fill='red')\n"
                "annotated.save(OUTPUTS / '{stem}_annotated.png')\n\n"
                "with open(OUTPUTS / '{stem}_detections.csv', 'w', newline='', encoding='utf-8') as f:\n"
                "    writer = csv.writer(f)\n"
                "    writer.writerow(['image', 'rank', 'label', 'score', 'x0', 'y0', 'x1', 'y1'])\n"
                "    for rank, d in enumerate(coco_result['detections']):\n"
                "        writer.writerow(['tutorial-scene', rank, d['label'], f\"{{d['score']:.4f}}\", *(f\"{{v:.1f}}\" for v in d['box'])])\n"
                "    for row, record in zip(new_data_rows, new_records, strict=True):\n"
                "        for rank, d in enumerate(adapter.detect(record['image'], threshold=threshold)['detections']):\n"
                "            writer.writerow([row['id'], rank, d['label'], f\"{{d['score']:.4f}}\", *(f\"{{v:.1f}}\" for v in d['box'])])\n\n"
                "result_export = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__,\n"
                "                'cuda': torch.cuda.is_available()}},\n"
                "    'device': pipe.device,\n"
                "    'threshold': threshold,\n"
                "    'coco_detections': coco_result['detections'],\n"
                "    'degenerate_probes': degenerate,\n"
                "    'adaptation': {{\n"
                "        'dataset': {{k: v for k, v in dataset_manifest.items() if k != 'schema'}},\n"
                "        'dataset_seed': DATASET_SEED,\n"
                "        'split': {{'train': len(train_records), 'held_out': len(held_out), 'seed': SEED, 'holdout': HOLDOUT, 'held_out_support': support}},\n"
                "        'finetune': run,\n"
                "        'baseline': baseline,\n"
                "        'adapted': adapted,\n"
                "        'new_data': new_data_rows,\n"
                "        'new_data_found': {{'at_threshold': found, 'at_eval_threshold': found_lenient, 'truth_boxes': n_truth, 'threshold': threshold, 'eval_threshold': EVAL_DETECTION_THRESHOLD}},\n"
                "        'artifact': descriptor,\n"
                "        'reload_check': reload_check,\n"
                "    }},\n"
                "}}\n"
                "with open(OUTPUTS / '{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_export, f, indent=2, default=str)\n\n"
                "for p in sorted(OUTPUTS.iterdir()):\n"
                "    if p.is_file():\n"
                "        print(f'  {{p.name:<44s}} {{p.stat().st_size:>12,d}} bytes')"
            ),
        },
        # ---------------------------------------------------------------- 13. BYOD
        {
            "md": (
                "## 13. Optional: Bring Your Own Data (BYOD) · [Engineering]\n\n"
                "Both branches are off by default, so `Run all` never stops here. Before you turn one on, read the contract:\n\n"
                "- **Image branch** (`USE_BYOD_IMAGE`): one image file PIL can open, each side between `MIN_IMAGE_SIDE` (16) and `MAX_IMAGE_SIDE` "
                "(4096) px. It runs through `validate_inputs`, `detect` and `evaluation_report` — the report is `not-measurable`, because no "
                "reference boxes come with it.\n"
                "- **Dataset branch** (`USE_BYOD_DATASET`): a directory with `annotations.json` (a list of `{{'file', 'boxes', 'labels'}}` objects, "
                "xyxy pixel boxes) and the images it names, at most 5,000 records. The notebook needs **at least 8 records** (so the 25 % held-out split has at least 2 "
                "images), and it prints the held-out boxes per class with a named warning for any class below 3; for a usable measurement aim for "
                "dozens of images with every class in several of them. File names must stay inside the directory. "
                "`BYOD_CLASS_NAMES` is a comma-separated vocabulary; leave it empty to use the sorted set of labels in the annotations. The "
                "branch runs the same validate → split → baseline → fine-tune → evaluate → export → reload stages as the sample, including "
                "the reload equivalence check, and writes `outputs/byod_yolos_result.json` (manifest, split and support, baseline and adapted `ap` "
                "and `ap50`, the adapter descriptor and the reload check).\n\n"
                "Set `BYOD_IMAGE_PATH` or `BYOD_DATASET_DIR` to read from a mounted or local location; leave them empty on Colab to get an upload "
                "dialog instead. Uploaded files are written under `outputs/byod/` in this runtime and are not sent anywhere else. The first lines of "
                "the cell show the validator refusing two malformed inputs with messages that name the failed rule."
            ),
            "code": (
                'USE_BYOD_IMAGE = False  # @param {{type:"boolean"}}\n'
                'BYOD_IMAGE_PATH = ""  # @param {{type:"string"}}\n'
                'USE_BYOD_DATASET = False  # @param {{type:"boolean"}}\n'
                'BYOD_DATASET_DIR = ""  # @param {{type:"string"}}\n'
                'BYOD_CLASS_NAMES = ""  # @param {{type:"string"}}\n\n'
                "for desc, probe in (\n"
                "    ('non-image object', lambda: validate_inputs('/not/an/image.png')),\n"
                "    ('box outside the image', lambda: validate_dataset([{{'image': blank_scene(), 'boxes': [[0, 0, 9999, 10]], 'labels': [SIGN_CLASSES[0]]}}], SIGN_CLASSES)),\n"
                "):\n"
                "    try:\n"
                "        probe()\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print(f'refused as expected: {{desc}} -> {{type(exc).__name__}}: {{exc}}')\n\n"
                "BYOD_DIR = OUTPUTS / 'byod'\n\n"
                "def _upload_into(target):\n"
                "    from google.colab import files  # type: ignore[import-not-found]\n"
                "    target.mkdir(parents=True, exist_ok=True)\n"
                "    for name, data in files.upload().items():\n"
                "        (target / Path(name).name).write_bytes(data)\n"
                "    return target\n\n"
                "if USE_BYOD_IMAGE:\n"
                "    image_path = Path(BYOD_IMAGE_PATH) if BYOD_IMAGE_PATH else next(iter(sorted(_upload_into(BYOD_DIR / 'image').iterdir())))\n"
                "    with Image.open(image_path) as handle:\n"
                "        byod_image = handle.convert('RGB')\n"
                "    print(validate_inputs(byod_image, threshold=threshold, names=[image_path.name])['verdict'])\n"
                "    byod_result = pipe.detect(byod_image, threshold=threshold)\n"
                "    for det in byod_result['detections'][:20]:\n"
                "        print(f\"{{det['label']:>16s}} {{det['score']:.3f}}  [{{', '.join(f'{{v:.0f}}' for v in det['box'])}}]\")\n"
                "    print(evaluation_report(byod_result, None, sample_kind='byod')['verdict'])\n"
                "else:\n"
                "    print('BYOD image branch is off; set USE_BYOD_IMAGE = True to run detection on your own image.')\n\n"
                "if USE_BYOD_DATASET:\n"
                "    dataset_dir = Path(BYOD_DATASET_DIR) if BYOD_DATASET_DIR else _upload_into(BYOD_DIR / 'dataset')\n"
                "    byod_records = read_detection_records(dataset_dir)\n"
                "    if len(byod_records) < 8:\n"
                "        raise ValueError(f'the BYOD dataset has {{len(byod_records)}} records; at least 8 are needed so the held-out split has 2 or more images')\n"
                "    byod_names = [n.strip() for n in BYOD_CLASS_NAMES.split(',') if n.strip()] or sorted({{l for r in byod_records for l in r['labels']}})\n"
                "    byod_manifest = validate_dataset(byod_records, byod_names, epochs=EPOCHS)\n"
                "    print(json.dumps({{k: v for k, v in byod_manifest.items() if k != 'schema'}}, indent=2))\n"
                "    byod_train, byod_held = split_dataset(byod_records, train_fraction=1.0 - HOLDOUT, seed=SEED)\n"
                "    byod_support = held_out_support(byod_held, byod_names)\n"
                "    print({{'byod_train': len(byod_train), 'byod_held_out': len(byod_held), 'held_out_boxes_per_class': byod_support['boxes_per_class']}})\n"
                "    for warning in byod_support['warnings']:\n"
                "        print('warning:', warning)\n"
                "    byod_pipe = YolosDetectionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=byod_names, seed=SEED)\n"
                "    byod_baseline = byod_pipe.evaluate(byod_held)\n"
                "    byod_pipe.finetune(byod_train, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, seed=SEED, freeze_early_layers=FREEZE_EARLY_LAYERS)\n"
                "    byod_adapted = byod_pipe.evaluate(byod_held)\n"
                "    print({{'baseline_ap': round(byod_baseline['ap'], 4), 'baseline_ap50': round(byod_baseline['ap50'], 4), 'adapted_ap': round(byod_adapted['ap'], 4), 'adapted_ap50': round(byod_adapted['ap50'], 4)}})\n"
                "    byod_descriptor = byod_pipe.save_artifact(OUTPUTS / 'byod_yolos_adapter.safetensors', notes='BYOD adaptation adapter')\n"
                "    byod_reloaded = YolosDetectionPipeline.load_artifact(OUTPUTS / 'byod_yolos_adapter.safetensors', weights_dir=WEIGHTS_DIR)\n"
                "    byod_reload_check = reload_equivalence(byod_pipe, byod_reloaded, byod_held[0]['image'])\n"
                "    print('BYOD adapter exported and reloaded:', byod_descriptor['sha256'][:16], byod_reload_check)\n"
                "    byod_export = {{'notebook_source': NOTEBOOK_SOURCE, 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION, 'dataset_dir': str(dataset_dir),\n"
                "                   'class_names': byod_names, 'dataset': {{k: v for k, v in byod_manifest.items() if k != 'schema'}},\n"
                "                   'split': {{'train': len(byod_train), 'held_out': len(byod_held), 'seed': SEED, 'holdout': HOLDOUT, 'held_out_support': byod_support}},\n"
                "                   'baseline': byod_baseline, 'adapted': byod_adapted, 'artifact': byod_descriptor, 'reload_check': byod_reload_check}}\n"
                "    with open(OUTPUTS / 'byod_yolos_result.json', 'w', encoding='utf-8') as f:\n"
                "        json.dump(byod_export, f, indent=2, default=str)\n"
                "else:\n"
                "    print('BYOD dataset branch is off; set USE_BYOD_DATASET = True to adapt YOLOS on your own labelled images.')"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "**What this notebook established, in this runtime.** The pinned `hustvl/yolos-small` snapshot was verified against a committed "
        "SHA-256 manifest before loading. The pretrained detector was run on a drawn scene and on two structure-free probes, and its boxes were "
        "compared with the drawn references by IoU. A three-class sign vocabulary that COCO does not contain was then adapted by replacing the "
        "last class-head layer, training the last encoder layers, the detection tokens and the heads with the early layers frozen, and scoring the held-out split with COCO-style average "
        "precision before and after. The adapter was exported, reloaded onto a fresh base model and checked against the in-memory detections.\n\n"
        "**What a green run proves.** Successful execution proves that the recorded repository revision, the pinned dependency set and the "
        "pinned checkpoint together reproduce these stages in a fresh runtime, without the repository being cloned or installed and without "
        "any DIMER worker or service. It does **not** establish benchmark superiority, fitness for any deployment, or that the adapted model "
        "generalises beyond the synthetic images it was fitted to. The held-out AP is measured on 10 drawn images and carries no dispersion "
        "estimate. The scores are not calibrated probabilities.\n\n"
        "**Reproducibility.** Seeds are form fields (`DATASET_SEED`, `SEED`, `NEW_DATA_SEED`), the run is float32 with no data augmentation, "
        "and the class head is initialised under `SEED`. GPU kernels are not forced to be deterministic, so repeated GPU runs can differ in "
        "the last digits of the loss and the scores.\n\n"
        "**Try next** (each changes only form fields): change `EPOCHS` and watch where held-out AP stops improving; change `SEED` and see how "
        "much the held-out AP moves with a different split and head; to transfer the workflow, point `BYOD_DATASET_DIR` at a small labelled "
        "set from your own domain.\n\n"
        "## 14. Activity: change one thing — train the whole encoder · [Concept]\n\n"
        "Optional; **Predict → Change one thing → Run → Observe → Explain**. It changes nothing unless you do it.\n\n"
        "1. **Predict:** with `FREEZE_EARLY_LAYERS = False` the patch embedding and all 12 encoder layers are trained. Will held-out AP go up, "
        "down or stay the same? Will the fine-tune take longer?\n"
        "2. **Change:** in Section 8 set `FREEZE_EARLY_LAYERS = False`. Change nothing else.\n"
        "3. **Run:** select the Section 8 cell and choose *Runtime → Run after*. Section 8 rebuilds `adapter` from the verified base with the "
        "same seeded head, so this is one clean training run with the new setting, not a second run on top of the first. Sections 9–12 then "
        "re-evaluate and overwrite the outputs (download `outputs/` first if you want to keep them).\n"
        "4. **Observe:** Section 9 prints `run_history` with both runs side by side: `ap`, `ap50` and the fine-tune seconds. Compare the "
        "trainable-parameter counts in Section 8.\n"
        "5. **Explain:** in one sentence, why can training more parameters on 30 images fail to raise a saturated held-out AP50, and what "
        "would you need to see a difference?\n\n"
        "<details><summary>Check your reasoning</summary>Unfreezing roughly doubles the trainable parameters and lengthens each epoch. On this "
        "sample AP50 is already at its ceiling of 1.0, so it cannot rise; `ap` (tighter boxes) or the unseen-image counts can move a little in "
        "either direction, and with 10 held-out images one seed cannot tell a real gain from noise. More trainable parameters on 30 images also "
        "fit them more easily, which can generalise worse. Seeing a real difference needs a harder, larger held-out set, or several seeds.</details>\n\n"
        "## Troubleshooting\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| Section 1 stops with `This notebook needs a Linux x86_64 runtime` | a local Windows or macOS kernel, or an ARM machine | Use Google Colab, Kaggle or a Linux x86_64 Jupyter; the lock holds manylinux x86_64 wheels. |\n"
        "| Section 1 fails while downloading, or `The pinned uv wheel failed its size/SHA-256 check` | a network failure, or an altered download | Run the Section 1 install cell again; a repeated mismatch means the download is being altered — never edit the digest. |\n"
        "| `holds Python …, not 3.12.12` in Section 1 | an older `dimer_isolated_env/` folder from another notebook version | Delete that folder (or start a fresh runtime) and run the install cell again. |\n"
        "| `The isolated environment's Python process exited` | the worker ran out of memory | Lower `BATCH_SIZE` in Section 8, restart the session and choose *Runtime → Run all*. |\n"
        "| `CUDA out of memory` in Section 8 | another process holds GPU memory, or `FREEZE_EARLY_LAYERS = False` with a large batch | Lower `BATCH_SIZE` and run Section 8 again. |\n"
        "| Sections 8–9 are very slow | no GPU is attached | Choose a T4 runtime, or lower `EPOCHS` for a CPU smoke run. |\n"
        "| A Hub download fails in Section 3, or `sha256 … != manifest` | a transient Hugging Face failure, or a changed file | Run the Section 3 cell again; a digest mismatch is never loaded — do not edit the manifest. |\n"
        "| `warning: …: only N held-out box(es)` or `no held-out box` | the held-out split is too small for that class | Read that class's AP as unreliable; for BYOD add images of that class or change `SEED`. |\n"
        "| `the BYOD dataset has N records; at least 8 are needed` | too few labelled images | Add labelled images; a few dozen give a more useful measurement. |\n"
        "| A BYOD `ValueError` naming a record, box or label | `annotations.json` breaks the dataset contract (Section 13) | Fix the named record and run Section 13 again. |\n"
        "| `AssertionError` in the reload check | the adapter did not reproduce the in-memory detections | Re-run from Section 8; do not ship the adapter if it repeats. |\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional. Fill in from your own run, one sentence each:\n\n"
        "1. On the drawn COCO scene the detector found ___ of 4 objects at threshold ___.\n"
        "2. Held-out AP50 went from ___ (untrained head) to ___ after fine-tuning; `ap` went from ___ to ___; the smallest class had ___ held-out boxes.\n"
        "3. On the unseen images ___ of ___ signs were found at 0.9 and ___ at 0.05, which tells me ___.\n"
        "4. With `FREEZE_EARLY_LAYERS = False` (Section 14), held-out AP ___ and the fine-tune took ___.\n"
        "5. What I would need before trusting this adapted detector on real road photographs: ___.\n\n"
        "## References\n\n"
        "- Fang, Y., Liao, B., Wang, X., Fang, J., Qi, J., Wu, R., Niu, J. and Liu, W. (2021). *You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection.* [arXiv:2106.00666](https://arxiv.org/abs/2106.00666).\n"
        "- Carion, N. et al. (2020). *End-to-End Object Detection with Transformers.* [arXiv:2005.12872](https://arxiv.org/abs/2005.12872) — the set-prediction loss.\n"
        "- Upstream repository: [hustvl/YOLOS](https://github.com/hustvl/YOLOS) — MIT.\n"
        "- Hugging Face checkpoint: [hustvl/yolos-small](https://huggingface.co/hustvl/yolos-small) — Apache-2.0.\n"
        "- Lin, T.-Y. et al. (2014). *Microsoft COCO: Common Objects in Context.* [arXiv:1405.0312](https://arxiv.org/abs/1405.0312).\n"
        "- Repository model card: https://github.com/kurtvalcorza/yolos-detection-pipeline/blob/main/MODEL_CARD.md\n"
        "- [`kurtvalcorza/yolos-detection-pipeline`](https://github.com/kurtvalcorza/yolos-detection-pipeline) — source repository for this pipeline."
    ),
}
