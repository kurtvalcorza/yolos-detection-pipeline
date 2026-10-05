# YOLOS-Small detection fine-tuning notebook — review fixes

**Review:** `yolos_detection_colab_Review.md` (5 October 2026; findings YOS-M1, YOS-M2 and YOS-m1 to YOS-m4).

**Fixed in:**
- the generator `tools/build_notebook.py`, now the isolated-runtime generator `build_notebook.py/2.1-swc`;
- `tools/notebook_template.py`;
- `samples.py` (`held_out_support`);
- the new hash lock `tutorials/requirements-colab.lock.txt`;
- the validator, `tutorials/README.md`, `README.md`, `docs/release-verification.md` and the tests.

The notebook was regenerated and `--check` passes.

**Readiness:** **Verification pending** until the hosted gates below are recorded. `STATUS.md` and every release label are unchanged.

## Findings

| ID | Status | Change | Cells / files | Evidence |
|---|---|---|---|---|
| YOS-M1 | Fixed in source; a hosted run is still needed | The in-kernel install and its restart guard are gone. In their place:<ul><li>a pinned `uv` 0.12.15 wheel, checked by size and SHA-256;</li><li>managed CPython 3.12.12;</li><li>a hash lock of 48 packages compiled from `pyproject.toml`;</li><li>a router that sends every later cell to one worker.</li></ul>The notebook now declares spec 2.2. The registry's Run-all cell says the 2026-09-25 run needed a restart and that the regenerated notebook has not been run. | Sections 1–3; validator; README files; record | `test_yos_M1_*`; `grep "Restart the runtime"` returns nothing |
| YOS-M2 | Fixed | The guided layer now includes:<ul><li>How to use, with an audience statement, cell kinds, form controls and the predict/check convention;</li><li>an I/M/O table and a roadmap;</li><li>a glossary covering detection token, no-object, threshold, IoU, Hungarian matching, GIoU, `eos_coefficient`, re-heading, AP50/AP75/AP, support, saturated and adapter;</li><li>Predict prompts before Sections 4, 7, 8, 9 and 10, each with What to notice and a collapsed answer;</li><li>a Section 14 activity built on the YOS-m4 fix;</li><li>a troubleshooting table and a conclusion template;</li><li>Infrastructure titles and `cellView: form` on the install and carrier cells.</li></ul> | template | `test_yos_M2_*` (12 tests) |
| YOS-m1 | Fixed | The prose now matches the record:<ul><li>"near zero" is replaced by what the baseline is and why it scores 0.2 overall, and 0.6 on the two-box speed-limit class.</li><li>The Prerequisites quote the recorded T4 timing (172.1 s per pass, 105.5 s fine-tune, 437.5 s with the restart) and label it as from the previous notebook.</li><li>Section 10 scores the unseen images at 0.9 and at 0.05 and prints both counts. The answers after Sections 9 and 10 explain the two thresholds and call AP50 1.0 saturated.</li></ul> | Sections 7, 9, 10; Prerequisites | `test_yos_m1_no_statement_contradicts_the_record` |
| YOS-m2 | Fixed (coverage reported and warned; split not stratified) | `held_out_support` prints held-out boxes per class beside every per-class AP, with a named warning below 3 boxes or at zero. On the default split that is 10 / 5 / 2 boxes, with a warning for the speed-limit sign. BYOD needs at least 8 records and gets the same warnings; the minimum is stated in the BYOD declaration and Section 13. **Not done:** stratifying the split. That would change the default split and every recorded number, so it is left to the maintainer. | `samples.held_out_support`; Sections 7, 9, 13 | `test_yos_m2_*` |
| YOS-m3 | Fixed | The equivalence check is now `reload_equivalence` in Section 11, and the BYOD branch uses it on a held-out BYOD image. The branch prints `ap` and `ap50` and writes `outputs/byod_yolos_result.json` (manifest, split and support, both metrics, descriptor, reload check). | Sections 11, 13 | `test_yos_m2_m3_byod_dataset_branch_*`, `test_yos_m3_reload_equivalence_fails_when_the_reload_differs` |
| YOS-m4 | Fixed | Section 8 rebuilds `adapter` from the verified base with the same seeded head before every fine-tune, so re-running it trains once from the start. It also times the fine-tune. Section 9 appends each run to `run_history` and prints the runs side by side. "Try next" and the activity say so. | Sections 8, 9; closing | `test_yos_m4_rerunning_the_finetune_cell_trains_a_fresh_adapter` |

Suggestions: S4 applied (spec 2.2). S1, S2 and S3 were not taken.

## User-visible changes

- Section 1 builds an isolated environment. It runs on Linux x86_64 only, and the CUDA build of `torch` is a larger download. Every later cell runs there.
- Section 7 prints the held-out support per class and any warnings.
- Section 8 rebuilds the adapter, which adds one model load, and prints `finetune_seconds`. The default results are expected to be unchanged, because the same seed gives the same head.
- Section 9 prints `run_history`.
- Section 10 prints counts at both thresholds. `result.json` gains `held_out_support` and `new_data_found`.
- The BYOD dataset branch refuses fewer than 8 records, checks reload equivalence and writes `byod_yolos_result.json`.

## Verification (offline, not clean-runtime evidence)

- `build_notebook.py --check`: OK. `validate_release_assets.py`: PASS. `ruff check src tests tools`: clean.
- `pytest` without torch, before: 53 passed, 1 failed, 1 skipped. The failure is `test_save_artifact_refuses_an_unadapted_pipeline`, which imports `safetensors.torch` and needs torch; CI installs it.
- `pytest` without torch, after: **72 passed**, with the same 1 failed and 1 skipped (19 new tests).
- **Real data:** `held_out_support` on the real drawn dataset reproduces the review's P1 counts (10 / 5 / 2).
- **Stand-in detector:** the BYOD branch ran with real images and `annotations.json`, and the Section 8 cell ran twice. These runs prove the plumbing, not the model.
- **Not run here:** no YOLOS inference or training (no Hub access, no torch).

## Remaining gates

1. A hosted one-pass **Run all** of the regenerated blob on a fresh T4, recorded with `restarted: false`. Confirm that the default numbers match the 2026-09-25 record (Section 8 now rebuilds the adapter with the same seed). Then re-run the export cell.
2. The REL12 BYOD journey: one image, one labelled directory, and one incompatible `annotations.json`.
3. Maintainer decision: stratify the split by class presence (this changes the recorded numbers).
