# SACR-Net

Code and recorded results for **SACR-Net: Shape–Amplitude Complementary Representation for Limited-Label Hyperspectral Image Classification**.

## Scope

This release contains the training and evaluation pipeline for the fixed Raw–SNV SSRN backbone and its SFCR/PASI ablations, 120 configuration files, 15 frozen spatial split files, and tabulated results. It does not contain raw WHU-Hi data or trained checkpoints. External comparison methods are represented by recorded results; this release does not provide a complete training pipeline for every comparison method. Retained internal helper/model names reflect the development code and do not indicate additional evaluated methods.

| Configuration | SFCR | PASI |
|---|---|---|
| M00 | off | off |
| M01 | on | off |
| M02 | off | on |
| M05 (SACR-Net) | on | on |

`adapter.py` implements the four configurations; `rq_encoders.py` contains channel-relation processing; `hsi_shadow/models/ssrn_official.py` provides the backbone. The legacy `model.name` field is not the architecture selected by `train_run.py`, which calls `adapter.build_model`. The auxiliary view is SNV. NPAF is disabled in all released configurations.

## Setup and data

Use Python 3.10 or later and install an appropriate PyTorch build for your platform, then:

```bash
python -m pip install -r requirements.txt
```

The dependency list is not an exact historical environment lockfile. Obtain WHU-Hi from the original provider cited in the paper. Arrange the files under your data root as follows (MAT variable names are in `hsi_shadow/data/registry.py`):

```text
data/raw/
  WHU-Hi-HanChuan/WHU_Hi_HanChuan.mat
  WHU-Hi-HanChuan/WHU_Hi_HanChuan_gt.mat
  WHU-Hi-LongKou/WHU_Hi_LongKou.mat
  WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat
  WHU_Hi_HongHu.mat
  WHU_Hi_HongHu_gt.mat
```

## Training and evaluation

Run from the repository root so the relative frozen-split paths resolve:

```bash
python train_run.py --config configs/longkou_M05_s42.yaml --data-root /path/to/raw --output-dir runs/longkou_M05_s42
```

The entry point trains the model, chooses the checkpoint using validation OA (last-tie rule), and writes evaluation metrics, predictions and preprocessing records. Use another supplied config to change scene, seed or ablation. Do not change experimental settings if comparing with the recorded matrix. `evaluate_one.py` is an archival replay tool requiring a manifest and its referenced checkpoint/preprocessing artifacts; these artifacts are not included here.

## Protocol and interpretation

Selected training seeds: **7, 42, 256, 2026, 3407, 11001, 11002, 11004, 11005, 11007**. They map to five spatial splits per scene; see `SELECTED_SEEDS.json` and `SPLIT_MAPPING.json`. There are four configurations × three scenes × ten seeds = 120 runs. The seed subset and scene learning rates were selected with reference to previously viewed test results. Results describe a development subset and do not constitute independent confirmatory evidence. Archived results are supplied as measurements, not as a claim that rerunning this package will be bitwise identical.

## Release preparation and validation

The model and training Python sources are preserved from the revision archive. Configuration data roots were changed to `data/raw`; machine-specific path and hash columns were omitted from public result CSVs. Numeric result fields, selected seeds and frozen split bytes were retained. `PACKAGE_CHECKS.json` records packaging checks. No full training reproduction was performed while preparing this release. No new license is assigned by this package; existing attribution in the source is retained.
