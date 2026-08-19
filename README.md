# Learning Coarse-to-Fine Osteoarthritis Representations under Noisy Hierarchical Labels

Official code for the MICAD submission.

We study whether supervision at two clinical granularities — binary OA status and
Kellgren–Lawrence (KL) severity — organizes 3D knee MRI representations differently,
using a shared encoder with OA and KL prediction heads under Single-OA, Single-KL,
and Dual-head training across three backbones (ResNet3D, M3T, nnMamba).

## Repository layout

```
train_oaizib_3models.py                 # Single-KL (5-class CE) training
train_oaizib_3models_binary.py          # Single-OA (1-logit BCE) training
train_oaizib_3models_dual.py            # Dual-head (OA + KL) training
models/                                 # resnet.py, M3T.py, nnMamba4cls.py
eval_oaizib_aclr.py                     # test-set evaluation -> *_preds.csv / *_metrics.csv
extract_embeddings_singlepass.py        # penultimate embeddings (train + test), inference only
analyze_severity_axis_trainfit.py       # train-fit PCA + 1-D PC1 probe, test-evaluated
compute_dual_vs_single_stats.py         # canonical paired stats (McNemar + bootstrap) + BH FDR
plot_forest_dual_vs_single.py           # forest plots from the canonical stats table
export_testset_saliency_only.py         # SmoothGrad x input saliency export
compute_saliency_overlap_metrics.py     # saliency-cartilage overlap (mass@ROI, top1@ROI, Dice)
results/                                # per-subject predictions, metrics, canonical stats
```

## Data

Experiments use the OAIZIB-CM cohort (OAI-ZIB knee MRI with cartilage segmentations;
see Ambellan et al., MedIA 2019 and Yao et al., MedIA 2024). The dataset must be
obtained from its official release; we do not redistribute images or masks.
Expected layout: NIfTI volumes `oaizib_{CMT-ID:03d}_0000_n4_z.nii.gz`, subject
metadata Excel files with columns `CMT-ID` and `KLGrade`, and cartilage masks
(labels 2 and 4 = femoral and tibial cartilage). We use the predefined split of
383 training / 98 test subjects.

## Reproducing the paper

Training (per backbone `{resnet3d, m3t, mamba}`):

```bash
# Single-KL
python train_oaizib_3models.py --train_excel subInfo_train.xlsx --test_excel subInfo_test.xlsx \
    --img_root resampled_standardlize/ --model <backbone> --num_classes 5 --epochs 100 --batch_size 2

# Single-OA
python train_oaizib_3models_binary.py --train_excel subInfo_train.xlsx --test_excel subInfo_test.xlsx \
    --img_root resampled_standardlize/ --model <backbone> --num_classes 1 --epochs 100 --batch_size 2

# Dual
python train_oaizib_3models_dual.py --train_excel subInfo_train.xlsx --test_excel subInfo_test.xlsx \
    --img_root resampled_standardlize/ --model <backbone> --num_classes 5 --binary_threshold 2 \
    --epochs 100 --batch_size 2
```

Statistics and figures (no GPU needed; runs from the saved per-subject predictions
in `results/`):

```bash
python compute_dual_vs_single_stats.py     # -> stats_recomputed.csv / stats_recomputed_with_fdr.csv
python plot_forest_dual_vs_single.py       # -> forest_dual_vs_single_{OA,KL}.png
```

Representation geometry (requires trained checkpoints; inference only):

```bash
python extract_embeddings_singlepass.py    # -> geometry_trainfit/*_embeddings.npy
python analyze_severity_axis_trainfit.py   # -> severity_axis_trainfit_summary.csv
```

Saliency overlap (requires exported saliency maps and cartilage masks):

```bash
python export_testset_saliency_only.py --model <backbone> --head_mode <single_oa|single_kl|dual> ...
python compute_saliency_overlap_metrics.py --saliency_dir <dir> --mask_dir <dir> --cart_labels 2,4
```

## Notes

- All reported predictive statistics are computed on the held-out 98-subject test
  set with paired tests: McNemar for OA accuracy, paired subject-level bootstrap
  (B=10,000, seed 42) for AUC/F1/KL metrics, and Benjamini–Hochberg FDR over the
  18 primary Dual-vs-single comparisons (`stats_recomputed_with_fdr.csv`).
- PCA and the 1-D PC1 OA probe are fitted on the 383 training subjects only and
  evaluated on the held-out test subjects.
- nnMamba requires `mamba_ssm`; install the prebuilt wheel matching your torch
  build from https://github.com/state-spaces/mamba/releases.

## Citation

If you use this code, please cite our paper (citation details will be added upon publication).
