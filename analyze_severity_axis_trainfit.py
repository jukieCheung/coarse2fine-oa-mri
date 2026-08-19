#!/usr/bin/env python3
"""Train-fit / test-evaluate severity-axis geometry (leakage-free Table 2).

Protocol (per backbone x supervision setting):
  1. StandardScaler is FIT on the 383 training embeddings only.
  2. PCA(n_components=2, random_state=0) is FIT on the scaled training embeddings.
  3. Test embeddings (98 held-out subjects) are TRANSFORMED with the fitted
     scaler + PCA; all reported correlations are computed on the test split.
  4. |Spearman(PC1, KL)| and |Spearman(PC1, OA)| are reported as magnitudes
     because the PCA axis sign is arbitrary.
  5. The OA probe is a TRUE one-dimensional logistic regression on PC1 scores
     only, FIT on training subjects, and evaluated (AUROC) on the held-out
     test subjects only. No supervised model is fitted on the test set.

Inputs: geometry_trainfit/{backbone}_{setting}_{split}_embeddings.npy and
        geometry_trainfit/{backbone}_{setting}_{split}_meta.csv
        written by extract_embeddings_trainfit.py.
Output: geometry_trainfit/severity_axis_trainfit_summary.csv
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BASE = __import__("os").environ.get("OACTF_GEOM", "./geometry_trainfit")
BACKBONES = ["resnet3d", "m3t", "mamba"]
SETTINGS = ["single_oa", "single_kl", "dual"]
NAMES = {"resnet3d": "ResNet3D", "m3t": "M3T", "mamba": "nnMamba",
         "single_oa": "Single-OA", "single_kl": "Single-KL", "dual": "Dual"}


def load(bb, setting, split):
    E = np.load(f"{BASE}/{bb}_{setting}_{split}_embeddings.npy")
    meta = pd.read_csv(f"{BASE}/{bb}_{setting}_{split}_meta.csv")
    # Embeddings and meta rows were written in the same DataLoader order
    # (sequential, shuffle=False), so row i of E corresponds to row i of meta.
    return E, meta


def main():
    rows = []
    for bb in BACKBONES:
        for setting in SETTINGS:
            Etr, mtr = load(bb, setting, "train")
            Ete, mte = load(bb, setting, "test")
            assert Etr.shape[0] == len(mtr) == 383, (bb, setting, Etr.shape, len(mtr))
            assert Ete.shape[0] == len(mte) == 98, (bb, setting, Ete.shape, len(mte))

            kl_tr = mtr["label_kl"].values
            oa_tr = (kl_tr >= 2).astype(int)
            kl_te = mte["label_kl"].values
            oa_te = (kl_te >= 2).astype(int)

            scaler = StandardScaler().fit(Etr)
            pca = PCA(n_components=2, random_state=0).fit(scaler.transform(Etr))
            pc1_tr = pca.transform(scaler.transform(Etr))[:, 0]
            pc1_te = pca.transform(scaler.transform(Ete))[:, 0]

            rho_kl = spearmanr(pc1_te, kl_te).correlation
            rho_oa = spearmanr(pc1_te, oa_te).correlation

            probe = LogisticRegression(max_iter=2000)
            probe.fit(pc1_tr.reshape(-1, 1), oa_tr)
            p_te = probe.predict_proba(pc1_te.reshape(-1, 1))[:, 1]
            auroc = roc_auc_score(oa_te, p_te)
            # also the train-fit AUROC, reported for transparency
            auroc_tr = roc_auc_score(oa_tr, probe.predict_proba(pc1_tr.reshape(-1, 1))[:, 1])

            rows.append(dict(
                backbone=NAMES[bb], setting=NAMES[setting],
                n_train=Etr.shape[0], n_test=Ete.shape[0], D=Etr.shape[1],
                EVR_PC1=pca.explained_variance_ratio_[0],
                spearman_PC1_KL_signed=rho_kl, abs_spearman_PC1_KL=abs(rho_kl),
                spearman_PC1_OA_signed=rho_oa, abs_spearman_PC1_OA=abs(rho_oa),
                OA_probe_AUROC_test=auroc, OA_probe_AUROC_train=auroc_tr,
            ))
            print(f"{bb:9s} {setting:9s} EVR1={rows[-1]['EVR_PC1']:.3f} "
                  f"|rhoKL|={abs(rho_kl):.3f} |rhoOA|={abs(rho_oa):.3f} "
                  f"probeAUC_test={auroc:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(f"{BASE}/severity_axis_trainfit_summary.csv", index=False)
    print("Wrote", f"{BASE}/severity_axis_trainfit_summary.csv")


if __name__ == "__main__":
    main()
