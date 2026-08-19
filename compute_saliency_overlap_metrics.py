#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, argparse
import numpy as np
import pandas as pd
import nibabel as nib

def load_nii(path):
    nii = nib.load(path)
    data = nii.get_fdata(dtype=np.float32)
    return data, nii

def parse_id(fname):
    m = re.search(r"oaizib_(\d+)", os.path.basename(fname))
    return m.group(1) if m else None

def safe_positive_saliency(S):
    # for grad / grad*input, keep positive part as "attention"
    S = np.maximum(S, 0.0)
    return S

def mass_at_roi(Spos, M, eps=1e-8):
    denom = float(Spos.sum())
    if denom < eps:
        return np.nan, denom
    num = float((Spos * M).sum())
    return num / (denom + eps), denom

def topk_mask_from_scores(Spos_flat, k_frac):
    # k_frac e.g. 0.01, 0.05, 0.10
    n = Spos_flat.size
    k = int(np.ceil(k_frac * n))
    k = max(k, 1)

    # If all zeros, topk is arbitrary; still return something deterministic
    # Using argpartition for O(n)
    idx = np.argpartition(Spos_flat, -k)[-k:]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask, k

def topk_metrics(Spos, M, k_list=(0.01, 0.05, 0.10)):
    # Spos, M are 3D arrays (same shape)
    Sflat = Spos.reshape(-1)
    Mflat = (M.reshape(-1) > 0.5)

    roi_n = int(Mflat.sum())
    out = {"roi_voxels": roi_n}

    for kf in k_list:
        tmask, k = topk_mask_from_scores(Sflat, kf)
        inter = int(np.logical_and(tmask, Mflat).sum())

        topk_at_roi = inter / float(k)
        dice = (2.0 * inter) / float(k + roi_n) if (k + roi_n) > 0 else np.nan

        out[f"top{int(kf*100)}_k_voxels"] = k
        out[f"top{int(kf*100)}_at_roi"] = topk_at_roi
        out[f"top{int(kf*100)}_dice"] = dice
        out[f"top{int(kf*100)}_intersect"] = inter

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saliency_dir", required=True,
                    help="Folder containing saliency nii.gz, directory of *_saliency.nii.gz files")
    ap.add_argument("--mask_dir", required=True,
                    help="Folder containing mask nii.gz, directory of cartilage mask nii.gz files")
    ap.add_argument("--cart_labels", default="2,4",
                    help="Comma-separated cartilage label ids in the mask. Default: 2,4")
    ap.add_argument("--out_csv", required=True,
                    help="Output CSV path")
    ap.add_argument("--check_affine", action="store_true",
                    help="Warn when saliency and mask affine/shape mismatch.")
    ap.add_argument("--k_list", default="0.01,0.05,0.10",
                    help="Comma-separated top-k fractions. Default: 0.01,0.05,0.10")
    args = ap.parse_args()

    cart_labels = [int(x) for x in args.cart_labels.split(",") if x.strip() != ""]
    k_list = [float(x) for x in args.k_list.split(",") if x.strip() != ""]

    sal_files = sorted(glob.glob(os.path.join(args.saliency_dir, "*.nii.gz")))
    if len(sal_files) == 0:
        raise FileNotFoundError(f"No nii.gz found in {args.saliency_dir}")

    rows = []
    for sf in sal_files:
        sid = parse_id(sf)
        if sid is None:
            print(f"[skip] cannot parse id from {sf}")
            continue

        mf = os.path.join(args.mask_dir, f"oaizib_{sid}.nii.gz")
        if not os.path.exists(mf):
            # fallback: try any file containing the id
            cand = glob.glob(os.path.join(args.mask_dir, f"*{sid}*.nii.gz"))
            mf = cand[0] if len(cand) else None

        if mf is None or not os.path.exists(mf):
            print(f"[skip] mask not found for id={sid}, sal={os.path.basename(sf)}")
            continue

        S, snii = load_nii(sf)
        L, mnii = load_nii(mf)

        if args.check_affine:
            if S.shape != L.shape:
                print(f"[warn] shape mismatch id={sid}: sal {S.shape} vs mask {L.shape}")
            if not np.allclose(snii.affine, mnii.affine, atol=1e-3):
                print(f"[warn] affine mismatch id={sid}")

        # ROI mask: cartilage labels
        M = np.zeros_like(L, dtype=bool)
        for lab in cart_labels:
            M |= (L == lab)
        M = M.astype(np.float32)

        Spos = safe_positive_saliency(S)

        mass_roi, denom = mass_at_roi(Spos, M)
        topk = topk_metrics(Spos, M, k_list=tuple(k_list))

        row = {
            "id": sid,
            "saliency_file": sf,
            "mask_file": mf,
            "cart_labels": ",".join(map(str, cart_labels)),
            "mass_at_roi": mass_roi,
            "saliency_sum_pos": denom,
            "saliency_shape": "x".join(map(str, S.shape)),
        }
        row.update(topk)

        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["id"])
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    # quick summary for key metrics
    key_cols = ["mass_at_roi"] + [f"top{int(k*100)}_at_roi" for k in k_list] + [f"top{int(k*100)}_dice" for k in k_list]
    print(df[key_cols].describe())
    print("Saved:", args.out_csv)

if __name__ == "__main__":
    main()
