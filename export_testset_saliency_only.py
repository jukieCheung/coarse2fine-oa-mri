#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
from tqdm import tqdm
import torch

# Reuse your proven implementation (model builders + saliency saver)
# Make sure this file is in the SAME folder as the eval script below.
from eval_oaizib_aclr_with_3dino_saliency_manifold_1logit_manifoldfix import (
    build_backbone,
    DualHeadModel,
    save_oaizib_input_saliency_for_one_case,
)

def _load_state_dict(ckpt_path: str):
    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    state = ckpt_obj
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            state = ckpt_obj["state_dict"]
        elif "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            state = ckpt_obj["model_state_dict"]
        elif "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            state = ckpt_obj["model"]

    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            kk = k
            if kk.startswith("module."):
                kk = kk[len("module."):]
            new_state[kk] = v
        state = new_state
    return state

def main():
    ap = argparse.ArgumentParser("Export saliency maps for OAIZIB TESTSET only (NIfTI)")

    ap.add_argument("--model", required=True, choices=["m3t", "mamba", "3dino", "resnet3d", "resnet"])
    ap.add_argument("--head_mode", required=True, choices=["single_oa", "single_kl", "dual"])
    ap.add_argument("--ckpt", required=True)

    ap.add_argument("--oaizib_excel", required=True, help="e.g. /mnt/g/OAI/info/subInfo_test.xlsx")
    ap.add_argument("--oaizib_img_root", required=True, help="e.g. /mnt/g/OAI/resampled_standardlize")
    ap.add_argument("--out_dir", required=True)

    # input resize (must match your training/eval)
    ap.add_argument("--input_D", type=int, default=128)
    ap.add_argument("--input_H", type=int, default=128)
    ap.add_argument("--input_W", type=int, default=128)

    # IMPORTANT: OA head output dim (1-logit BCE vs 2-logit CE)
    ap.add_argument("--oa_out_dim", type=int, default=1, choices=[1, 2],
                    help="For single_oa: 1 = single-logit (BCE), 2 = 2-logit (softmax). Default 1.")
    ap.add_argument("--dual_oa_out_dim", type=int, default=1, choices=[1, 2],
                    help="For dual: OA head out dim. Default 1.")

    # saliency params
    ap.add_argument("--saliency_target", type=str, default="oa",
                    choices=["oa", "kl", "auto"],
                    help="single_* ignores auto. dual: auto exports both.")
    ap.add_argument("--saliency_method", type=str, default="grad*input", choices=["grad", "grad*input"])
    ap.add_argument("--saliency_smooth", type=int, default=8)
    ap.add_argument("--saliency_noise_std", type=float, default=0.05)
    ap.add_argument("--saliency_class_idx", type=int, default=-1,
                    help="For KL saliency: -1 means argmax; otherwise 0-4.")

    # optional subset
    ap.add_argument("--only_cmt_ids", type=int, nargs="+", default=[],
                    help="If set, only export these CMT-IDs (e.g. --only_cmt_ids 12 44 99).")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # M3T commonly fixed 128^3 in your pipeline; keep same behavior:
    resize = (128, 128, 128) if args.model == "m3t" else (args.input_D, args.input_H, args.input_W)

    # ---- build model ----
    if args.head_mode == "single_kl":
        model = build_backbone(args.model, num_classes=5)

    elif args.head_mode == "single_oa":
        model = build_backbone(args.model, num_classes=int(args.oa_out_dim))

    elif args.head_mode == "dual":
        backbone = build_backbone(args.model, num_classes=5)
        model = DualHeadModel(backbone, num_classes_5=5, num_classes_2=int(args.dual_oa_out_dim))

    model = model.to(device)

    # ---- load ckpt ----
    state = _load_state_dict(args.ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[ckpt] Missing keys (first 30):", missing[:30])
    if unexpected:
        print("[ckpt] Unexpected keys (first 30):", unexpected[:30])
    print("Loaded ckpt:", args.ckpt)

    # ---- collect test IDs ----
    df = pd.read_excel(args.oaizib_excel)
    if "CMT-ID" not in df.columns:
        raise ValueError("Excel must contain column: 'CMT-ID'")
    cmt_ids = [int(x) for x in df["CMT-ID"].dropna().tolist()]

    if args.only_cmt_ids:
        keep = set(int(x) for x in args.only_cmt_ids)
        cmt_ids = [x for x in cmt_ids if x in keep]

    print(f"Total cases to export: {len(cmt_ids)}")

    # ---- export ----
    class_idx = None if args.saliency_class_idx < 0 else int(args.saliency_class_idx)

    for cmt_id in tqdm(cmt_ids, desc="Export saliency", ncols=140):
        # dual: maybe export both
        if args.head_mode == "dual" and args.saliency_target == "auto":
            targets = ["kl", "oa"]
        else:
            targets = [args.saliency_target]

        for target in targets:
            # single_oa/single_kl: force correct target
            if args.head_mode == "single_oa":
                target = "oa"
            if args.head_mode == "single_kl":
                target = "kl"

            # output path
            subdir = os.path.join(args.out_dir, "saliency", f"{args.model}_{args.head_mode}_{target}")
            os.makedirs(subdir, exist_ok=True)
            out_path = os.path.join(subdir, f"oaizib_{cmt_id:03d}_{target}_saliency.nii.gz")

            save_oaizib_input_saliency_for_one_case(
                model=model,
                device=device,
                head_mode=args.head_mode,
                excel_path=args.oaizib_excel,
                img_root=args.oaizib_img_root,
                resize=resize,
                cmt_id=int(cmt_id),
                out_path=out_path,
                sal_target=target,
                class_idx=class_idx,
                method=args.saliency_method,
                smooth=args.saliency_smooth,
                noise_std=args.saliency_noise_std,
            )

    print("Done. Saliency maps saved under:", os.path.join(args.out_dir, "saliency"))

if __name__ == "__main__":
    main()
