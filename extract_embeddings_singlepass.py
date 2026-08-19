#!/usr/bin/env python3
"""Single-pass embedding extraction: each NIfTI volume is read once from the
network mount and all 9 model configurations are evaluated on it.

Same protocol as extract_embeddings_trainfit.py, restructured for I/O
efficiency (9x fewer network reads). Models are the existing *_final.pth
checkpoints; no training. Penultimate embeddings are captured exactly as in
eval_oaizib_aclr_with_3dino_saliency_manifold_dualembfix.py (input of the
backbone's classification layer / backbone return_emb).
"""
import os, sys, types
import numpy as np
import pandas as pd
import torch
import nibabel as nib
import torch.nn.functional as F

BASE = __import__("os").environ.get("OACTF_BASE", ".")
OUT = f"{BASE}/geometry_trainfit"
os.makedirs(OUT, exist_ok=True)

MAMBA_NATIVE = True
try:
    import mamba_ssm  # noqa: F401
except Exception:
    MAMBA_NATIVE = False
    for name in ("selective_scan_cuda", "causal_conv1d_cuda"):
        sys.modules.setdefault(name, types.ModuleType(name))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "evalmod", f"{BASE}/eval_oaizib_aclr_with_3dino_saliency_manifold_dualembfix.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_ROOT = __import__("os").environ.get("OACTF_IMG_ROOT", "./resampled_standardlize")
SPLITS = {"train": __import__("os").environ.get("OACTF_TRAIN_XLSX", "./subInfo_train.xlsx"),
          "test": __import__("os").environ.get("OACTF_TEST_XLSX", "./subInfo_test.xlsx")}

CONFIGS = []
for bb, dname, mname in [("resnet3d", "resnet", "resnet3d"), ("m3t", "m3t", "m3t"), ("mamba", "mamba", "mamba")]:
    CONFIGS += [
        (bb, "single_oa", f"{BASE}/2_class/results_{dname}/{mname}_final.pth", 1),
        (bb, "single_kl", f"{BASE}/5_class/results_{dname}/{mname}_final.pth", 5),
        (bb, "dual", f"{BASE}/results_{dname}_dual/{mname}_dualhead_final.pth", 5),
    ]


def build_and_load(bb, setting, ckpt_path, ncls):
    if setting == "dual":
        model = ev.DualHeadModel(ev.build_backbone(bb, num_classes=5), 5, 2)
    else:
        model = ev.build_backbone(bb, num_classes=ncls)
    if not MAMBA_NATIVE:
        for m in model.modules():
            if hasattr(m, "use_fast_path"):
                m.use_fast_path = False
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"{bb}/{setting}: missing={missing[:5]} unexpected={unexpected[:5]}"
    return model.to(DEVICE).eval()


def load_volume(path):
    """Match load_nii_as_tensor: z-score, then [H,W,D] -> [D,H,W]."""
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    mean = data.mean()
    std = data.std()
    if std < 1e-6:
        std = 1e-6
    data = (data - mean) / std
    return np.ascontiguousarray(data.transpose(2, 0, 1))  # [D,H,W]


@torch.no_grad()
def main():
    models = {}
    for bb, setting, ckpt, ncls in CONFIGS:
        models[(bb, setting)] = build_and_load(bb, setting, ckpt, ncls)
        print(f"[loaded] {bb}/{setting}", flush=True)

    for split, excel in SPLITS.items():
        df = pd.read_excel(excel).dropna(subset=["KLGrade"]).copy()
        df["KLGrade"] = df["KLGrade"].astype(int)
        embs = {(bb, s): [] for bb, s, _, _ in CONFIGS}
        rows = {(bb, s): [] for bb, s, _, _ in CONFIGS}
        for i, (_, row) in enumerate(df.iterrows()):
            cmt = int(row["CMT-ID"]); kl = int(row["KLGrade"])
            path = os.path.join(IMG_ROOT, f"oaizib_{cmt:03d}_0000_n4_z.nii.gz")
            vol = load_volume(path)
            t = torch.from_numpy(vol)[None, None]  # [1,1,D,H,W]
            x160 = F.interpolate(t, size=(160, 256, 256), mode="trilinear", align_corners=False).to(DEVICE)
            x128 = F.interpolate(t, size=(128, 128, 128), mode="trilinear", align_corners=False).to(DEVICE)
            for bb, setting, _, _ in CONFIGS:
                x = x128 if bb == "m3t" else x160
                model = models[(bb, setting)]
                if setting == "dual":
                    logits_5, logits_2, emb = ev._forward_dual_with_backbone_embedding(model, x)
                    probs_5, _ = ev.kl_softmax_probs_preds(logits_5)
                    prob_oa, _ = ev.sigmoid_or_softmax_oa_prob(logits_2)
                else:
                    try:
                        logits, emb = model(x, return_emb=True)
                    except TypeError:
                        logits, emb = ev._forward_with_penultimate_embedding(model, x)
                    if setting == "single_kl":
                        probs_5, _ = ev.kl_softmax_probs_preds(logits)
                        prob_oa = ev.kl_probs_to_oa_prob(probs_5)
                    else:
                        probs_5 = None
                        prob_oa, _ = ev.sigmoid_or_softmax_oa_prob(logits)
                embs[(bb, setting)].append(emb[0].detach().cpu().numpy())
                r = {"CMT_ID": cmt, "label_kl": kl, "prob_oa": float(prob_oa[0].item())}
                if probs_5 is not None:
                    p5 = probs_5.detach().cpu().numpy()[0]
                    for c in range(5):
                        r[f"prob_kl_{c}"] = float(p5[c])
                rows[(bb, setting)].append(r)
            if (i + 1) % 25 == 0:
                print(f"[{split}] {i+1}/{len(df)}", flush=True)
        for bb, setting, _, _ in CONFIGS:
            tag = f"{bb}_{setting}_{split}"
            np.save(f"{OUT}/{tag}_embeddings.npy",
                    np.stack(embs[(bb, setting)]).astype(np.float32))
            pd.DataFrame(rows[(bb, setting)]).to_csv(f"{OUT}/{tag}_meta.csv", index=False)
        print(f"[done] {split}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
