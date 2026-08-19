import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os, sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "3DINO"))

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)



from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def _forward_with_penultimate_embedding(model, x):
    """Forward pass that also returns a penultimate embedding vector.

    If `model(x, return_emb=True)` is supported, prefer that upstream.
    Otherwise, we capture the input to the *last* nn.Linear module as an embedding.
    This makes manifold plots meaningful even for 1-logit binary heads.
    """
    last_linear = None
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            last_linear = m
    if last_linear is None:
        y = model(x)
        return y, y

    cache = {}
    def _hook(mod, inp, out):
        if inp and isinstance(inp[0], torch.Tensor):
            cache["emb"] = inp[0]

    h = last_linear.register_forward_hook(_hook)
    y = model(x)
    h.remove()
    emb = cache.get("emb", None)
    if emb is None:
        emb = y if isinstance(y, torch.Tensor) else (y[0] if isinstance(y, (tuple, list)) and len(y) else y)
    return y, emb


def _save_neural_manifold(embeddings_np, meta_df, out_prefix: str,
                          method: str = "tsne",
                          color_by: str = "label_oa",
                          max_n: int = 2000,
                          seed: int = 42,
                          perplexity: float = 30.0):
    """Save per-sample embeddings and a 2D manifold projection (t-SNE/PCA)."""
    if embeddings_np is None or len(embeddings_np) == 0:
        print("[manifold] No embeddings collected; skip.")
        return

    E = np.asarray(embeddings_np, dtype=np.float32)
    if E.ndim != 2:
        E = E.reshape(E.shape[0], -1)

    emb_path = out_prefix + "_embeddings.npy"
    np.save(emb_path, E)
    print("[manifold] Saved embeddings:", emb_path, E.shape)

    if meta_df is None or len(meta_df) == 0:
        meta_df = pd.DataFrame({"idx": np.arange(E.shape[0])})
    if len(meta_df) != E.shape[0]:
        n = min(len(meta_df), E.shape[0])
        print(f"[manifold] Warning: meta rows ({len(meta_df)}) != embeddings ({E.shape[0]}). Truncating to {n}.")
        meta_df = meta_df.iloc[:n].copy()
        E = E[:n]

    rng = np.random.RandomState(seed)
    if E.shape[0] > max_n:
        sel = rng.choice(E.shape[0], size=max_n, replace=False)
        sel.sort()
        E_sub = E[sel]
        meta_sub = meta_df.iloc[sel].copy()
    else:
        E_sub = E
        meta_sub = meta_df.copy()

    E_sub = StandardScaler().fit_transform(E_sub)

    method = method.lower()
    if method == "pca":
        proj = (PCA(n_components=2, random_state=seed).fit_transform(E_sub)
                if E_sub.shape[1] >= 2 else np.stack([E_sub[:, 0], np.zeros(E_sub.shape[0], dtype=E_sub.dtype)], axis=1))
    elif method == "tsne":
        n = E_sub.shape[0]
        perp = float(perplexity)
        if n <= 5:
            print("[manifold] Too few samples for t-SNE; falling back to PCA.")
            proj = (PCA(n_components=2, random_state=seed).fit_transform(E_sub)
                if E_sub.shape[1] >= 2 else np.stack([E_sub[:, 0], np.zeros(E_sub.shape[0], dtype=E_sub.dtype)], axis=1))
            method = "pca"
        else:
            perp = min(perp, max(2.0, (n - 1) / 3.0))
            proj = TSNE(
                n_components=2,
                perplexity=perp,
                init=("pca" if E_sub.shape[1] >= 2 else "random"),
                learning_rate="auto",
                random_state=seed,
            ).fit_transform(E_sub)
    else:
        raise ValueError(f"Unknown manifold method: {method}")

    meta_sub["manifold_x"] = proj[:, 0]
    meta_sub["manifold_y"] = proj[:, 1]

    csv_path = out_prefix + f"_manifold_{method}.csv"
    meta_sub.to_csv(csv_path, index=False)
    print("[manifold] Saved manifold CSV:", csv_path)

    try:
        plt.figure(figsize=(7, 6))
        if color_by in meta_sub.columns:
            c = meta_sub[color_by].values
            sc = plt.scatter(meta_sub["manifold_x"], meta_sub["manifold_y"], c=c, s=10, alpha=0.8)
            plt.colorbar(sc, label=color_by)
        else:
            plt.scatter(meta_sub["manifold_x"], meta_sub["manifold_y"], s=10, alpha=0.8)
        plt.title(f"Neural manifold ({method.upper()})")
        plt.xlabel("dim-1")
        plt.ylabel("dim-2")
        plt.tight_layout()
        fig_path = out_prefix + f"_manifold_{method}.png"
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print("[manifold] Saved manifold plot:", fig_path)
    except Exception as e:
        print("[manifold] Plot failed:", repr(e))
# =======================
#   NIfTI load (same style as your original)
# =======================

def load_nii_as_tensor(path, resize=None):
    """
    .nii.gz -> z-score -> torch [1, D, H, W]
    resize: (D,H,W) or None
    """
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)

    mean = data.mean()
    std = data.std()
    if std < 1e-6:
        std = 1e-6
    data = (data - mean) / std

    vol = torch.from_numpy(data)  # [H,W,D]
    if vol.ndim != 3:
        raise ValueError(f"Expect 3D volume, got {vol.shape} for {path}")

    vol = vol.permute(2, 0, 1)           # [D,H,W]
    vol = vol.unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]

    if resize is not None:
        vol = F.interpolate(vol, size=resize, mode="trilinear", align_corners=False)

    vol = vol.squeeze(0)  # [1,D,H,W]
    return vol


# =======================
#   OAIZIB Dataset (KL label; OA derived later)
# =======================

def _forward_dual_with_backbone_embedding(dual_model, x):
    """Forward a DualHeadModel and also capture the penultimate embedding from its backbone.

    This avoids using the OA binary head input (which is KL logits and only 5-D).
    Instead we hook the *backbone*'s last Linear and capture its input (e.g., 512/128/32-D),
    making embeddings comparable across single-head and dual-head runs.
    """
    # Find last Linear inside backbone
    last_linear = None
    for m in dual_model.backbone.modules():
        if isinstance(m, torch.nn.Linear):
            last_linear = m
    cache = {}
    if last_linear is not None:
        def _hook(mod, inp, out):
            if inp and isinstance(inp[0], torch.Tensor):
                cache["emb"] = inp[0]
        h = last_linear.register_forward_hook(_hook)
        logits_5, logits_2 = dual_model(x)
        h.remove()
        emb = cache.get("emb", None)
        return logits_5, logits_2, emb
    # Fallback: try backbone return_emb
    try:
        logits_5, emb = dual_model.backbone(x, return_emb=True)
        logits_2 = dual_model.binary_head(logits_5)
        return logits_5, logits_2, emb
    except Exception:
        logits_5, logits_2 = dual_model(x)
        return logits_5, logits_2, None


class OAIZIBDataset(Dataset):
    """
    Excel needs:
      - 'CMT-ID'
      - 'KLGrade' (0-4)
    Image:
      oaizib_{CMT-ID:03d}_0000_n4_z.nii.gz
    """
    def __init__(self, excel_path, img_root, resize=None):
        super().__init__()
        self.img_root = img_root
        self.resize = resize

        df = pd.read_excel(excel_path)
        df = df.dropna(subset=["KLGrade"]).copy()
        df["KLGrade"] = df["KLGrade"].astype(int)

        self.samples = []
        for _, row in df.iterrows():
            cmt_id = int(row["CMT-ID"])
            kl = int(row["KLGrade"])
            fname = f"oaizib_{cmt_id:03d}_0000_n4_z.nii.gz"
            fpath = os.path.join(img_root, fname)
            if not os.path.isfile(fpath):
                raise FileNotFoundError(f"Image not found: {fpath}")
            self.samples.append((fpath, cmt_id, kl))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cmt_id, kl = self.samples[idx]
        vol = load_nii_as_tensor(img_path, resize=self.resize)
        return vol, int(cmt_id), int(kl), img_path


# =======================
#   Model builders
# =======================

# =======================
#   3DINO wrapper (same model skeleton as your training script)
# =======================

class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Dino3DClassifier(nn.Module):
    """backbone.forward_features(x) -> cls token (or pooled tokens) -> MLP head"""
    def __init__(self, backbone: nn.Module, embed_dim: int, num_classes: int,
                 head_hidden=512, head_dropout=0.2):
        super().__init__()
        self.backbone = backbone
        self.head = MLPHead(embed_dim, num_classes, hidden_dim=head_hidden, dropout=head_dropout)

    def forward(self, x, return_emb: bool = False):
        feats = self.backbone.forward_features(x)
        if isinstance(feats, dict):
            if "x_norm_clstoken" in feats:
                z = feats["x_norm_clstoken"]
            elif "x_norm_patchtokens" in feats:
                z = feats["x_norm_patchtokens"].mean(dim=1)
            else:
                z = next(v for v in feats.values() if torch.is_tensor(v))
        else:
            z = feats

        # if tokens: [B, N, C] -> mean over tokens
        if z.ndim == 3:
            z = z.mean(dim=1)
        logits = self.head(z)
        if return_emb:
            return logits, z
        return logits

def build_backbone(name: str, num_classes: int):
    name = name.lower()

    if name == "mamba":
        from nnMamba4cls import nnMambaEncoder
        model = nnMambaEncoder(in_ch=1, channels=32, blocks=3, number_classes=num_classes)

    elif name == "3dino":
        # Needs 3DINO repo at ./3DINO next to this script (sys.path inserted above)
        # NOTE: the exact constructor name can vary by your 3DINO repo version.
        # If you get an import/attribute error here, open 3DINO/dinov2/models/vision_transformer.py
        # and use the corresponding vit_*_3d factory that exists in your checkout.
        from dinov2.models import vision_transformer as vits

        backbone = vits.vit_large_3d(patch_size=16, in_chans=1)
        model = Dino3DClassifier(backbone=backbone, embed_dim=backbone.embed_dim, num_classes=num_classes)

    elif name == "m3t":
        from M3T import M3T
        model = M3T(in_channels=1, out_channels=16, emb_size=128, depth=4, n_classes=num_classes)

    elif name in ("resnet3d", "resnet"):
        from resnet import ResNet, BasicBlock, Bottleneck

        class ResNet3DCls(ResNet):
            def __init__(self, depth=18, num_classes=5):
                if depth in (10, 18, 34):
                    block = BasicBlock
                else:
                    block = Bottleneck
                layers_dict = {
                    10: [1, 1, 1, 1],
                    18: [2, 2, 2, 2],
                    34: [3, 4, 6, 3],
                    50: [3, 4, 6, 3],
                    101: [3, 4, 23, 3],
                }
                layers = layers_dict[depth]
                super().__init__(
                    block=block,
                    layers=layers,
                    sample_input_D=128,
                    sample_input_H=128,
                    sample_input_W=128,
                    num_seg_classes=2,
                    shortcut_type="B",
                    no_cuda=False,
                )
                self.conv_seg = nn.Identity()
                self.global_pool = nn.AdaptiveAvgPool3d(1)
                self.fc = nn.Linear(512 * block.expansion, num_classes)

            def forward(self, x, return_emb: bool = False):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.global_pool(x)
                emb = x.view(x.size(0), -1)
                logits = self.fc(emb)
                if return_emb:
                    return logits, emb
                return logits

        model = ResNet3DCls(depth=18, num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model name: {name}")

    return model



class DualHeadModel(nn.Module):
    """
    backbone -> KL logits (5)
    OA head: Linear(5 -> 2) on top of KL logits
    """
    def __init__(self, backbone: nn.Module, num_classes_5: int = 5, num_classes_2: int = 2):
        super().__init__()
        self.backbone = backbone
        self.num_classes_5 = num_classes_5
        self.binary_head = nn.Linear(num_classes_5, num_classes_2)

    def forward(self, x, return_emb: bool = False):
        # backbone may support return_emb (e.g., Dino3DClassifier/ResNet3DCls)
        if return_emb:
            try:
                logits_5, emb = self.backbone(x, return_emb=True)
            except TypeError:
                logits_5 = self.backbone(x)
                emb = logits_5
            logits_2 = self.binary_head(logits_5)
            return logits_5, logits_2, emb

        logits_5 = self.backbone(x)           # [B,5]
        logits_2 = self.binary_head(logits_5) # [B,2]
        return logits_5, logits_2


# =======================
#   Prob helpers
# =======================

def sigmoid_or_softmax_oa_prob(logits):
    """
    logits: [B,1] -> sigmoid
            [B,2] -> softmax[:,1]
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B,C], got {logits.shape}")
    if logits.shape[1] == 1:
        prob_oa = torch.sigmoid(logits).squeeze(1)
    elif logits.shape[1] == 2:
        prob_oa = torch.softmax(logits, dim=1)[:, 1]
    else:
        raise ValueError(f"Binary head expects C=1 or 2, got C={logits.shape[1]}")
    pred_oa = (prob_oa >= 0.5).long()
    return prob_oa, pred_oa


def kl_softmax_probs_preds(logits_5):
    probs_5 = torch.softmax(logits_5, dim=1)
    preds_5 = probs_5.argmax(dim=1)
    return probs_5, preds_5


def kl_probs_to_oa_prob(probs_5, binary_threshold=2):
    # OA = KL>=2 by default
    return probs_5[:, binary_threshold:].sum(dim=1)


# =======================
#   Input Saliency (same style as your original)
# =======================

def _score_from_model(model, x, head_mode: str, sal_target: str, class_idx=None):
    """
    Returns a scalar score for backprop (logit).
    head_mode: single_kl / single_oa / dual
    sal_target:
      - for single_kl: must be 'kl'
      - for single_oa: must be 'oa'
      - for dual: 'kl' or 'oa'
    class_idx: for KL, which class logit to use; None -> argmax
    """
    out = model(x)

    if head_mode == "single_kl":
        logits_5 = out
        if logits_5.shape[1] != 5:
            raise ValueError(f"single_kl expects 5 logits, got {logits_5.shape}")
        if class_idx is None:
            class_idx = logits_5.argmax(dim=1).item()
        return logits_5[0, int(class_idx)]

    if head_mode == "single_oa":
        logits = out
        if logits.shape[1] == 1:
            return logits[0, 0]
        if logits.shape[1] == 2:
            return logits[0, 1]  # OA positive
        raise ValueError(f"single_oa expects 1 or 2 logits, got {logits.shape}")

    if head_mode == "dual":
        logits_5, logits_2 = out
        if sal_target == "kl":
            if class_idx is None:
                class_idx = logits_5.argmax(dim=1).item()
            return logits_5[0, int(class_idx)]
        elif sal_target == "oa":
            if logits_2.shape[1] == 1:
                return logits_2[0, 0]
            if logits_2.shape[1] == 2:
                return logits_2[0, 1]
            raise ValueError(f"dual oa head expects 1 or 2 logits, got {logits_2.shape}")
        else:
            raise ValueError("dual sal_target must be 'kl' or 'oa'")

    raise ValueError(f"Unknown head_mode: {head_mode}")


class InputSaliency3D:
    """
    Same idea as your original:
      - method = "grad"       : |∂score/∂x|
      - method = "grad*input" : |x * ∂score/∂x|
      - smooth > 0            : SmoothGrad (n samples)
    """
    def __init__(self, model, head_mode: str):
        self.model = model
        self.head_mode = head_mode

    def generate(
        self,
        x,
        sal_target: str,
        class_idx=None,
        method="grad*input",
        smooth=8,
        noise_std=0.05,
    ):
        """
        x: [1,1,D,H,W] on device
        Returns: saliency [D,H,W] numpy in [0,1]
        """
        device = x.device
        self.model.eval()

        n_samples = max(1, int(smooth) if smooth is not None else 1)
        grad_acc = torch.zeros_like(x, device=device)

        for _ in range(n_samples):
            if smooth and noise_std > 0:
                noise = torch.randn_like(x) * noise_std
                x_pert = (x + noise).clone().detach().requires_grad_(True)
            else:
                x_pert = x.clone().detach().requires_grad_(True)

            score = _score_from_model(
                self.model, x_pert,
                head_mode=self.head_mode,
                sal_target=sal_target,
                class_idx=class_idx,
            )

            self.model.zero_grad(set_to_none=True)
            score.backward()

            grad = x_pert.grad  # [1,1,D,H,W]
            if method == "grad*input":
                grad = grad * x_pert
            elif method == "grad":
                pass
            else:
                raise ValueError(f"Unknown method: {method}")

            grad_acc += grad.detach()

        grad_acc /= n_samples
        sal = grad_acc[0, 0].abs()  # [D,H,W]

        sal_min, sal_max = sal.min(), sal.max()
        if (sal_max - sal_min) > 1e-6:
            sal = (sal - sal_min) / (sal_max - sal_min)
        else:
            sal = torch.zeros_like(sal)

        return sal.cpu().numpy().astype(np.float32)


def save_oaizib_input_saliency_for_one_case(
    model,
    device,
    head_mode: str,
    excel_path,
    img_root,
    resize,
    cmt_id: int,
    out_path: str,
    sal_target: str,
    class_idx=None,
    method="grad*input",
    smooth=8,
    noise_std=0.05,
):
    """
    Same output style as your original:
      - generate saliency at resized volume
      - interpolate back to original (D0,H0,W0)
      - transpose to [H0,W0,D0]
      - save NIfTI with original affine/header
    """
    df = pd.read_excel(excel_path)
    rows = df[df["CMT-ID"] == cmt_id]
    if rows.empty:
        print(f"[Saliency] CMT-ID={cmt_id} not found, skip.")
        return

    fname = f"oaizib_{cmt_id:03d}_0000_n4_z.nii.gz"
    img_path = os.path.join(img_root, fname)

    vol = load_nii_as_tensor(img_path, resize=resize)  # [1,D,H,W]
    x = vol.unsqueeze(0).to(device)                    # [1,1,D,H,W]

    sal_gen = InputSaliency3D(model, head_mode=head_mode)
    salmap = sal_gen.generate(
        x,
        sal_target=sal_target,
        class_idx=class_idx,
        method=method,
        smooth=smooth,
        noise_std=noise_std,
    )  # [D_res,H_res,W_res]

    nii_orig = nib.load(img_path)
    orig_data = nii_orig.get_fdata()
    H0, W0, D0 = orig_data.shape  # nib: (H,W,D)

    sal_t = torch.from_numpy(salmap)[None, None, ...].to(device)  # [1,1,D,H,W]
    sal_res = F.interpolate(
        sal_t,
        size=(D0, H0, W0),
        mode="trilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0).detach().cpu().numpy()  # [D0,H0,W0]

    sal_res = np.transpose(sal_res, (1, 2, 0))  # [H0,W0,D0]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sal_nii = nib.Nifti1Image(sal_res.astype(np.float32), nii_orig.affine, nii_orig.header)
    nib.save(sal_nii, out_path)
    print(f"[Saliency] Saved to {out_path} (shape={sal_res.shape})")


# =======================
#   Evaluation + CSV export
# =======================

def eval_and_export(
    model,
    loader,
    device,
    head_mode: str,
    out_prefix: str,
    binary_threshold: int = 2,
    do_plot: bool = True,
    model_name: str = "",
    save_manifold: bool = False,
    manifold_method: str = "tsne",
    manifold_color_by: str = "label_oa",
    manifold_max_n: int = 2000,
    manifold_seed: int = 42,
    manifold_perplexity: float = 30.0,
):
    model.eval()
    rows = []
    emb_rows = []  # per-sample embedding vectors (for neural manifold)

    all_label_oa, all_prob_oa, all_pred_oa = [], [], []
    all_label_kl, all_pred_kl, all_probs_kl = [], [], []

    with torch.no_grad():
        for vol, cmt_id, kl, img_path in tqdm(loader, desc="Eval OAIZIB"):
            vol = vol.to(device)  # [B,1,D,H,W]
            kl_t = torch.tensor(kl, device=device).long()
            label_oa = (kl_t >= binary_threshold).long()

            if head_mode == "single_kl":
                if save_manifold:
                    try:
                        logits_5, emb = model(vol, return_emb=True)
                    except TypeError:
                        logits_5, emb = _forward_with_penultimate_embedding(model, vol)
                        # if model returned tuple, keep first as KL logits
                        if isinstance(logits_5, (tuple, list)):
                            logits_5 = logits_5[0]
                else:
                    logits_5 = model(vol)
                    emb = None
                probs_5, pred_kl = kl_softmax_probs_preds(logits_5)
                prob_oa = kl_probs_to_oa_prob(probs_5, binary_threshold=binary_threshold)
                pred_oa = (prob_oa >= 0.5).long()

                all_label_kl.append(kl_t.cpu().numpy())
                all_pred_kl.append(pred_kl.cpu().numpy())
                all_probs_kl.append(probs_5.cpu().numpy())

                probs_5_np = probs_5.cpu().numpy()
                for i in range(vol.size(0)):
                    r = {
                        "CMT_ID": int(cmt_id[i]),
                        "img_path": str(img_path[i]),
                        "label_kl": int(kl_t[i].item()),
                        "pred_kl": int(pred_kl[i].item()),
                        "label_oa": int(label_oa[i].item()),
                        "prob_oa": float(prob_oa[i].item()),
                        "pred_oa": int(pred_oa[i].item()),
                    }
                    for c in range(5):
                        r[f"prob_kl_{c}"] = float(probs_5_np[i, c])
                    rows.append(r)
                    if save_manifold:
                        if emb is not None:
                            emb_rows.append(emb[i].detach().cpu().numpy())

            elif head_mode == "single_oa":
                if save_manifold:
                    try:
                        logits, emb = model(vol, return_emb=True)
                    except TypeError:
                        logits, emb = _forward_with_penultimate_embedding(model, vol)
                        if isinstance(logits, (tuple, list)):
                            logits = logits[0]
                else:
                    logits = model(vol)
                    emb = None
                prob_oa, pred_oa = sigmoid_or_softmax_oa_prob(logits)

                for i in range(vol.size(0)):
                    rows.append(
                        {
                            "CMT_ID": int(cmt_id[i]),
                            "img_path": str(img_path[i]),
                            "label_kl": int(kl_t[i].item()),
                            "label_oa": int(label_oa[i].item()),
                            "prob_oa": float(prob_oa[i].item()),
                            "pred_oa": int(pred_oa[i].item()),
                        }
                    )
                    if save_manifold:
                        if emb is not None:
                            emb_rows.append(emb[i].detach().cpu().numpy())

            elif head_mode == "dual":
                emb = None
                if save_manifold:
                    logits_5, logits_2, emb = _forward_dual_with_backbone_embedding(model, vol)
                else:
                    logits_5, logits_2 = model(vol)
                probs_5, pred_kl = kl_softmax_probs_preds(logits_5)
                prob_oa_from_kl = kl_probs_to_oa_prob(probs_5, binary_threshold=binary_threshold)
                prob_oa, pred_oa = sigmoid_or_softmax_oa_prob(logits_2)

                all_label_kl.append(kl_t.cpu().numpy())
                all_pred_kl.append(pred_kl.cpu().numpy())
                all_probs_kl.append(probs_5.cpu().numpy())

                probs_5_np = probs_5.cpu().numpy()
                for i in range(vol.size(0)):
                    r = {
                        "CMT_ID": int(cmt_id[i]),
                        "img_path": str(img_path[i]),
                        "label_kl": int(kl_t[i].item()),
                        "pred_kl": int(pred_kl[i].item()),
                        "label_oa": int(label_oa[i].item()),
                        "prob_oa": float(prob_oa[i].item()),
                        "pred_oa": int(pred_oa[i].item()),
                        "prob_oa_from_kl": float(prob_oa_from_kl[i].item()),
                    }
                    for c in range(5):
                        r[f"prob_kl_{c}"] = float(probs_5_np[i, c])
                    rows.append(r)
                    if save_manifold:
                        if emb is not None:
                            emb_rows.append(emb[i].detach().cpu().numpy())
            else:
                raise ValueError(f"Unknown head_mode: {head_mode}")

            all_label_oa.append(label_oa.cpu().numpy())
            all_prob_oa.append(prob_oa.detach().cpu().numpy())
            all_pred_oa.append(pred_oa.detach().cpu().numpy())

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    df_preds = pd.DataFrame(rows)
    preds_csv = out_prefix + "_preds.csv"
    df_preds.to_csv(preds_csv, index=False)
    print("Saved per-sample predictions to:", preds_csv)

    # ---- neural manifold ----
    if save_manifold:
        _save_neural_manifold(
            embeddings_np=emb_rows,
            meta_df=df_preds,
            out_prefix=out_prefix,
            method=manifold_method,
            color_by=manifold_color_by,
            max_n=manifold_max_n,
            seed=manifold_seed,
            perplexity=manifold_perplexity,
        )

    y_oa = np.concatenate(all_label_oa)
    p_oa = np.concatenate(all_prob_oa)
    pr_oa = np.concatenate(all_pred_oa)

    metrics = {}
    metrics["acc_oa"] = float(accuracy_score(y_oa, pr_oa))
    try:
        metrics["auc_oa"] = float(roc_auc_score(y_oa, p_oa))
    except Exception:
        metrics["auc_oa"] = float("nan")
    metrics["precision_oa"] = float(precision_score(y_oa, pr_oa, zero_division=0))
    metrics["recall_oa"] = float(recall_score(y_oa, pr_oa, zero_division=0))
    metrics["f1_oa"] = float(f1_score(y_oa, pr_oa, zero_division=0))

    cm_oa = confusion_matrix(y_oa, pr_oa)
    print("\n[OA] Confusion matrix:\n", cm_oa)
    print("[OA] Classification report:\n", classification_report(y_oa, pr_oa, digits=3))

    has_kl = (head_mode in ("single_kl", "dual"))
    if has_kl:
        y_kl = np.concatenate(all_label_kl)
        pr_kl = np.concatenate(all_pred_kl)
        pb_kl = np.concatenate(all_probs_kl)

        metrics["acc_kl"] = float(accuracy_score(y_kl, pr_kl))
        metrics["precision_macro_kl"] = float(precision_score(y_kl, pr_kl, average="macro", zero_division=0))
        metrics["recall_macro_kl"] = float(recall_score(y_kl, pr_kl, average="macro", zero_division=0))
        metrics["f1_macro_kl"] = float(f1_score(y_kl, pr_kl, average="macro", zero_division=0))

        y_onehot = np.eye(5)[y_kl]
        try:
            metrics["auc_macro_kl"] = float(roc_auc_score(y_onehot, pb_kl, multi_class="ovr", average="macro"))
        except Exception:
            metrics["auc_macro_kl"] = float("nan")

        cm_kl = confusion_matrix(y_kl, pr_kl)
        print("\n[KL] Confusion matrix:\n", cm_kl)
        print("[KL] Classification report:\n", classification_report(y_kl, pr_kl, digits=3))

    df_metrics = pd.DataFrame([{
        "head_mode": head_mode,
        "binary_threshold": binary_threshold,
        **metrics
    }])
    metrics_csv = out_prefix + "_metrics.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    print("Saved metrics to:", metrics_csv)

    if do_plot:
        try:
            import matplotlib.pyplot as plt
            # OA ROC
            fpr, tpr, _ = roc_curve(y_oa, p_oa)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={metrics['auc_oa']:.3f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Binary ROC (OA vs non-OA)")
            plt.legend(loc="lower right")
            plt.savefig(out_prefix + "_roc_oa.png", dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print("ROC plot (OA) failed:", e)

        try:
            import matplotlib.pyplot as plt
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_oa, display_labels=["non-OA", "OA"])
            fig, ax = plt.subplots(figsize=(5, 5))
            disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
            plt.title("Confusion Matrix (OA)")
            plt.savefig(out_prefix + "_cm_oa.png", dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print("Confusion matrix plot (OA) failed:", e)

        if has_kl:
            try:
                import matplotlib.pyplot as plt
                y_kl = np.concatenate(all_label_kl)
                pb_kl = np.concatenate(all_probs_kl)
                y_onehot = np.eye(5)[y_kl]
                plt.figure()
                for c in range(5):
                    fpr, tpr, _ = roc_curve(y_onehot[:, c], pb_kl[:, c])
                    try:
                        auc_c = roc_auc_score(y_onehot[:, c], pb_kl[:, c])
                    except Exception:
                        auc_c = np.nan
                    plt.plot(fpr, tpr, label=f"Class {c} (AUC={auc_c:.3f})")
                plt.plot([0, 1], [0, 1], "k--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("Multi-class ROC (KL 0-4)")
                plt.legend(loc="lower right")
                plt.savefig(out_prefix + "_roc_kl5.png", dpi=300, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print("ROC plot (KL) failed:", e)

            try:
                import matplotlib.pyplot as plt
                pr_kl = np.concatenate(all_pred_kl)
                cm_kl = confusion_matrix(y_kl, pr_kl)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm_kl, display_labels=[str(i) for i in range(5)])
                fig, ax = plt.subplots(figsize=(5, 5))
                disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
                plt.title("Confusion Matrix (KL 0-4)")
                plt.savefig(out_prefix + "_cm_kl5.png", dpi=300, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print("Confusion matrix plot (KL) failed:", e)

    return df_preds, df_metrics


# =======================
#   main
# =======================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, choices=["mamba", "m3t", "resnet3d", "3dino"], required=True)
    parser.add_argument("--ckpt", type=str, required=True)

    parser.add_argument("--head_mode", type=str, choices=["single_kl", "single_oa", "dual"], required=True)

    parser.add_argument("--oaizib_excel", type=str, required=True)
    parser.add_argument("--oaizib_img_root", type=str, required=True)

    parser.add_argument("--input_D", type=int, default=160)
    parser.add_argument("--input_H", type=int, default=256)
    parser.add_argument("--input_W", type=int, default=256)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="./eval_results")

    parser.add_argument("--binary_threshold", type=int, default=2, help="KL>=threshold -> OA=1")
    parser.add_argument("--no_plot", action="store_true")

    # ---- saliency (NIfTI, original style) ----
    parser.add_argument("--save_saliency", action="store_true",
                        help="save InputSaliency3D as NIfTI (same style as original eval)")
    parser.add_argument("--saliency_cmt_ids", type=int, nargs="+", default=[],
                        help="CMT-IDs to export saliency, e.g. --saliency_cmt_ids 440 12 99")
    parser.add_argument("--saliency_target", type=str, default="auto",
                        choices=["auto", "oa", "kl"],
                        help="for dual: auto -> export both; otherwise export one head only")
    parser.add_argument("--saliency_method", type=str, default="grad*input",
                        choices=["grad", "grad*input"])
    parser.add_argument("--saliency_smooth", type=int, default=8)
    parser.add_argument("--saliency_noise_std", type=float, default=0.05)
    parser.add_argument("--saliency_class_idx", type=int, default=-1,
                        help="for KL saliency: -1 means argmax; otherwise use specific class 0-4")
    # ---- neural manifold (embeddings -> 2D projection) ----
    parser.add_argument("--save_manifold", action="store_true",
                        help="save per-sample embeddings + 2D manifold (t-SNE/PCA) plots")
    parser.add_argument("--manifold_method", type=str, default="tsne", choices=["tsne", "pca"])
    parser.add_argument("--manifold_color_by", type=str, default="label_oa",
                        help="column name in preds csv to color by (e.g., label_oa, pred_oa, label_kl, pred_kl)")
    parser.add_argument("--manifold_max_n", type=int, default=2000)
    parser.add_argument("--manifold_seed", type=int, default=42)
    parser.add_argument("--manifold_perplexity", type=float, default=30.0)



    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # M3T fixed 128^3
    resize = (128, 128, 128) if args.model == "m3t" else (args.input_D, args.input_H, args.input_W)

    # build model
    if args.head_mode == "single_kl":
        model = build_backbone(args.model, num_classes=5)
    elif args.head_mode == "single_oa":
        # default build 2 logits; if your ckpt is 1-logit, set num_classes=1 and change this line accordingly
        model = build_backbone(args.model, num_classes=2)
    elif args.head_mode == "dual":
        backbone = build_backbone(args.model, num_classes=5)
        model = DualHeadModel(backbone, num_classes_5=5, num_classes_2=2)
    else:
        raise ValueError(args.head_mode)

    model = model.to(device)

    # ---- load checkpoint ----
    ckpt_obj = torch.load(args.ckpt, map_location="cpu")
    state = ckpt_obj
    # support common checkpoint formats
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            state = ckpt_obj["state_dict"]
        elif "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            state = ckpt_obj["model_state_dict"]
        elif "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            state = ckpt_obj["model"]

    # strip common prefixes (e.g., 'module.')
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            kk = k
            if kk.startswith("module."):
                kk = kk[len("module."):]
            new_state[kk] = v
        state = new_state

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] missing keys (first 30):", missing[:30])
    if unexpected:
        print("[WARN] unexpected keys (first 30):", unexpected[:30])

    model.eval()
    print("Loaded ckpt:", args.ckpt)

    out_prefix = os.path.join(args.out_dir, f"{args.model}_{args.head_mode}")

    # ---- saliency export first (optional) ----
    if args.save_saliency:
        if len(args.saliency_cmt_ids) == 0:
            print("[Saliency] --save_saliency is on, but no --saliency_cmt_ids provided. Skip saliency.")
        else:
            for cmt_id in args.saliency_cmt_ids:
                class_idx = None if args.saliency_class_idx < 0 else int(args.saliency_class_idx)

                if args.head_mode == "single_kl":
                    out_path = out_prefix + f"_SAL_KL_cmt{cmt_id:03d}.nii.gz"
                    save_oaizib_input_saliency_for_one_case(
                        model=model, device=device, head_mode=args.head_mode,
                        excel_path=args.oaizib_excel, img_root=args.oaizib_img_root, resize=resize,
                        cmt_id=cmt_id, out_path=out_path,
                        sal_target="kl",
                        class_idx=class_idx,
                        method=args.saliency_method,
                        smooth=args.saliency_smooth,
                        noise_std=args.saliency_noise_std,
                    )

                elif args.head_mode == "single_oa":
                    out_path = out_prefix + f"_SAL_OA_cmt{cmt_id:03d}.nii.gz"
                    save_oaizib_input_saliency_for_one_case(
                        model=model, device=device, head_mode=args.head_mode,
                        excel_path=args.oaizib_excel, img_root=args.oaizib_img_root, resize=resize,
                        cmt_id=cmt_id, out_path=out_path,
                        sal_target="oa",
                        class_idx=None,
                        method=args.saliency_method,
                        smooth=args.saliency_smooth,
                        noise_std=args.saliency_noise_std,
                    )

                elif args.head_mode == "dual":
                    # auto -> both
                    if args.saliency_target in ("auto", "kl"):
                        out_path = out_prefix + f"_SAL_KL_cmt{cmt_id:03d}.nii.gz"
                        save_oaizib_input_saliency_for_one_case(
                            model=model, device=device, head_mode=args.head_mode,
                            excel_path=args.oaizib_excel, img_root=args.oaizib_img_root, resize=resize,
                            cmt_id=cmt_id, out_path=out_path,
                            sal_target="kl",
                            class_idx=class_idx,
                            method=args.saliency_method,
                            smooth=args.saliency_smooth,
                            noise_std=args.saliency_noise_std,
                        )
                    if args.saliency_target in ("auto", "oa"):
                        out_path = out_prefix + f"_SAL_OA_cmt{cmt_id:03d}.nii.gz"
                        save_oaizib_input_saliency_for_one_case(
                            model=model, device=device, head_mode=args.head_mode,
                            excel_path=args.oaizib_excel, img_root=args.oaizib_img_root, resize=resize,
                            cmt_id=cmt_id, out_path=out_path,
                            sal_target="oa",
                            class_idx=None,
                            method=args.saliency_method,
                            smooth=args.saliency_smooth,
                            noise_std=args.saliency_noise_std,
                        )

    # ---- dataset / loader ----
    ds = OAIZIBDataset(args.oaizib_excel, args.oaizib_img_root, resize=resize)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    # ---- eval + export ----
    eval_and_export(
        model=model,
        loader=dl,
        device=device,
        head_mode=args.head_mode,
        out_prefix=out_prefix,
        binary_threshold=args.binary_threshold,
        do_plot=(not args.no_plot),
        model_name=args.model,
        save_manifold=args.save_manifold,
        manifold_method=args.manifold_method,
        manifold_color_by=args.manifold_color_by,
        manifold_max_n=args.manifold_max_n,
        manifold_seed=args.manifold_seed,
        manifold_perplexity=args.manifold_perplexity,
    )


if __name__ == "__main__":
    main()
