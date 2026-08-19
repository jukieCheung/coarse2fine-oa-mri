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

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ===== Dataset =====

class OAIZIBDataset(Dataset):
    """
    OAIZIB-CM dataset loader.
    假设 Excel 里有列 'CMT-ID'，图像命名为:
        oaizib_{CMT-ID:03d}_0000_n4_z.nii.gz
    标签列: 'KLGrade' (0-4).
    """

    def __init__(self, excel_path, img_root, resize):
        super().__init__()
        self.df = pd.read_excel(excel_path)
        self.img_root = img_root
        self.resize = resize

        self.samples = []
        for _, row in self.df.iterrows():
            cmt_id = int(row["CMT-ID"])
            kl = int(row["KLGrade"])
            fname = f"oaizib_{cmt_id:03d}_0000_n4_z.nii.gz"
            fpath = os.path.join(img_root, fname)
            if not os.path.isfile(fpath):
                raise FileNotFoundError(f"Image not found: {fpath}")
            self.samples.append((fpath, kl))

    def __len__(self):
        return len(self.samples)

    def _load_nii(self, path):
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        # 简单 z-score 强度归一化
        mean = data.mean()
        std = data.std()
        if std < 1e-6:
            std = 1e-6
        data = (data - mean) / std

        vol = torch.from_numpy(data)      # nifti 一般是 [H, W, D]
        if vol.ndim != 3:
            raise ValueError(f"Expect 3D volume, got shape {vol.shape}")
        # 约定 [H,W,D] -> [D,H,W]
        vol = vol.permute(2, 0, 1)
        vol = vol.unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]

        if self.resize is not None:
            vol = F.interpolate(
                vol,
                size=self.resize,   # (D,H,W)
                mode="trilinear",
                align_corners=False,
            )

        vol = vol.squeeze(0)  # [1,D,H,W]
        return vol

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        vol = self._load_nii(img_path)
        return vol, label


# ===== Backbone builder（仍然只建 5 类 KL 模型） =====

def build_backbone(name: str, num_classes: int):
    name = name.lower()
    if name == "mamba":
        from nnMamba4cls import nnMambaEncoder
        model = nnMambaEncoder(
            in_ch=1, channels=32, blocks=3, number_classes=num_classes
        )
    elif name == "m3t":
        from M3T import M3T
        model = M3T(
            in_channels=1,
            out_channels=16,
            emb_size=128,
            depth=4,
            n_classes=num_classes,
        )
    elif name in ("resnet3d", "resnet"):
        from resnet import ResNet, BasicBlock, Bottleneck

        class ResNet3DCls(ResNet):
            """在原 3D ResNet 分割的基础上改成分类头"""

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

                # 调用原 ResNet 的 __init__，conv_seg 之后会被我们 ignore
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

                # 把分割头换成分类头
                self.conv_seg = nn.Identity()
                self.global_pool = nn.AdaptiveAvgPool3d(1)
                self.fc = nn.Linear(512 * block.expansion, num_classes)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.global_pool(x)  # [B,C,1,1,1]
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        model = ResNet3DCls(depth=18, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return model


# ===== Dual-head wrapper：5 类 KL + 2 类 OA/非 OA =====

class DualHeadModel(nn.Module):
    """
    包一个现有的 KL-5 类 backbone，再加一个独立的 2 类头。
    这里为了不改 backbone 内部结构，二分类头直接基于 5 类 logits。
    """
    def __init__(self, backbone: nn.Module, num_classes_5: int = 5, num_classes_2: int = 2):
        super().__init__()
        self.backbone = backbone
        self.num_classes_5 = num_classes_5
        self.binary_head = nn.Linear(num_classes_5, num_classes_2)

    def forward(self, x):
        logits_5 = self.backbone(x)          # [B,5]
        logits_2 = self.binary_head(logits_5)  # [B,2]
        return logits_5, logits_2


# ===== Training & evaluation =====

def train_one_epoch(
    model,
    loader,
    criterion_5,
    criterion_2,
    optimizer,
    device,
    epoch,
    binary_threshold=2,
    lambda_bin=1.0,
):
    model.train()
    running_loss = 0.0
    running_loss_5 = 0.0
    running_loss_2 = 0.0

    for imgs, labels_kl in tqdm(loader, desc=f"Epoch {epoch}"):
        imgs = imgs.to(device)  # [B,1,D,H,W]
        labels_kl = labels_kl.to(device).long()  # 0..4

        # KL -> 二分类 label
        labels_bin = (labels_kl >= binary_threshold).long()  # 0/1

        optimizer.zero_grad()
        logits_5, logits_2 = model(imgs)
        loss_5 = criterion_5(logits_5, labels_kl)
        loss_2 = criterion_2(logits_2, labels_bin)
        loss = loss_5 + lambda_bin * loss_2
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_loss_5 += loss_5.item() * bs
        running_loss_2 += loss_2.item() * bs

    n = len(loader.dataset)
    avg_loss = running_loss / n
    avg_loss_5 = running_loss_5 / n
    avg_loss_2 = running_loss_2 / n
    print(
        f"Epoch {epoch}: total loss={avg_loss:.4f}, KL-5 CE={avg_loss_5:.4f}, bin CE={avg_loss_2:.4f}"
    )
    return avg_loss


def evaluate(model, loader, device, num_classes_5, out_prefix, binary_threshold=2):
    model.eval()
    all_labels_kl, all_probs_5, all_preds_5 = [], [], []
    all_labels_bin, all_probs_bin, all_preds_bin = [], [], []

    with torch.no_grad():
        for imgs, labels_kl in tqdm(loader, desc="Eval"):
            imgs = imgs.to(device)
            labels_kl = labels_kl.to(device).long()
            labels_bin = (labels_kl >= binary_threshold).long()

            logits_5, logits_2 = model(imgs)

            # 5 类 KL
            probs_5 = torch.softmax(logits_5, dim=1)
            preds_5 = probs_5.argmax(dim=1)

            # 2 类 OA/非 OA
            probs_2 = torch.softmax(logits_2, dim=1)
            probs_oa = probs_2[:, 1]          # 预测 OA 概率
            preds_bin = probs_oa.ge(0.5).long()

            all_labels_kl.append(labels_kl.cpu().numpy())
            all_probs_5.append(probs_5.cpu().numpy())
            all_preds_5.append(preds_5.cpu().numpy())

            all_labels_bin.append(labels_bin.cpu().numpy())
            all_probs_bin.append(probs_oa.cpu().numpy())
            all_preds_bin.append(preds_bin.cpu().numpy())

    all_labels_kl = np.concatenate(all_labels_kl)
    all_probs_5 = np.concatenate(all_probs_5)
    all_preds_5 = np.concatenate(all_preds_5)

    all_labels_bin = np.concatenate(all_labels_bin)
    all_probs_bin = np.concatenate(all_probs_bin)
    all_preds_bin = np.concatenate(all_preds_bin)

    metrics = {}

    # ===== 5 类 KL 指标 =====
    metrics["acc_5"] = accuracy_score(all_labels_kl, all_preds_5)
    y_onehot_5 = np.eye(num_classes_5)[all_labels_kl]
    try:
        auc_macro_5 = roc_auc_score(
            y_onehot_5, all_probs_5, multi_class="ovr", average="macro"
        )
    except Exception:
        auc_macro_5 = np.nan
    precision_5 = precision_score(all_labels_kl, all_preds_5, average="macro")
    recall_5 = recall_score(all_labels_kl, all_preds_5, average="macro")
    f1_5 = f1_score(all_labels_kl, all_preds_5, average="macro")
    metrics.update(
        {
            "auc_macro_5": auc_macro_5,
            "precision_macro_5": precision_5,
            "recall_macro_5": recall_5,
            "f1_macro_5": f1_5,
        }
    )

    # ===== 二分类 OA 指标 =====
    metrics["acc_bin"] = accuracy_score(all_labels_bin, all_preds_bin)
    try:
        auc_bin = roc_auc_score(all_labels_bin, all_probs_bin)
    except Exception:
        auc_bin = np.nan
    precision_bin = precision_score(all_labels_bin, all_preds_bin)
    recall_bin = recall_score(all_labels_bin, all_preds_bin)
    f1_bin = f1_score(all_labels_bin, all_preds_bin)
    metrics.update(
        {
            "auc_bin": auc_bin,
            "precision_bin": precision_bin,
            "recall_bin": recall_bin,
            "f1_bin": f1_bin,
        }
    )

    # ===== 画 ROC =====
    try:
        import matplotlib.pyplot as plt

        # 5 类 ROC
        plt.figure()
        for c in range(num_classes_5):
            fpr, tpr, _ = roc_curve(y_onehot_5[:, c], all_probs_5[:, c])
            try:
                auc_c = roc_auc_score(y_onehot_5[:, c], all_probs_5[:, c])
            except Exception:
                auc_c = np.nan
            plt.plot(fpr, tpr, label=f"Class {c} (AUC={auc_c:.3f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("False Positive Rate")
        plt.title("Multi-class ROC curves (KL 0-4)")
        plt.legend(loc="lower right")
        plt.savefig(out_prefix + "_roc_5class.png", dpi=300)
        plt.close()

        # 二分类 ROC
        fpr, tpr, _ = roc_curve(all_labels_bin, all_probs_bin)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc_bin:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Binary OA vs non-OA (KL>={binary_threshold})")
        plt.legend(loc="lower right")
        plt.savefig(out_prefix + "_roc_binary.png", dpi=300)
        plt.close()
    except Exception as e:
        print("ROC plot failed:", e)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_excel", type=str, required=True)
    parser.add_argument("--test_excel", type=str, required=True)
    parser.add_argument("--img_root", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        choices=["mamba", "m3t", "resnet3d"],
        default="mamba",
    )
    parser.add_argument("--num_classes", type=int, default=5)  # KL 0-4
    # 你的体数据目前是 256x256x160，这里保持这个默认
    parser.add_argument("--input_D", type=int, default=160)
    parser.add_argument("--input_H", type=int, default=256)
    parser.add_argument("--input_W", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="./results_dualhead")
    parser.add_argument(
        "--binary_threshold",
        type=int,
        default=2,
        help="KL >= threshold 视作 OA=1 (binary)",
    )
    parser.add_argument(
        "--lambda_bin",
        type=float,
        default=1.0,
        help="二分类辅助 loss 的权重",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ⚠️ M3T 的结构强假设输入是 N=N=N=128，所以这里自动把它重采样成 128³
    if args.model == "m3t":
        resize = (128, 128, 128)
    else:
        resize = (args.input_D, args.input_H, args.input_W)

    train_dataset = OAIZIBDataset(args.train_excel, args.img_root, resize=resize)
    test_dataset = OAIZIBDataset(args.test_excel, args.img_root, resize=resize)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    backbone = build_backbone(args.model, args.num_classes)
    model = DualHeadModel(backbone, num_classes_5=args.num_classes, num_classes_2=2)
    model = model.to(device)

    criterion_5 = nn.CrossEntropyLoss()
    criterion_2 = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            model,
            train_loader,
            criterion_5,
            criterion_2,
            optimizer,
            device,
            epoch,
            binary_threshold=args.binary_threshold,
            lambda_bin=args.lambda_bin,
        )

    # 保存模型
    ckpt_path = os.path.join(args.out_dir, f"{args.model}_dualhead_final.pth")
    torch.save(model.state_dict(), ckpt_path)
    print("Saved model to", ckpt_path)

    # 在测试集上推理 + 计算 5 类和 2 类的指标
    metrics = evaluate(
        model,
        test_loader,
        device,
        args.num_classes,
        out_prefix=os.path.join(args.out_dir, f"{args.model}_test"),
        binary_threshold=args.binary_threshold,
    )
    print("Test metrics:", metrics)


if __name__ == "__main__":
    main()
