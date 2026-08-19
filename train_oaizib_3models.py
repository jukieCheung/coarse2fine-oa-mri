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


# ===== Models wrapper =====

def build_model(name: str, num_classes: int):
    name = name.lower()
    if name == "mamba":
        from nnMamba4cls import nnMambaEncoder
        model = nnMambaEncoder(
            in_ch=1, channels=32, blocks=3, number_classes=num_classes
        )
    elif name == "m3t":
        from M3T import M3T
        model = M3T(
            in_channels=1, out_channels=16,
            emb_size=128, depth=4, n_classes=num_classes
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


# ===== Training & evaluation =====

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(loader, desc=f"Epoch {epoch}"):
        imgs = imgs.to(device)  # [B,1,D,H,W]
        labels = labels.to(device).long()

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    avg_loss = running_loss / len(loader.dataset)
    return avg_loss


def evaluate(model, loader, device, num_classes, out_prefix):
    model.eval()
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Eval"):
            imgs = imgs.to(device)
            labels = labels.to(device).long()
            logits = model(imgs)

            if num_classes == 1:
                probs = torch.sigmoid(logits).squeeze(1)
                preds = (probs >= 0.5).long()
            else:
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    metrics = {"acc": accuracy_score(all_labels, all_preds)}

    if num_classes == 1:
        auc = roc_auc_score(all_labels, all_probs)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        metrics.update(
            {"auc": auc, "precision": precision, "recall": recall, "f1": f1}
        )

        # ROC
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        try:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC curve")
            plt.legend(loc="lower right")
            plt.savefig(out_prefix + "_roc.png", dpi=300)
            plt.close()
        except Exception as e:
            print("ROC plot failed:", e)
    else:
        # multi-class
        y_onehot = np.eye(num_classes)[all_labels]
        auc_macro = roc_auc_score(
            y_onehot, all_probs, multi_class="ovr", average="macro"
        )
        precision = precision_score(all_labels, all_preds, average="macro")
        recall = recall_score(all_labels, all_preds, average="macro")
        f1 = f1_score(all_labels, all_preds, average="macro")
        metrics.update(
            {
                "auc_macro": auc_macro,
                "precision_macro": precision,
                "recall_macro": recall,
                "f1_macro": f1,
            }
        )

        # 每一类的 ROC
        try:
            import matplotlib.pyplot as plt

            plt.figure()
            for c in range(num_classes):
                fpr, tpr, _ = roc_curve(y_onehot[:, c], all_probs[:, c])
                auc_c = roc_auc_score(y_onehot[:, c], all_probs[:, c])
                plt.plot(fpr, tpr, label=f"Class {c} (AUC={auc_c:.3f})")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Multi-class ROC curves")
            plt.legend(loc="lower right")
            plt.savefig(out_prefix + "_roc.png", dpi=300)
            plt.close()
        except Exception as e:
            print("Multi-class ROC plot failed:", e)

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
    parser.add_argument("--num_classes", type=int, default=5)
    # 你的体数据目前是 256x256x160，这里保持这个默认
    parser.add_argument("--input_D", type=int, default=160)
    parser.add_argument("--input_H", type=int, default=256)
    parser.add_argument("--input_W", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="./results")
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

    model = build_model(args.model, args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        print(f"Epoch {epoch}: train loss = {loss:.4f}")

    # 保存模型
    ckpt_path = os.path.join(args.out_dir, f"{args.model}_final.pth")
    torch.save(model.state_dict(), ckpt_path)
    print("Saved model to", ckpt_path)

    # 在测试集上推理 + 计算 ROC/AUC/ACC/Precision/Recall/F1
    metrics = evaluate(
        model,
        test_loader,
        device,
        args.num_classes,
        out_prefix=os.path.join(args.out_dir, f"{args.model}_test"),
    )
    print("Test metrics:", metrics)


if __name__ == "__main__":
    main()
