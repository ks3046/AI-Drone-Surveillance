import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from fusion_model import AVFusionTransformer


# =========================================================
# DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# =========================================================
# DATASET
# =========================================================
class AVDataset(Dataset):

    def __init__(
        self,
        clip_root="data/clip_features",
        audio_root="data/vggish_features",
        max_len=256
    ):

        self.samples = []
        self.max_len = max_len

        self.classes = {
            "Normal": 0,
            "Fighting": 1,
            "Shooting": 2
        }

        for cls, label in self.classes.items():

            clip_dir = os.path.join(clip_root, cls)
            audio_dir = os.path.join(audio_root, cls)

            if not os.path.exists(clip_dir):
                continue

            files = sorted(os.listdir(clip_dir))

            for file in files:

                if not file.endswith(".npy"):
                    continue

                clip_path = os.path.join(clip_dir, file)
                audio_path = os.path.join(audio_dir, file)

                if os.path.exists(audio_path):

                    self.samples.append(
                        (clip_path, audio_path, label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        clip_path, audio_path, label = self.samples[idx]

        # LOAD FEATURES
        video_feat = np.load(clip_path)[::8]
        audio_feat = np.load(audio_path)[::4]

        # HARD TEMPORAL CAP
        video_feat = video_feat[:self.max_len]
        audio_feat = audio_feat[:self.max_len]

        # TO TENSOR
        video_feat = torch.tensor(
            video_feat,
            dtype=torch.float32
        )

        audio_feat = torch.tensor(
            audio_feat,
            dtype=torch.float32
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return video_feat, audio_feat, label


# =========================================================
# COLLATE FUNCTION
# =========================================================
def collate_fn(batch):

    videos, audios, labels = zip(*batch)

    max_v = max(v.shape[0] for v in videos)
    max_a = max(a.shape[0] for a in audios)

    padded_videos = []
    padded_audios = []

    for v in videos:

        pad = max_v - v.shape[0]

        if pad > 0:
            v = F.pad(v, (0, 0, 0, pad))

        padded_videos.append(v)

    for a in audios:

        pad = max_a - a.shape[0]

        if pad > 0:
            a = F.pad(a, (0, 0, 0, pad))

        padded_audios.append(a)

    return (
        torch.stack(padded_videos),
        torch.stack(padded_audios),
        torch.tensor(labels)
    )


# =========================================================
# ALIGNMENT
# =========================================================
def align_audio_to_video(a, target_len):

    return F.interpolate(
        a.permute(0, 2, 1),
        size=target_len,
        mode='linear',
        align_corners=False
    ).permute(0, 2, 1)


# =========================================================
# MIL LOSS + HARD NEGATIVE MINING
# =========================================================
def mil_loss(
    logits,
    labels,
    criterion,
    k_ratio=0.1,
    hard_ratio=0.5
):

    B, T, C = logits.shape

    probs = torch.softmax(logits, dim=-1)

    anomaly_scores = probs[:, :, 1:].mean(dim=-1)

    k = max(1, int(T * k_ratio))

    topk_scores, _ = torch.topk(
        anomaly_scores,
        k,
        dim=1
    )

    video_scores = topk_scores.mean(dim=1)

    video_logits = logits.mean(dim=1)

    cls_loss = criterion(video_logits, labels)

    normal_idx = (labels == 0).nonzero(as_tuple=True)[0]
    abnormal_idx = (labels != 0).nonzero(as_tuple=True)[0]

    rank_loss = 0.0

    if len(normal_idx) > 0 and len(abnormal_idx) > 0:

        normal_scores = video_scores[normal_idx]
        abnormal_scores = video_scores[abnormal_idx]

        num_hard = max(
            1,
            int(len(normal_scores) * hard_ratio)
        )

        hard_normals, _ = torch.topk(
            normal_scores,
            num_hard
        )

        diff = (
            abnormal_scores.unsqueeze(1)
            - hard_normals.unsqueeze(0)
        )

        rank_loss = torch.mean(
            torch.clamp(1 - diff, min=0)
        )

    return cls_loss + 0.5 * rank_loss


# =========================================================
# CONTRASTIVE LOSS
# =========================================================
def contrastive_loss(
    v,
    a,
    temperature=0.07
):

    v = v.mean(dim=1)
    a = a.mean(dim=1)

    v = F.normalize(v, dim=-1)
    a = F.normalize(a, dim=-1)

    logits = torch.matmul(v, a.T) / temperature

    labels = torch.arange(
        v.size(0)
    ).to(v.device)

    loss_v = F.cross_entropy(logits, labels)
    loss_a = F.cross_entropy(logits.T, labels)

    return (loss_v + loss_a) / 2


# =========================================================
# PSEUDO LABELS
# =========================================================
def generate_pseudo_labels(
    logits,
    labels,
    threshold=0.6
):

    probs = torch.softmax(logits, dim=-1)

    anomaly_scores = probs[:, :, 1:].mean(dim=-1)

    pseudo = torch.zeros_like(anomaly_scores)

    for i in range(len(labels)):

        if labels[i] != 0:

            mask = anomaly_scores[i] > threshold

            pseudo[i][mask] = 1

    return pseudo


# =========================================================
# FRAME LOSS
# =========================================================
def frame_loss(logits, pseudo_labels):

    probs = torch.softmax(logits, dim=-1)

    anomaly_scores = probs[:, :, 1:].mean(dim=-1)

    return F.binary_cross_entropy(
        anomaly_scores,
        pseudo_labels
    )


# =========================================================
# ATTENTION VISUALIZATION
# =========================================================
def plot_attention(attn_map, title="Attention"):

    attn = attn_map[0].detach().cpu().numpy()

    # Handle 1D attention
    if len(attn.shape) == 1:
        attn = attn[np.newaxis, :]

    # Handle 3D attention
    elif len(attn.shape) == 3:
        attn = attn.mean(axis=0)

    plt.figure(figsize=(6, 5))

    plt.imshow(attn, aspect='auto')

    plt.colorbar()

    plt.title(title)

    plt.xlabel("Audio Time")
    plt.ylabel("Video Time")

    os.makedirs("attention_maps", exist_ok=True)

    plt.savefig(
        os.path.join(
            "attention_maps",
            f"{title}.png"
        )
    )

    plt.close()


# =========================================================
# TRAIN
# =========================================================
def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    epoch
):

    model.train()

    total_loss = 0

    for v, a, y in tqdm(loader, desc="Train"):

        v = v.to(device)
        a = a.to(device)
        y = y.to(device)

        # ALIGN AUDIO
        a = align_audio_to_video(
            a,
            v.shape[1]
        )

        # MODEL
        logits, attn_v, attn_a = model(v, a)

        # MIL LOSS
        mil = mil_loss(
            logits,
            y,
            criterion
        )

        # CONTRASTIVE
        v_proj = model.video_fc(
            F.normalize(v, dim=-1)
        )

        a_proj = model.audio_fc(
            F.normalize(a, dim=-1)
        )

        contr = contrastive_loss(
            v_proj,
            a_proj
        )

        loss = mil + 0.05 * contr

        # PSEUDO LABELS
        if epoch > 5:

            pseudo = generate_pseudo_labels(
                logits.detach(),
                y
            )

            frame = frame_loss(
                logits,
                pseudo
            )

            loss += 0.2 * frame

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5
        )

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# =========================================================
# VALIDATE
# =========================================================
def validate(
    model,
    loader,
    epoch
):

    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():

        for i, (v, a, y) in enumerate(
            tqdm(loader, desc="Val")
        ):

            v = v.to(device)
            a = a.to(device)
            y = y.to(device)

            a = align_audio_to_video(
                a,
                v.shape[1]
            )

            logits, attn_v, attn_a = model(v, a)

            probs = torch.softmax(
                logits,
                dim=-1
            )

            anomaly_score = 1 - probs[:, :, 0]

            # TEMPORAL SMOOTHING
            anomaly_score = F.avg_pool1d(
                anomaly_score.unsqueeze(1),
                kernel_size=5,
                stride=1,
                padding=2
            ).squeeze(1)

            # TOP-K SCORING
            k = max(
                1,
                int(anomaly_score.shape[1] * 0.1)
            )

            topk, _ = torch.topk(
                anomaly_score,
                k,
                dim=1
            )

            score = topk.mean(dim=1)

            all_scores.extend(
                score.cpu().numpy()
            )

            all_labels.extend(
                (y != 0).cpu().numpy()
            )

            # SAVE ATTENTION MAPS
            if epoch % 5 == 0 and i == 0:

                plot_attention(
                    attn_v,
                    f"epoch_{epoch+1}_attention"
                )

    auc = roc_auc_score(
        all_labels,
        all_scores
    )

    mAP = average_precision_score(
        all_labels,
        all_scores
    )

    return auc, mAP


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    # DATASETS
    train_dataset = AVDataset(
        clip_root="data/clip_features",
        audio_root="data/vggish_features"
    )

    val_dataset = AVDataset(
        clip_root="data/clip_features",
        audio_root="data/vggish_features"
    )

    # DATALOADERS
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # DEBUG
    print("Dataset size:", len(train_dataset))

    v, a, y = train_dataset[0]

    print("Video shape:", v.shape)
    print("Audio shape:", a.shape)
    print("Label:", y)

    # MODEL
    model = AVFusionTransformer().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    weights = torch.tensor(
        [0.2, 1.0, 1.0]
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

    # TRAIN LOOP
    EPOCHS = 20

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}")

        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            epoch
        )

        auc, mAP = validate(
            model,
            val_loader,
            epoch
        )

        print(f"Loss: {loss:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"mAP: {mAP:.4f}")

        torch.save(
            model.state_dict(),
            "last.pth"
        )

        print("Model saved: last.pth")
