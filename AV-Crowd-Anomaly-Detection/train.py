import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt

from fusionmodel import AVFusionModel


# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AVFusionModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

weights = torch.tensor([0.2, 1.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)


# =========================
# ALIGNMENT
# =========================
def align_audio_to_video(a, target_len):
    return F.interpolate(
        a.permute(0, 2, 1),
        size=target_len,
        mode='linear',
        align_corners=False
    ).permute(0, 2, 1)


# =========================
#  MIL + HARD NEGATIVE
# =========================
def mil_loss(logits, labels, k_ratio=0.1, hard_ratio=0.5):

    B, T, C = logits.shape
    probs = torch.softmax(logits, dim=-1)

    anomaly_scores = probs[:, :, 1:].mean(dim=-1)

    k = max(1, int(T * k_ratio))
    topk_scores, _ = torch.topk(anomaly_scores, k, dim=1)
    video_scores = topk_scores.mean(dim=1)

    video_logits = logits.mean(dim=1)
    cls_loss = criterion(video_logits, labels)

    normal_idx = (labels == 0).nonzero(as_tuple=True)[0]
    abnormal_idx = (labels != 0).nonzero(as_tuple=True)[0]

    rank_loss = 0.0

    if len(normal_idx) > 0 and len(abnormal_idx) > 0:

        normal_scores = video_scores[normal_idx]
        abnormal_scores = video_scores[abnormal_idx]

        #  Hard negatives
        num_hard = max(1, int(len(normal_scores) * hard_ratio))
        hard_normals, _ = torch.topk(normal_scores, num_hard)

        diff = abnormal_scores.unsqueeze(1) - hard_normals.unsqueeze(0)
        rank_loss = torch.mean(torch.clamp(1 - diff, min=0))

    return cls_loss + 0.5 * rank_loss


# =========================
#  CONTRASTIVE LOSS
# =========================
def contrastive_loss(v, a, temperature=0.07):

    v = v.mean(dim=1)
    a = a.mean(dim=1)

    v = F.normalize(v, dim=-1)
    a = F.normalize(a, dim=-1)

    logits = torch.matmul(v, a.T) / temperature
    labels = torch.arange(v.size(0)).to(v.device)

    loss_v = F.cross_entropy(logits, labels)
    loss_a = F.cross_entropy(logits.T, labels)

    return (loss_v + loss_a) / 2


# =========================
#  PSEUDO LABELS
# =========================
def generate_pseudo_labels(logits, labels, threshold=0.6):

    probs = torch.softmax(logits, dim=-1)
    anomaly_scores = probs[:, :, 1:].mean(dim=-1)

    pseudo = torch.zeros_like(anomaly_scores)

    for i in range(len(labels)):
        if labels[i] != 0:
            mask = anomaly_scores[i] > threshold
            pseudo[i][mask] = 1

    return pseudo


def frame_loss(logits, pseudo_labels):

    probs = torch.softmax(logits, dim=-1)
    anomaly_scores = probs[:, :, 1:].mean(dim=-1)

    return F.binary_cross_entropy(anomaly_scores, pseudo_labels)


# =========================
#  ATTENTION VIS
# =========================
def plot_attention(attn_map, title="Attention"):

    attn = attn_map[0].mean(0).detach().cpu().numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(attn, aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Audio Time")
    plt.ylabel("Video Time")
    plt.show()


# =========================
# TRAIN
# =========================
def train_one_epoch(loader, epoch):
    model.train()
    total_loss = 0

    for v, a, y in tqdm(loader, desc="Train"):

        v, a, y = v.to(device), a.to(device), y.to(device)

        #  ALIGNMENT
        a = align_audio_to_video(a, v.shape[1])

        logits, attn_v, attn_a = model(v, a)

        mil = mil_loss(logits, y)
        contr = contrastive_loss(v, a)

        loss = mil + 0.05 * contr

        #  PSEUDO LABELS AFTER WARMUP
        if epoch > 5:
            pseudo = generate_pseudo_labels(logits.detach(), y)
            frame = frame_loss(logits, pseudo)
            loss += 0.2 * frame

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# =========================
# VALIDATE
# =========================
def validate(loader, epoch):

    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for v, a, y in tqdm(loader, desc="Val"):

            v, a = v.to(device), a.to(device)

            a = align_audio_to_video(a, v.shape[1])

            logits, attn_v, attn_a = model(v, a)

            probs = torch.softmax(logits, dim=-1)
            anomaly_score = 1 - probs[:, :, 0]

            # smoothing
            anomaly_score = F.avg_pool1d(
                anomaly_score.unsqueeze(1),
                kernel_size=5,
                stride=1,
                padding=2
            ).squeeze(1)

            #  top-k scoring
            k = max(1, int(anomaly_score.shape[1] * 0.1))
            topk, _ = torch.topk(anomaly_score, k, dim=1)
            score = topk.mean(dim=1)

            all_scores.extend(score.cpu().numpy())
            all_labels.extend((y != 0).numpy())

            #  visualize occasionally
            if epoch % 5 == 0:
                plot_attention(attn_v, "Video attends Audio")

    auc = roc_auc_score(all_labels, all_scores)
    mAP = average_precision_score(all_labels, all_scores)

    return auc, mAP


# =========================
# LOOP
# =========================
EPOCHS = 20

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}")

    loss = train_one_epoch(train_loader, epoch)
    auc, mAP = validate(val_loader, epoch)

    print(f"Loss: {loss:.4f}")
    print(f"AUC: {auc:.4f}, mAP: {mAP:.4f}")

    torch.save(model.state_dict(), "last.pth")
