import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from fusionmodel import AVFusionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AVFusionModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# 🔥 Weighted CE
weights = torch.tensor([0.2, 1.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)


# =====================================
# 🔥 MIL LOSS
# =====================================
def mil_loss(logits, labels, k_ratio=0.1):

    B, T, C = logits.shape

    probs = torch.softmax(logits, dim=-1)

    # 🔥 anomaly score (exclude normal)
    anomaly_scores = probs[:, :, 1:].mean(dim=-1)  # (B, T)

    k = max(1, int(T * k_ratio))

    topk_scores, _ = torch.topk(anomaly_scores, k, dim=1)
    video_scores = topk_scores.mean(dim=1)

    # 🔹 classification loss (video-level)
    video_logits = logits.mean(dim=1)
    cls_loss = criterion(video_logits, labels)

    # 🔹 ranking loss
    normal = labels == 0
    abnormal = labels != 0

    if abnormal.sum() > 0 and normal.sum() > 0:
        normal_score = video_scores[normal].mean()
        abnormal_score = video_scores[abnormal].mean()

        rank_loss = torch.clamp(1 - abnormal_score + normal_score, min=0)
    else:
        rank_loss = 0.0

    return cls_loss + 0.5 * rank_loss


# =====================================
# TRAIN
# =====================================
def train_one_epoch(loader):
    model.train()
    total_loss = 0

    for v, a, y in tqdm(loader, desc="Train"):

        v, a, y = v.to(device), a.to(device), y.to(device)

        logits = model(v, a)

        loss = mil_loss(logits, y)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# =====================================
# VALIDATE
# =====================================
def validate(loader):

    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for v, a, y in tqdm(loader, desc="Val"):

            v, a = v.to(device), a.to(device)

            logits = model(v, a)
            probs = torch.softmax(logits, dim=-1)

            anomaly_score = 1 - probs[:, :, 0]  # (B, T)

            # 🔥 temporal smoothing
            score = anomaly_score.mean(dim=1)

            all_scores.extend(score.cpu().numpy())
            all_labels.extend((y != 0).numpy())

    auc = roc_auc_score(all_labels, all_scores)
    mAP = average_precision_score(all_labels, all_scores)

    return auc, mAP


# =====================================
# LOOP
# =====================================
for epoch in range(20):

    print(f"\nEpoch {epoch+1}")

    loss = train_one_epoch(train_loader)
    auc, mAP = validate(val_loader)

    print(f"Loss: {loss:.4f}")
    print(f"AUC: {auc:.4f}, mAP: {mAP:.4f}")

    torch.save(model.state_dict(), "last.pth")
