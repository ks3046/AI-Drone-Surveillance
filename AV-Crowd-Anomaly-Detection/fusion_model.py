import torch
import torch.nn as nn
import torch.nn.functional as F


class AVFusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        dim = 256

        # 🔹 Projection
        self.video_fc = nn.Linear(512, dim)
        self.audio_fc = nn.Linear(128, dim)

        # 🔹 Cross Attention
        self.cross_attn_v = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.cross_attn_a = nn.MultiheadAttention(dim, 4, batch_first=True)

        # 🔹 Fusion
        self.fusion_fc = nn.Linear(dim * 2, dim)

        # 🔹 Temporal (GRU)
        self.temporal = nn.GRU(
            input_size=dim,
            hidden_size=128,
            batch_first=True
        )

        # 🔹 Dropout
        self.dropout = nn.Dropout(0.3)

        # 🔹 Frame-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )

    def forward(self, video_feat, audio_feat):

        # 🔥 Normalize
        video_feat = F.normalize(video_feat, dim=-1)
        audio_feat = F.normalize(audio_feat, dim=-1)

        # 🔹 Projection
        v = self.video_fc(video_feat)
        a = self.audio_fc(audio_feat)

        # 🔹 Align time
        min_len = min(v.shape[1], a.shape[1])
        v = v[:, :min_len, :]
        a = a[:, :min_len, :]

        # 🔥 Cross Attention
        v_attn, _ = self.cross_attn_v(v, a, a)
        a_attn, _ = self.cross_attn_a(a, v, v)

        # 🔹 Fusion
        x = torch.cat([v_attn, a_attn], dim=-1)
        x = self.fusion_fc(x)

        x = self.dropout(x)

        # 🔹 Temporal modeling
        x, _ = self.temporal(x)   # (B, T, 128)

        # 🔥 Frame-level logits
        out = self.classifier(x)  # (B, T, 3)

        return out
