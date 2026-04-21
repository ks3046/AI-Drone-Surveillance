import torch
import torch.nn as nn
import torch.nn.functional as F


class AVFusionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.dim = 256

        #  Projection
        self.video_fc = nn.Linear(512, self.dim)
        self.audio_fc = nn.Linear(128, self.dim)

        #  Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 500, self.dim))  # max 500 timesteps

        #  Cross Attention (bidirectional)
        self.cross_attn_v = nn.MultiheadAttention(self.dim, 4, batch_first=True)
        self.cross_attn_a = nn.MultiheadAttention(self.dim, 4, batch_first=True)

        #  Fusion
        self.fusion_fc = nn.Linear(self.dim * 2, self.dim)

        #  Transformer Encoder (replaces GRU)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.3,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        #  Dropout
        self.dropout = nn.Dropout(0.3)

        #  Frame-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    #  ALIGNMENT FUNCTION
    def align_audio_to_video(self, audio_feat, target_len):
        """
        Align audio features to video length using interpolation
        """
        return F.interpolate(
            audio_feat.permute(0, 2, 1),
            size=target_len,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)

    def forward(self, video_feat, audio_feat):

        #  Normalize
        video_feat = F.normalize(video_feat, dim=-1)
        audio_feat = F.normalize(audio_feat, dim=-1)

        #  Projection
        v = self.video_fc(video_feat)   # (B, Tv, D)
        a = self.audio_fc(audio_feat)   # (B, Ta, D)

        #  Proper temporal alignment
        a = self.align_audio_to_video(a, v.shape[1])

        #  Cross-modal attention
        v_attn, _ = self.cross_attn_v(v, a, a)
        a_attn, _ = self.cross_attn_a(a, v, v)

        #  Fusion
        x = torch.cat([v_attn, a_attn], dim=-1)
        x = self.fusion_fc(x)

        #  Positional encoding
        T = x.shape[1]
        x = x + self.pos_embed[:, :T, :]

        x = self.dropout(x)

        #  Transformer temporal modeling
        x = self.transformer(x)   # (B, T, D)

        #  Frame-level logits
        out = self.classifier(x)  # (B, T, 3)

        return out
