import torch
import torch.nn as nn

class AVFusionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.video_fc = nn.Linear(512,256)

        self.audio_fc = nn.Linear(64,128)

        self.fusion_fc = nn.Linear(384,128)

        self.temporal = nn.LSTM(
            input_size=128,
            hidden_size=64,
            batch_first=True
        )

        self.classifier = nn.Linear(64,1)

    def forward(self, video_feat, audio_feat):

        v = self.video_fc(video_feat)

        a = self.audio_fc(audio_feat)

        x = torch.cat([v,a],dim=-1)

        x = self.fusion_fc(x)

        x,_ = self.temporal(x)

        out = self.classifier(x)

        return out
