import torch
import numpy as np
import os
from fusion_model import AVFusionModel

feature_dir = "data/features"

model = AVFusionModel()

optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(10):

    for file in os.listdir(feature_dir):

        if "_video" not in file:
            continue

        name = file.replace("_video.npy","")

        video = np.load(os.path.join(feature_dir,name+"_video.npy"))

        audio = np.load(os.path.join(feature_dir,name+"_audio.npy"))

        video = torch.tensor(video).float().unsqueeze(0)
        audio = torch.tensor(audio).float().unsqueeze(0)

        label = torch.ones(1,video.shape[1],1)

        pred = model(video,audio)

        loss = criterion(pred,label)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print("epoch",epoch,"loss",loss.item())

torch.save(model.state_dict(),"models/av_model.pth")
