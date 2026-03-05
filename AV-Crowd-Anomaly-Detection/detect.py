import torch
import numpy as np
from fusion_model import AVFusionModel
import matplotlib.pyplot as plt

model = AVFusionModel()

model.load_state_dict(torch.load("models/av_model.pth"))

video = np.load("sample_video.npy")
audio = np.load("sample_audio.npy")

video = torch.tensor(video).float().unsqueeze(0)
audio = torch.tensor(audio).float().unsqueeze(0)

with torch.no_grad():

    scores = model(video,audio)

scores = scores.squeeze().numpy()
for i,s in enumerate(scores):

    if s>0.5:
        print("Anomaly detected at frame",i


plt.plot(scores)
plt.title("Anomaly Score Over Time")
plt.xlabel("Frame")
plt.ylabel("Score")
plt.show())
