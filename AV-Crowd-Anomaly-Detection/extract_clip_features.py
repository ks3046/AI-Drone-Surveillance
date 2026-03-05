import torch
import clip
import os
import cv2
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/16", device=device)

frame_dir = "data/frames"
feature_dir = "data/features"

os.makedirs(feature_dir, exist_ok=True)

for folder in os.listdir(frame_dir):

    folder_path = os.path.join(frame_dir, folder)

    features = []

    for frame in sorted(os.listdir(folder_path)):

        frame_path = os.path.join(folder_path, frame)

        image = preprocess(cv2.imread(frame_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = model.encode_image(image)

        features.append(feature.cpu().numpy())

    features = np.concatenate(features)

    np.save(os.path.join(feature_dir, folder + "_video.npy"), features)

print("CLIP features extracted")
