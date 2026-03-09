import torch
import numpy as np
import os
from tqdm import tqdm
import torchvggish
import librosa

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained VGGish model
vggish = torchvggish.vggish()
vggish.eval()
vggish.to(device)

audio_dir = "data/audio"
feature_dir = "data/features"

os.makedirs(feature_dir, exist_ok=True)

for audio in tqdm(os.listdir(audio_dir)):

    audio_path = os.path.join(audio_dir, audio)

    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)

    # Convert to tensor
    y = torch.tensor(y).float().to(device)

    # Extract embeddings
    with torch.no_grad():
        embedding = vggish(y)

    embedding = embedding.cpu().numpy()

    np.save(
        os.path.join(feature_dir, audio.split(".")[0] + "_audio.npy"),
        embedding
    )

print("VGGish audio features extracted")
