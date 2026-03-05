import librosa
import numpy as np
import os

audio_dir = "data/audio"
feature_dir = "data/features"

os.makedirs(feature_dir, exist_ok=True)

for audio in os.listdir(audio_dir):

    audio_path = os.path.join(audio_dir, audio)

    y, sr = librosa.load(audio_path, sr=16000)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)

    mel = mel.T

    np.save(os.path.join(feature_dir, audio.split(".")[0] + "_audio.npy"), mel)

print("Audio features extracted")
