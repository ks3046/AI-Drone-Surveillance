import os
import subprocess

video_dir = "data/videos"
audio_dir = "data/audio"

os.makedirs(audio_dir, exist_ok=True)

for video in os.listdir(video_dir):

    video_path = os.path.join(video_dir, video)

    name = video.split(".")[0]

    audio_path = os.path.join(audio_dir, name + ".wav")

    command = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"

    subprocess.call(command, shell=True)

print("Audio extraction complete")
