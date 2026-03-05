import cv2
import os
from tqdm import tqdm

video_dir = "data/videos"
frame_dir = "data/frames"

os.makedirs(frame_dir, exist_ok=True)

for video in os.listdir(video_dir):

    video_path = os.path.join(video_dir, video)
    cap = cv2.VideoCapture(video_path)

    name = video.split(".")[0]
    save_dir = os.path.join(frame_dir, name)

    os.makedirs(save_dir, exist_ok=True)

    frame_id = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(save_dir, f"{frame_id}.jpg")

        cv2.imwrite(frame_path, frame)

        frame_id += 1

    cap.release()

print("Frame extraction complete")