import cv2
import os
import time

RTSP_URL = 'rtsp://192.168.72.181:8080/h264_pcm.sdp'
SAVE_DIR = 'captured_frames'
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("[ERROR] Could not open RTSP stream.")
    exit()

frame_id = 0
max_frames = 100  # Set to None for infinite
frame_skip = 5    # Adjust to save fewer or more

print("[+] Starting frame capture...")

while True:
    ret, frame = cap.read()

    if not ret or frame is None or frame.size == 0:
        print("[!] Skipped corrupt/broken frame.")
        time.sleep(0.1)
        continue

    if frame_id % frame_skip == 0:
        filename = os.path.join(SAVE_DIR, f"frame_{frame_id:04d}.jpg")
        success = cv2.imwrite(filename, frame)
        if success:
            print(f"[✓] Saved: {filename}")
        else:
            print(f"[✗] Failed to save: {filename}")

    frame_id += 1

    if max_frames and frame_id >= max_frames:
        break

cap.release()
print("[+] Done.")
