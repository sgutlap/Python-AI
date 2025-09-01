import cv2
import os
import google.generativeai as genai

# Configure Google GenAI
genai.configure(api_key="AIzaSyDKnu87M0x5IO7YiIug0p6wl0jWXizBPVc")
model = genai.GenerativeModel("gemma-3-27b-it")

def extract_keyframes(video_path, max_frames=5):
    """
    Extracts key frames from a video file.
    Returns a list of saved image paths.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // max_frames, 1)
    frame_paths = []

    os.makedirs("frames", exist_ok=True)
    count = 0
    frame_idx = 0

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frame_path = f"frames/frame_{count}.jpg"
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            count += 1
        frame_idx += 1
    cap.release()
    return frame_paths


def analyze_video(video_path, prompt="Summarize this video content"):
    """
    Analyzes a video and returns an AI-generated summary.
    """
    print("[INFO] Extracting frames...")
    frames = extract_keyframes(video_path)

    print("[INFO] Sending frames to AI model...")
    images = [{"mime_type": "image/jpeg", "data": open(img, "rb").read()} for img in frames]

    request = [{"role": "user", "parts": [{"text": prompt}] + images}]
    response = model.generate_content(request)

    return response.text


if __name__ == "__main__":
    test_video = "Basketball.mp4"  # Replace with a sample video
    print(analyze_video(test_video))
