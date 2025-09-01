# Video_Analysis.py
import cv2

def analyze_video(video_path: str):
    """
    Analyze a video frame by frame. Currently, it only prints frame count,
    resolution, and duration. You can expand it with AI analysis later.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Frame count: {frame_count}")
    print(f"Duration: {duration:.2f} seconds")

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Example: display every 60th frame
        if frame_number % 60 == 0:
            cv2.imshow("Video Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    analyze_video("example.mp4")
