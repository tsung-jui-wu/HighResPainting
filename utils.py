import cv2

def readData(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Video Frame", frame)
        frames.append(frame)

    cap.release()
    cv2.destroyAllWindows()

    return frames