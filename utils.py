import cv2
import matplotlib.pyplot as plt

def readData(video_path, step = None):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    frames = []
    while True:
        if step is not None:
            for _ in range(step):  # Skip frames according to frame_step
                ret, frame = cap.read()
                if not ret:
                    break
        else:
            ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()
    cv2.destroyAllWindows()

    print(f"Read {len(frames)} frames from video")

    return frames

def visualize_panorama(panorama, title="Video Panorama"):
    """
    Visualize the panorama.
    
    Args:
        panorama: The panoramic image
        title: Title for the plot
    """
    plt.figure(figsize=(20, 10))
    if len(panorama.shape) == 3:
        plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(panorama, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def save_panorama(panorama, output_path):
    """
    Save the panorama to a file.
    
    Args:
        panorama: The panoramic image
        output_path: Path to save the panorama
    """
    cv2.imwrite(output_path, panorama)
    print(f"Panorama saved to: {output_path}")