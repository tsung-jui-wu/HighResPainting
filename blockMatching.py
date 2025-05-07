import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import readData

# --------- Calculating Tranformation ------------------------

def phase_correlation(frame1, frame2, threshold=5.0, use_window = False):
    # Convert to grayscale if needed
    if len(frame1.shape) == 3:
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    h, w = frame1.shape
    
    # Add windowing
    if use_window:
        window = np.outer(np.hanning(h), np.hanning(w))
        frame1 = frame1 * window
        frame2 = frame2 * window
    
    # Compute FFT
    fft1 = np.fft.fft2(frame1)
    fft2 = np.fft.fft2(frame2)
    
    # Compute cross-power spectrum
    cross_power = fft1 * np.conj(fft2)
    cross_power = cross_power / (np.abs(cross_power) + 1e-10)  # Normalize
    
    # Inverse FFT to get correlation
    correlation = np.abs(np.fft.ifft2(cross_power))
    
    # Find peak in correlation
    y_peak, x_peak = np.unravel_index(np.argmax(correlation), correlation.shape)
    
    # Adjust for circular shift
    if y_peak > h // 2:
        y_peak = y_peak - h
    if x_peak > w // 2:
        x_peak = x_peak - w

    if abs(x_peak) < threshold and abs(y_peak) < threshold:
        x_peak, y_peak = 0, 0
    
    return x_peak, y_peak

def hierarchical_phase_correlation(frame1, frame2, levels=3, use_window=False):
    # Convert to grayscale
    if len(frame1.shape) == 3:
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        frame1_gray = frame1
        frame2_gray = frame2
    
    # Build image pyramids
    pyramid1 = [frame1_gray]
    pyramid2 = [frame2_gray]
    
    for _ in range(levels - 1):
        pyramid1.append(cv2.pyrDown(pyramid1[-1]))
        pyramid2.append(cv2.pyrDown(pyramid2[-1]))
    
    # Start with identity transformation
    dx_total, dy_total = 0, 0
    
    # Process from coarsest to finest level
    for level in range(levels - 1, -1, -1):
        # Get frames at current level
        current_frame1 = pyramid1[level]
        current_frame2 = pyramid2[level]
        
        # Apply current accumulated shift to warp frame1
        h, w = current_frame1.shape
        tx_matrix = np.array([[1.0, 0.0, dx_total], [0.0, 1.0, dy_total]], dtype=np.float32)
        warped_frame1 = cv2.warpAffine(current_frame1, tx_matrix, (w, h))
        
        # Find remaining shift between warped_frame1 and frame2
        dx, dy = phase_correlation(warped_frame1, current_frame2, use_window=use_window)
        
        # Accumulate shift
        dx_total += dx
        dy_total += dy
        
        # Scale accumulated shift for next finer level
        if level > 0:
            dx_total *= 2
            dy_total *= 2
    
    # Create final affine transformation matrix (translation only)
    A = np.array([[1.0, 0.0, dx_total], [0.0, 1.0, dy_total]], dtype=np.float32)
    
    return A

def camera_motion_estimation(frame1, frame2, levels=3, use_window=False):
    return hierarchical_phase_correlation(frame1, frame2, levels, use_window=use_window)


# ------ Utils for printing to Plane ------------------

def expand_canvas(panorama, left=0, right=0, top=0, bottom=0):
    """
    Expand the canvas in the specified directions.
    
    Args:
        panorama: Current panorama image
        left, right, top, bottom: Number of pixels to add in each direction
        
    Returns:
        Expanded panorama
    """
    h, w = panorama.shape[:2]
    new_h = h + top + bottom
    new_w = w + left + right
    
    # Create new canvas
    if len(panorama.shape) == 3:
        new_panorama = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    else:
        new_panorama = np.zeros((new_h, new_w), dtype=np.uint8)
    
    # Copy existing content to new position
    new_panorama[top:top+h, left:left+w] = panorama
    
    return new_panorama


# --------------- Stitching Images ------------------------

def CreateDynamicPanorama(video_path, step=10, levels=3, alpha=0.5, initial_border=100, expand_size=200, use_window=False):
    """
    Create a panorama from a video using a dynamically expanded canvas.
    
    Args:
        video_path: Path to the video file
        step: Interval between frames to use
        levels: Levels of image pyramid
        alpha: Blending rate, (1 - alpha) * current + alpha * new_frame [0 - 1]
        initial_border: Initial border around the first frame
        expand_size: Amount to expand the canvas when needed
        
    Returns:
        Panorama image
    """
    # Read frames from the video
    frames = readData(video_path, step=step)
    
    if not frames:
        print("Error: No frames extracted from video")
        return None
    
    # Initialize with the first frame
    first_frame = frames[0]
    h, w = first_frame.shape[:2]
    
    # Create initial canvas with border
    if len(first_frame.shape) == 3:
        panorama = np.zeros((h + 2*initial_border, w + 2*initial_border, 3), dtype=np.uint8)
    else:
        panorama = np.zeros((h + 2*initial_border, w + 2*initial_border), dtype=np.uint8)
    
    # Start position for the first frame (with border)
    start_x = initial_border
    start_y = initial_border
    
    # Place the first frame
    panorama[start_y:start_y+h, start_x:start_x+w] = first_frame
    
    # Keep track of all frame positions
    frame_positions = [(start_y, start_x)]
    
    # Keep track of the canvas boundaries
    min_x, max_x = start_x, start_x + w - 1
    min_y, max_y = start_y, start_y + h - 1

    motion_history = []
    # Progressively add frames
    for i in tqdm(range(1, len(frames))):
        # Get new frame
        new_frame = frames[i]
        
        # Get previous frame position
        prev_y, prev_x = frame_positions[-1]
        
        # Crop the previous frame from the panorama
        prev_frame = panorama[prev_y:prev_y+h, prev_x:prev_x+w].copy()
        
        # Estimate motion between new frame and previous frame
        transform_matrix = camera_motion_estimation(new_frame, prev_frame, levels=levels, use_window=use_window)
        dx, dy = transform_matrix[0, 2], transform_matrix[1, 2]
        
        # Apply temporal filtering for smoother motion
        # if len(motion_history) >= motion_history_size:
        #     # Calculate median motion
        #     median_dx = np.median([m[0] for m in motion_history])
        #     median_dy = np.median([m[1] for m in motion_history])
            
        #     # If current motion is very different from median, consider it as noise
        #     if abs(dx - median_dx) > motion_threshold or abs(dy - median_dy) > motion_threshold:
        #         dx = median_dx
        #         dy = median_dy
        #         transform_matrix[0, 2] = dx
        #         transform_matrix[1, 2] = dy
            
        #     # Remove oldest motion
        #     motion_history.pop(0)
        
        # # Add current motion to history
        # motion_history.append((dx, dy))

        # Calculate new position (as integers)
        new_x = int(prev_x - dx)
        new_y = int(prev_y - dy)
        
        # Check if we need to expand the canvas
        expand_left = 0
        expand_right = 0
        expand_top = 0
        expand_bottom = 0
        
        if new_x < 0:
            expand_left = max(expand_size, -new_x)
            new_x += expand_left
            for j in range(len(frame_positions)):
                frame_positions[j] = (frame_positions[j][0], frame_positions[j][1] + expand_left)
            min_x += expand_left
            max_x += expand_left
        
        if new_y < 0:
            expand_top = max(expand_size, -new_y)
            new_y += expand_top
            for j in range(len(frame_positions)):
                frame_positions[j] = (frame_positions[j][0] + expand_top, frame_positions[j][1])
            min_y += expand_top
            max_y += expand_top
        
        if new_x + w >= panorama.shape[1]:
            expand_right = max(expand_size, new_x + w - panorama.shape[1] + 1)
        
        if new_y + h >= panorama.shape[0]:
            expand_bottom = max(expand_size, new_y + h - panorama.shape[0] + 1)
        
        # Expand the canvas if needed
        if expand_left > 0 or expand_right > 0 or expand_top > 0 or expand_bottom > 0:
            panorama = expand_canvas(panorama, expand_left, expand_right, expand_top, expand_bottom)
            print(f"Canvas expanded to {panorama.shape[0]}x{panorama.shape[1]}")
        
        # Update content boundaries
        min_x = min(min_x, new_x)
        max_x = max(max_x, new_x + w - 1)
        min_y = min(min_y, new_y)
        max_y = max(max_y, new_y + h - 1)
        
        # Apply the new frame to the panorama
        # Create a mask for non-black pixels
        if len(new_frame.shape) == 3:
            # For color images, a pixel is non-black if any channel is non-zero
            mask = (new_frame[:,:,0] > 0) | (new_frame[:,:,1] > 0) | (new_frame[:,:,2] > 0)
        else:
            # For grayscale, a pixel is non-black if it's non-zero
            mask = new_frame > 0
        
        # Apply the frame with alpha blending in overlap regions
        if len(new_frame.shape) == 3:
            for c in range(3):
                # Current values in panorama
                current = panorama[new_y:new_y+h, new_x:new_x+w, c].astype(np.float32)
                # New values from frame
                new_values = new_frame[:, :, c].astype(np.float32)
                
                # Create mask for where both images have content
                panorama_content = panorama[new_y:new_y+h, new_x:new_x+w, c] > 0
                overlap = panorama_content & mask
                
                # Blend in overlap regions
                blended = current.copy()
                blended[overlap] = (1 - alpha) * current[overlap] + alpha * new_values[overlap]
                
                # Copy new frame content where panorama is empty
                non_overlap = ~panorama_content & mask
                blended[non_overlap] = new_values[non_overlap]
                
                # Update panorama
                panorama[new_y:new_y+h, new_x:new_x+w, c] = blended.astype(np.uint8)
        else:
            # Grayscale case
            current = panorama[new_y:new_y+h, new_x:new_x+w].astype(np.float32)
            new_values = new_frame.astype(np.float32)
            
            # Create masks
            panorama_content = panorama[new_y:new_y+h, new_x:new_x+w] > 0
            overlap = panorama_content & mask
            non_overlap = ~panorama_content & mask
            
            # Blend in overlap regions
            blended = current.copy()
            blended[overlap] = (1 - alpha) * current[overlap] + alpha * new_values[overlap]
            blended[non_overlap] = new_values[non_overlap]
            
            # Update panorama
            panorama[new_y:new_y+h, new_x:new_x+w] = blended.astype(np.uint8)
        
        # Store this frame's position
        frame_positions.append((new_y, new_x))
    
    # Crop the final panorama to the content area with a small border
    border = 10
    min_y = max(0, min_y - border)
    max_y = min(panorama.shape[0] - 1, max_y + border)
    min_x = max(0, min_x - border)
    max_x = min(panorama.shape[1] - 1, max_x + border)
    
    cropped_panorama = panorama[min_y:max_y+1, min_x:max_x+1]
    
    return cropped_panorama

if __name__ == "__main__":
    filename = "refer_007"
    video_path = f"./dataset/{filename}.mp4"
    step = 5
    levels = 1
    alpha = 0.5
    use_window = False

    # Create panorama from video
    panorama = CreateDynamicPanorama(video_path, step=step, levels=levels, alpha=alpha, use_window=use_window)
    
    if panorama is not None:
        # Save the result
        cv2.imwrite(f"{filename}_fft.jpg", panorama)
        
        # Display the panorama
        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
        plt.title("Video Panorama")
        plt.axis('off')
        plt.tight_layout()
        plt.show()