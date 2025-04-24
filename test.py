import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *

def read_video_frames(video_path, frame_step=1, max_frames=None):
    """
    Read frames from a video file.
    
    Args:
        video_path: Path to the video file
        frame_step: Use every nth frame (higher values for faster processing)
        max_frames: Maximum number of frames to read (None for all frames)
    
    Returns:
        frames: List of video frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    frame_count = 0
    
    while True:
        for _ in range(frame_step):  # Skip frames according to frame_step
            ret, frame = cap.read()
            if not ret:
                break
        
        if not ret:
            break
        
        frames.append(frame)
        frame_count += 1
        
        if max_frames is not None and frame_count >= max_frames:
            break
    
    cap.release()
    
    if len(frames) < 2:
        raise ValueError(f"Not enough frames in video: {video_path}")
    
    print(f"Read {len(frames)} frames from video")
    return frames

def calculate_camera_motion_homography(prev_frame, curr_frame):
    """
    Calculate the homography transformation between two frames.
    
    Args:
        prev_frame: Previous frame image
        curr_frame: Current frame image
        
    Returns:
        homography_matrix: 3x3 homography transformation matrix
    """
    # Convert images to grayscale
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame
        
    if len(curr_frame.shape) == 3:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:
        curr_gray = curr_frame
    
    # Detect feature points in previous frame
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray, 
        maxCorners=400,  
        qualityLevel=0.01,
        minDistance=15,
        blockSize=3
    )
    
    if prev_pts is None or len(prev_pts) < 4:
        print("Warning: Not enough features detected in previous frame")
        return np.eye(3, dtype=np.float32)
    
    # Calculate optical flow
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    
    # Filter only valid points
    idx = np.where(status == 1)[0]
    
    if len(idx) < 4:
        print("Warning: Not enough matching features between frames")
        return np.eye(3, dtype=np.float32)
    
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]
    
    # Find homography matrix
    homography_matrix, mask = cv2.findHomography(
        prev_pts, 
        curr_pts, 
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    )
    
    # If homography could not be found, return identity matrix
    if homography_matrix is None:
        print("Warning: Could not estimate homography matrix")
        homography_matrix = np.eye(3, dtype=np.float32)
    
    return homography_matrix

def calculate_camera_motion_affine(prev_frame, curr_frame):
    """
    Calculate the affine transformation between two frames.
    
    Args:
        prev_frame: Previous frame image
        curr_frame: Current frame image
        
    Returns:
        affine_matrix: 2x3 affine transformation matrix
    """
    # Convert images to grayscale
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame
        
    if len(curr_frame.shape) == 3:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:
        curr_gray = curr_frame
    
    # Detect feature points
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray, 
        maxCorners=400, 
        qualityLevel=0.01,
        minDistance=15, 
        blockSize=3
    )
    
    if prev_pts is None or len(prev_pts) < 3:
        print("Warning: Not enough features detected in previous frame")
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    
    # Calculate optical flow
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    
    # Filter only valid points
    idx = np.where(status == 1)[0]
    
    if len(idx) < 3:
        print("Warning: Not enough matching features between frames")
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]
    
    # Find affine transformation
    affine_matrix, inliers = cv2.estimateAffine2D(
        prev_pts, 
        curr_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    )
    
    # If transformation could not be found, return identity matrix
    if affine_matrix is None:
        print("Warning: Could not estimate affine matrix")
        affine_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    
    return affine_matrix

def calculate_transformations(frames, method='homography', verbose=True):
    """
    Calculate transformations between consecutive frames.
    
    Args:
        frames: List of video frames
        method: 'homography' or 'affine'
        verbose: Whether to print progress
    
    Returns:
        transformations: List of transformation matrices between consecutive frames
    """
    transformations = []
    iterator = tqdm(range(len(frames) - 1)) if verbose else range(len(frames) - 1)
    
    for i in iterator:
        if method == 'homography':
            H = calculate_camera_motion_homography(frames[i], frames[i+1])
        else:  # affine
            H = calculate_camera_motion_affine(frames[i], frames[i+1])
            # Convert 2x3 affine matrix to 3x3 homography format
            H_full = np.eye(3, dtype=np.float32)
            H_full[:2, :] = H
            H = H_full
        
        transformations.append(H)
    
    return transformations

def calculate_cumulative_transforms(transformations):
    """
    Calculate cumulative transformations with respect to the first frame.
    
    Args:
        transformations: List of transformation matrices between consecutive frames
    
    Returns:
        cumulative_transforms: List of cumulative transformation matrices
    """
    cumulative_transforms = [np.eye(3, dtype=np.float32)]  # Identity for the first frame
    
    current_transform = np.eye(3, dtype=np.float32)
    for H in transformations:
        # We need to calculate the inverse since we're transforming from 
        # current frame to the first frame (reverse direction)
        current_transform = H @ current_transform
        cumulative_transforms.append(current_transform)
    
    return cumulative_transforms

def exposure_compensate(frames, transformations):
    """
    Apply exposure compensation to make brightness consistent across frames.
    
    Args:
        frames: List of video frames
        transformations: List of transformation matrices between consecutive frames
    
    Returns:
        compensated_frames: List of exposure-compensated frames
    """
    compensated_frames = frames.copy()
    
    # Process each frame after the first one
    for i in range(1, len(frames)):
        prev_frame = frames[i-1]
        curr_frame = frames[i]
        
        # Warp previous frame to align with current frame
        h, w = prev_frame.shape[:2]
        aligned_prev = cv2.warpPerspective(prev_frame, transformations[i-1], (w, h))
        
        # Create masks for overlapping regions
        prev_mask = (aligned_prev.sum(axis=2) if len(aligned_prev.shape) == 3 else aligned_prev) > 0
        curr_mask = (curr_frame.sum(axis=2) if len(curr_frame.shape) == 3 else curr_frame) > 0
        overlap_mask = prev_mask & curr_mask
        
        # Skip if there's no overlap
        if not np.any(overlap_mask):
            continue
        
        # Calculate exposure ratio in overlapping region
        if len(curr_frame.shape) == 3:  # Color image
            # Calculate per-channel ratios
            ratios = []
            for c in range(3):
                prev_mean = np.mean(aligned_prev[overlap_mask, c])
                curr_mean = np.mean(curr_frame[overlap_mask, c])
                if prev_mean > 0:
                    ratios.append(curr_mean / prev_mean)
            
            # Use median ratio across channels
            exposure_ratio = np.median(ratios)
        else:  # Grayscale
            prev_mean = np.mean(aligned_prev[overlap_mask])
            curr_mean = np.mean(curr_frame[overlap_mask])
            exposure_ratio = curr_mean / prev_mean if prev_mean > 0 else 1.0
        
        # Compensate exposure
        compensated_frames[i] = np.clip(curr_frame * exposure_ratio, 0, 255).astype(np.uint8)
    
    return compensated_frames

def determine_panorama_size(frames, cumulative_transforms):
    """
    Determine the size of the panorama by finding the transformed corners of each frame.
    
    Args:
        frames: List of video frames
        cumulative_transforms: List of cumulative transformation matrices
    
    Returns:
        panorama_width: Width of the panorama
        panorama_height: Height of the panorama
        offset_x: X offset to bring all points to positive coordinates
        offset_y: Y offset to bring all points to positive coordinates
    """
    h, w = frames[0].shape[:2]
    corners = np.array([
        [0, 0, 1],          # Top-left
        [w, 0, 1],          # Top-right
        [0, h, 1],          # Bottom-left
        [w, h, 1]           # Bottom-right
    ]).T  # Transpose to get 3x4 matrix
    
    # Find the transformed corners for all frames
    all_corners = []
    for H in cumulative_transforms:
        # Transform corners using homography
        transformed_corners = H @ corners
        # Convert from homogeneous to Cartesian coordinates
        transformed_corners = transformed_corners / transformed_corners[2]
        all_corners.append(transformed_corners[:2].T)  # Keep only x,y coordinates
    
    all_corners = np.concatenate(all_corners)
    
    # Find the minimum and maximum x,y coordinates to determine panorama size
    min_x, min_y = np.min(all_corners, axis=0)
    max_x, max_y = np.max(all_corners, axis=0)
    
    # Calculate translation to bring all points to positive coordinates
    offset_x = max(0, -min_x)
    offset_y = max(0, -min_y)
    
    # Calculate panorama dimensions
    panorama_width = int(max_x + offset_x) + 1
    panorama_height = int(max_y + offset_y) + 1
    
    print(f"Panorama dimensions: {panorama_width}x{panorama_height}")
    
    return panorama_width, panorama_height, offset_x, offset_y

def create_distance_weight_map(h, w):
    """
    Create a weight map that's higher in the center and lower at the edges.
    
    Args:
        h: Height of the frame
        w: Width of the frame
    
    Returns:
        weight: Weight map
    """
    y, x = np.indices((h, w))
    center_y, center_x = h // 2, w // 2
    weight = 1.0 - np.sqrt(((y - center_y) / (h/2)) ** 2 + ((x - center_x) / (w/2)) ** 2)
    weight = np.clip(weight, 0, 1)
    return weight

def create_panorama(frames, cumulative_transforms, use_blending=True):
    """
    Create a panoramic image from video frames using their transformations.
    
    Args:
        frames: List of video frames
        cumulative_transforms: List of cumulative transformation matrices
        use_blending: Whether to use advanced blending
    
    Returns:
        panorama: The stitched panoramic image
    """
    # Determine panorama size
    panorama_width, panorama_height, offset_x, offset_y = determine_panorama_size(frames, cumulative_transforms)
    
    # Create translation matrix to shift all frames
    T = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Initialize panorama and weight map for blending
    if len(frames[0].shape) == 3:  # Color image
        panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.float32)
    else:  # Grayscale image
        panorama = np.zeros((panorama_height, panorama_width), dtype=np.float32)
    
    weight_map = np.zeros((panorama_height, panorama_width), dtype=np.float32)
    
    # Pre-calculate weight maps for each frame if using blending
    frame_weights = None
    if use_blending:
        frame_weights = [create_distance_weight_map(frame.shape[0], frame.shape[1]) for frame in frames]
    
    # Combine all frames into the panorama
    for i, frame in enumerate(tqdm(frames, desc="Creating panorama")):
        # Adjust the frame's transformation to include the offset
        H_adjusted = T @ cumulative_transforms[i]
        
        if use_blending and frame_weights is not None:
            # Warp the frame and its weight map
            warped_frame = cv2.warpPerspective(frame.astype(np.float32), H_adjusted, 
                                             (panorama_width, panorama_height))
            warped_weight = cv2.warpPerspective(frame_weights[i], H_adjusted, 
                                             (panorama_width, panorama_height))
            
            # Create a mask to identify non-black pixels
            if len(frame.shape) == 3:  # Color image
                mask = warped_frame.sum(axis=2) > 0
                mask_3d = np.dstack([mask, mask, mask])
                
                # Add the weighted frame to the panorama
                panorama += warped_frame * np.expand_dims(warped_weight, axis=2) * mask_3d
            else:  # Grayscale
                mask = warped_frame > 0
                panorama += warped_frame * warped_weight * mask
            
            # Update the weight map
            weight_map += warped_weight * mask
        else:
            # Simple overlay without blending
            warped_frame = cv2.warpPerspective(frame, H_adjusted, 
                                              (panorama_width, panorama_height))
            
            # Create a mask to identify non-black pixels in the warped frame
            if len(frames[0].shape) == 3:  # Color image
                mask = (warped_frame != 0).any(axis=2)
                mask = np.dstack([mask, mask, mask])
            else:  # Grayscale image
                mask = (warped_frame != 0)
            
            # Blend the warped frame with the existing panorama
            panorama = np.where(mask, warped_frame, panorama)
    
    # Normalize the panorama by the accumulated weights if using blending
    if use_blending:
        valid_pixels = weight_map > 0
        if len(frames[0].shape) == 3:  # Color image
            for c in range(3):
                normalized_channel = np.zeros_like(panorama[..., c])
                normalized_channel[valid_pixels] = panorama[..., c][valid_pixels] / weight_map[valid_pixels]
                panorama[..., c] = normalized_channel
        else:  # Grayscale
            normalized_panorama = np.zeros_like(panorama)
            normalized_panorama[valid_pixels] = panorama[valid_pixels] / weight_map[valid_pixels]
            panorama = normalized_panorama
    
    # Convert back to uint8
    panorama = np.clip(panorama, 0, 255).astype(np.uint8)
    
    return panorama

def build_video_panorama(video_path, frame_step=1, max_frames=None, method='homography', 
                        use_exposure_comp=True, use_blending=True):
    """
    Build a panorama from a video.
    
    Args:
        video_path: Path to the video file
        frame_step: Use every nth frame (higher values for faster processing)
        max_frames: Maximum number of frames to use
        method: 'homography' or 'affine'
        use_exposure_comp: Whether to apply exposure compensation
        use_blending: Whether to use advanced blending
    
    Returns:
        panorama: The stitched panoramic image
    """
    # 1. Read video frames
    frames = read_video_frames(video_path, frame_step, max_frames)
    
    # 2. Calculate transformations between consecutive frames
    transformations = calculate_transformations(frames, method)
    
    # 3. Calculate cumulative transformations
    cumulative_transforms = calculate_cumulative_transforms(transformations)
    
    # 4. Apply exposure compensation if requested
    if use_exposure_comp:
        frames = exposure_compensate(frames, transformations)
    
    # 5. Create the panorama
    panorama = create_panorama(frames, cumulative_transforms, use_blending)
    
    return panorama

# Example usage:
if __name__ == "__main__":
    # Set parameters
    VIDEO_PATH = "./dataset/real_001.mp4"  # Replace with your video path
    OUTPUT_PATH = "panorama_output.jpg"     # Output path for the panorama
    
    # Build the panorama
    panorama = build_video_panorama(
        video_path=VIDEO_PATH,
        frame_step=5,              # Use every 5th frame (adjust as needed)
        max_frames=None,             # Limit to 30 frames (adjust or set to None for all frames)
        method='homography',       # Use 'homography' for perspective changes, 'affine' for simpler motion
        use_exposure_comp=False,    # Apply exposure compensation
        use_blending=False          # Use advanced blending
    )
    
    # Visualize the result
    visualize_panorama(panorama)
    
    # Save the panorama
    # save_panorama(panorama, OUTPUT_PATH)