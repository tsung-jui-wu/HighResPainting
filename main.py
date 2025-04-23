import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import *

def motion_comp(prev_frame, curr_frame, num_points=500, points_to_use=500, transform_type='homography'):

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)

    # get features for first frame
    corners = cv2.goodFeaturesToTrack(prev_gray, num_points, qualityLevel=0.01, minDistance=10)

    # get matching features in next frame with Sparse Optical Flow Estimation
    matched_corners, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners, None)

    # reformat previous and current corner points
    prev_points = corners[status==1]
    curr_points = matched_corners[status==1]

    # sub sample number of points so we don't overfit
    if points_to_use > prev_points.shape[0]:
        points_to_use = prev_points.shape[0]

    index = np.random.choice(prev_points.shape[0], size=points_to_use, replace=False)
    prev_points_used = prev_points[index]
    curr_points_used = curr_points[index]

    # find transformation matrix from frame 1 to frame 2
    if transform_type == 'affine':
        A, _ = cv2.estimateAffine2D(prev_points_used, curr_points_used, method=cv2.RANSAC)
    elif transform_type == 'homography':
        A, _ = cv2.findHomography(prev_points_used, curr_points_used)

    return A, prev_points, curr_points

def decompose_homography(homography, K=None):

    # If camera matrix is not provided, use a reasonable estimate
    if K is None:
        # Assume a reasonable camera matrix for a standard camera
        # Format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        # Where fx,fy are focal lengths and cx,cy is the principal point
        K = np.array([
            [1000, 0, 500],  # Estimated values, adjust based on your camera
            [0, 1000, 500],
            [0, 0, 1]
        ], dtype=np.float32)
    
    # OpenCV's decomposeHomographyMat returns multiple solutions
    num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(
        homography, K
    )
    
    return rotations, translations, normals

def visualize_transformation(prev_frame, curr_frame, transformation_matrix, is_homography=True):
    """
    Top Left = Prev Frame
    Top Right = Curr Frame
    Bottom Left = Transformed Prev Frame
    Bottom Right = Difference between two frames
    """
    # Get image dimensions
    h, w = prev_frame.shape[:2]
    
    # Apply transformation to the previous frame
    if is_homography:
        # For homography (3x3 matrix)
        warped_frame = cv2.warpPerspective(prev_frame, transformation_matrix, (w, h))
    else:
        # For affine transformation (2x3 matrix)
        warped_frame = cv2.warpAffine(prev_frame, transformation_matrix, (w, h))
    
    # Calculate difference between warped previous frame and current frame
    # This shows areas where the transformation doesn't align perfectly
    difference = cv2.absdiff(warped_frame, curr_frame)
    
    # Convert to grayscale if the images are color
    if len(difference.shape) == 3:
        difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    else:
        difference_gray = difference
    
    # Calculate metrics to quantify the quality of alignment
    mean_error = np.mean(difference_gray)
    max_error = np.max(difference_gray)
    std_error = np.std(difference_gray)
    
    # Create a more visual representation of the difference
    # Apply a colormap to make differences more visible
    difference_color = cv2.applyColorMap(difference_gray, cv2.COLORMAP_JET)
    
    # Create a composite view for comparison
    # Stack the images horizontally for comparison
    composite1 = np.hstack((prev_frame, curr_frame))
    composite2 = np.hstack((warped_frame, difference_color))
    composite = np.vstack((composite1, composite2))
    
    # Display the results
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    plt.title(f'Transformation Validation\nMean Error: {mean_error:.2f}, Max Error: {max_error}, StdDev: {std_error:.2f}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"Alignment Metrics:")
    print(f"Mean Error: {mean_error:.2f}")
    print(f"Max Error: {max_error}")
    print(f"Standard Deviation: {std_error:.2f}")
    
    return mean_error, max_error, std_error, warped_frame, difference_color

def main():
    video_path = './dataset/real_001.mp4'
    frames = readData(video_path)
    
    transformMatrix, prevPoints, currPoints = motion_comp(frames[0], frames[1])
    print("Transform Matrix:", transformMatrix)
    rotations, translations, norms = decompose_homography(transformMatrix)
    r = rotations[0]
    t = translations[0]
    print("rotations:", r)
    print("translations:", t)

    mean_error, max_error, std_error, warped, diff = visualize_transformation(
        frames[0], frames[1], transformMatrix, is_homography=True
    )


if __name__ == '__main__':
    main()
