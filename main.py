import cv2
import numpy as np

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
    """
    Decompose homography matrix into rotation and translation components.
    
    Args:
        homography: 3x3 homography matrix
        K: 3x3 camera intrinsic matrix (if None, an estimate will be used)
        
    Returns:
        rotations: List of possible rotation matrices
        translations: List of possible translation vectors
        normals: List of possible plane normals
    """
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

# Example usage:
# rotations, translations, normals = decompose_homography(homography_matrix)
# 
# # Select the most likely solution (usually the first one for simple cases)
# R = rotations[0]
# t = translations[0]
# 
# # Convert rotation matrix to Euler angles
# euler_angles = rotationMatrixToEulerAngles(R)
# print(f"Camera rotation (degrees): {np.degrees(euler_angles)}")
# print(f"Camera translation: {t}")

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

if __name__ == '__main__':
    main()
