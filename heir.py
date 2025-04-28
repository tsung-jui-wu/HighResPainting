import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *

def GetNewFrameSizeAndMatrix(HomographyMatrix, Sec_ImageShape, Base_ImageShape):
    # Reading the size of the image
    (Height, Width) = Sec_ImageShape
    
    # Taking the matrix of initial coordinates of the corners of the secondary image
    # Stored in the following format: [[x1, x2, x3, x4], [y1, y2, y3, y4], [1, 1, 1, 1]]
    # Where (xt, yt) is the coordinate of the i th corner of the image. 
    InitialMatrix = np.array([[0, Width - 1, Width - 1, 0],
                              [0, 0, Height - 1, Height - 1],
                              [1, 1, 1, 1]])
    
    # Finding the final coordinates of the corners of the image after transformation.
    # NOTE: Here, the coordinates of the corners of the frame may go out of the 
    # frame(negative values). We will correct this afterwards by updating the 
    # homography matrix accordingly.
    FinalMatrix = np.dot(HomographyMatrix, InitialMatrix)

    [x, y, c] = FinalMatrix
    x = np.divide(x, c)
    y = np.divide(y, c)

    # Finding the dimentions of the stitched image frame and the "Correction" factor
    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))

    New_Width = max_x
    New_Height = max_y
    Correction = [0, 0]
    if min_x < 0:
        New_Width -= min_x
        Correction[0] = abs(min_x)
    if min_y < 0:
        New_Height -= min_y
        Correction[1] = abs(min_y)
    
    # Again correcting New_Width and New_Height
    # Helpful when secondary image is overlaped on the left hand side of the Base image.
    if New_Width < Base_ImageShape[1] + Correction[0]:
        New_Width = Base_ImageShape[1] + Correction[0]
    if New_Height < Base_ImageShape[0] + Correction[1]:
        New_Height = Base_ImageShape[0] + Correction[1]

    # Finding the coordinates of the corners of the image if they all were within the frame.
    x = np.add(x, Correction[0])
    y = np.add(y, Correction[1])
    OldInitialPoints = np.float32([[0, 0],
                                   [Width - 1, 0],
                                   [Width - 1, Height - 1],
                                   [0, Height - 1]])
    NewFinalPonts = np.float32(np.array([x, y]).transpose())

    # Updating the homography matrix. Done so that now the secondary image completely 
    # lies inside the frame
    HomographyMatrix = cv2.getPerspectiveTransform(OldInitialPoints, NewFinalPonts)
    
    return [New_Height, New_Width], Correction, HomographyMatrix

def ProjectOntoPlane(InitialImage):
    global w, h, center
    h, w = InitialImage.shape[:2]
    center = [w // 2, h // 2]
    # No need for focal length f since we're not doing cylindrical projection
    
    # For a normal plane projection, we can simply return the original image
    # and the coordinate arrays for the full image
    
    # Creating arrays for all coordinates in the image
    ti_x = np.array([i for i in range(w) for j in range(h)])
    ti_y = np.array([j for i in range(w) for j in range(h)])
    
    return InitialImage, ti_x, ti_y

def FilterMatchesByMotionConsistency(Matches, BaseImage_kp, SecImage_kp, max_motion_deviation=5.0):
    """
    Filter matches based on consistency of motion vectors.
    
    Args:
        Matches: List of matches
        BaseImage_kp: Keypoints from base image
        SecImage_kp: Keypoints from secondary image
        max_motion_deviation: Maximum allowed deviation from median motion
        
    Returns:
        Filtered matches
    """
    if len(Matches) < 8:  # Need a minimum number of matches
        return Matches
    
    # Extract motion vectors from all matches
    motion_vectors = []
    for match in Matches:
        p1 = np.array(BaseImage_kp[match[0].queryIdx].pt)
        p2 = np.array(SecImage_kp[match[0].trainIdx].pt)
        motion = p1 - p2
        motion_vectors.append(motion)
    
    motion_vectors = np.array(motion_vectors)
    
    # Calculate median motion in x and y directions
    median_motion = np.median(motion_vectors, axis=0)
    
    # Calculate deviation from median for each match
    deviations = np.linalg.norm(motion_vectors - median_motion, axis=1)
    
    # Filter matches based on deviation
    consistent_matches = []
    for i, match in enumerate(Matches):
        if deviations[i] < max_motion_deviation:
            consistent_matches.append(match)
    
    print(f"Filtered matches: {len(Matches)} -> {len(consistent_matches)}")
    return consistent_matches

def FindMatches(BaseImage, SecImage, filter_static=True):
    # Using SIFT to find the keypoints and descriptors in the images
    Sift = cv2.SIFT_create()
    BaseImage_kp, BaseImage_des = Sift.detectAndCompute(cv2.cvtColor(BaseImage, cv2.COLOR_BGR2GRAY), None)
    SecImage_kp, SecImage_des = Sift.detectAndCompute(cv2.cvtColor(SecImage, cv2.COLOR_BGR2GRAY), None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    InitialMatches = flann.knnMatch(BaseImage_des, SecImage_des, k=2)

    # Applying ratio test and filtering out the good matches
    GoodMatches = []
    for m, n in InitialMatches:
        if m.distance < 0.75 * n.distance:
            GoodMatches.append([m])
    
    # For static scenes, filter matches by motion consistency
    if filter_static and len(GoodMatches) > 10:
        GoodMatches = FilterMatchesByMotionConsistency(GoodMatches, BaseImage_kp, SecImage_kp)
    
    return GoodMatches, BaseImage_kp, SecImage_kp

def FindHomography(Matches, BaseImage_kp, SecImage_kp):
    # If less than 4 matches found, exit the code.
    if len(Matches) < 4:
        print("\nNot enough matches found between the images.\n")
        exit(0)

    # Storing coordinates of points corresponding to the matches found in both the images
    BaseImage_pts = []
    SecImage_pts = []
    for Match in Matches:
        BaseImage_pts.append(BaseImage_kp[Match[0].queryIdx].pt)
        SecImage_pts.append(SecImage_kp[Match[0].trainIdx].pt)

    # Changing the datatype to "float32" for finding homography
    BaseImage_pts = np.float32(BaseImage_pts)
    SecImage_pts = np.float32(SecImage_pts)

    # For static scenes, assume a very rigid transformation (less degrees of freedom)
    # Use a more restrictive model like affine transformation for more stability
    if len(Matches) >= 10:  # If we have enough matches, try affine first
        (AffineMatrix, AffineStatus) = cv2.estimateAffine2D(SecImage_pts, BaseImage_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if AffineMatrix is not None and np.sum(AffineStatus) >= 6:  # At least 6 inliers
            # Convert affine to homography (add the [0,0,1] row)
            HomographyMatrix = np.vstack([AffineMatrix, np.array([0, 0, 1])])
            Status = AffineStatus
            print("Using affine transformation for static scene")
        else:
            # Fall back to homography if affine fails
            (HomographyMatrix, Status) = cv2.findHomography(SecImage_pts, BaseImage_pts, cv2.RANSAC, 3.0)
            print("Using homography transformation")
    else:
        # Not enough matches for affine, use regular homography
        (HomographyMatrix, Status) = cv2.findHomography(SecImage_pts, BaseImage_pts, cv2.RANSAC, 3.0)
        print("Using homography transformation (limited matches)")

    return HomographyMatrix, Status

def ComputeOpticalFlow(prev_frame, curr_frame):
    """
    Compute dense optical flow between two frames.
    
    Args:
        prev_frame: Previous frame
        curr_frame: Current frame
        
    Returns:
        Mask of static regions
    """
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Calculate magnitude of flow for each pixel
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create a mask of static regions (low magnitude of motion)
    static_threshold = 0.5  # Adjust based on your scene
    static_mask = magnitude < static_threshold
    
    # Convert to proper mask format
    static_mask = static_mask.astype(np.uint8) * 255
    
    return static_mask

def PreprocessForStaticScene(frames):
    """
    Preprocess frames to handle a mostly static scene with small movements.
    
    Args:
        frames: List of video frames
        
    Returns:
        Processed frames with small movements suppressed
    """
    print("Preprocessing frames to handle small movements...")
    processed_frames = []
    base_frame = frames[0].copy()
    processed_frames.append(base_frame)
    
    for i in tqdm(range(1, len(frames))):
        # Compute optical flow to identify static regions
        static_mask = ComputeOpticalFlow(base_frame, frames[i])
        
        # Dilate the mask to ensure we cover all moving areas
        kernel = np.ones((5, 5), np.uint8)
        moving_mask = cv2.dilate(255 - static_mask, kernel, iterations=2)
        
        # Create a blended frame that uses the base frame for moving regions
        blended_frame = frames[i].copy()
        moving_mask_3ch = cv2.merge([moving_mask, moving_mask, moving_mask]) / 255.0
        blended_frame = (blended_frame * (1 - moving_mask_3ch) + base_frame * moving_mask_3ch).astype(np.uint8)
        
        processed_frames.append(blended_frame)
    
    return processed_frames

def StitchImages(BaseImage, SecImage):
    # Applying Cylindrical projection on SecImage
    SecImage_Cyl, mask_x, mask_y = ProjectOntoPlane(SecImage)

    # Getting SecImage Mask
    SecImage_Mask = np.zeros(SecImage_Cyl.shape, dtype=np.uint8)
    SecImage_Mask[mask_y, mask_x, :] = 255

    # Finding matches between the 2 images and their keypoints
    Matches, BaseImage_kp, SecImage_kp = FindMatches(BaseImage, SecImage_Cyl, filter_static=True)
    
    # Finding homography matrix.
    HomographyMatrix, Status = FindHomography(Matches, BaseImage_kp, SecImage_kp)
    
    # Finding size of new frame of stitched images and updating the homography matrix 
    NewFrameSize, Correction, HomographyMatrix = GetNewFrameSizeAndMatrix(HomographyMatrix, SecImage_Cyl.shape[:2], BaseImage.shape[:2])

    # Finally placing the images upon one another.
    SecImage_Transformed = cv2.warpPerspective(SecImage_Cyl, HomographyMatrix, (NewFrameSize[1], NewFrameSize[0]))
    SecImage_Transformed_Mask = cv2.warpPerspective(SecImage_Mask, HomographyMatrix, (NewFrameSize[1], NewFrameSize[0]))
    BaseImage_Transformed = np.zeros((NewFrameSize[0], NewFrameSize[1], 3), dtype=np.uint8)
    BaseImage_Transformed[Correction[1]:Correction[1]+BaseImage.shape[0], Correction[0]:Correction[0]+BaseImage.shape[1]] = BaseImage

    # Create a mask for the base image
    BaseImage_Mask = np.zeros((NewFrameSize[0], NewFrameSize[1], 3), dtype=np.uint8)
    BaseImage_Mask[Correction[1]:Correction[1]+BaseImage.shape[0], Correction[0]:Correction[0]+BaseImage.shape[1]] = 255

    # Calculate the overlap region for blending
    overlap_region = cv2.bitwise_and(BaseImage_Mask, SecImage_Transformed_Mask)
    
    # Create feathered blending masks using distance transforms
    overlap_gray = cv2.cvtColor(overlap_region, cv2.COLOR_BGR2GRAY)
    
    # Make sure we have an overlap before attempting to blend
    if np.sum(overlap_gray) > 0:
        # Use weighted blending at the seams
        # This helps to hide any slight misalignments due to small movements
        
        # Create a 1-pixel border around the overlap region to serve as a blend boundary
        kernel = np.ones((5, 5), np.uint8)
        overlap_dilated = cv2.dilate(overlap_gray, kernel, iterations=3)
        
        # Create a seamless clone - this uses Poisson blending under the hood
        # Center point for seamless cloning - center of the overlap region
        moments = cv2.moments(overlap_dilated)
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            
            # Create binary masks for the two images
            sec_mask = cv2.cvtColor(SecImage_Transformed_Mask, cv2.COLOR_BGR2GRAY)
            
            # Set a region for blending
            blending_mask = sec_mask
            
            try:
                # Attempt seamless cloning
                StitchedImage = cv2.seamlessClone(
                    SecImage_Transformed, 
                    BaseImage_Transformed, 
                    blending_mask, 
                    (center_x, center_y), 
                    cv2.NORMAL_CLONE
                )
            except cv2.error:
                # Fallback if seamless cloning fails
                print("Seamless cloning failed, using binary masking instead")
                StitchedImage = cv2.bitwise_or(
                    SecImage_Transformed, 
                    cv2.bitwise_and(BaseImage_Transformed, cv2.bitwise_not(SecImage_Transformed_Mask))
                )
        else:
            # Fallback if moments calculation fails
            StitchedImage = cv2.bitwise_or(
                SecImage_Transformed, 
                cv2.bitwise_and(BaseImage_Transformed, cv2.bitwise_not(SecImage_Transformed_Mask))
            )
    else:
        # No overlap - use simple binary masking
        StitchedImage = cv2.bitwise_or(
            SecImage_Transformed, 
            cv2.bitwise_and(BaseImage_Transformed, cv2.bitwise_not(SecImage_Transformed_Mask))
        )

    return StitchedImage

def crop_black_borders(image):
    """
    Crop the black borders from the stitched image.
    
    Args:
        image: The stitched image
        
    Returns:
        Cropped image without black borders
    """
    # Convert to grayscale to find non-black regions
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to identify non-black regions
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours of the non-black region
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If contours exist, find the largest one (should be the content)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop the image to the bounding rectangle
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image
    else:
        # If no contours found (unlikely), return the original image
        return image

def main():
    video_path = "./dataset/real_001.mp4"
    frames = readData(video_path, early_stop=None)
    
    print(f"Total frames to process: {len(frames)}")
    
    # Preprocess frames to handle small movements
    keyframes = PreprocessForStaticScene(frames)
    
    print(f"Using {len(keyframes)} keyframes for stitching...")
    
    # Set up sequential stitching
    BaseImage, _, _ = ProjectOntoPlane(keyframes[0])
    
    # Loop through frames and stitch sequentially
    for i in tqdm(range(1, len(keyframes)), desc="Stitching frames"):
        StitchedImage = StitchImages(BaseImage, keyframes[i])
        BaseImage = StitchedImage.copy()
    
    # Optional: Try to close the loop by stitching with the first frame again
    # if your video makes a full 360-degree panorama
    # StitchedImage = StitchImages(BaseImage, keyframes[0])
    
    # Crop black borders from the final result
    CroppedImage = crop_black_borders(BaseImage)
    
    cv2.imwrite("outputs/horse12.png", CroppedImage)

if __name__ == '__main__':
    main()