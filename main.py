import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils import *

def main():
    video_path = './dataset/real_001.mp4'
    frames = readData(video_path)

    stitchy = cv2.Stitcher.create()
    medium = []

    batches = 3
    for i in tqdm(range(len(frames)//batches)):
        (dummy, output) = stitchy.stitch(frames[i:i+batches])
        if dummy != cv2.STITCHER_OK:
            continue
        else:
            medium.append(output)

    (dummy, output) = stitchy.stitch(medium)

    if dummy != cv2.STITCHER_OK:
        print("FAIL")
    else:
        visualize_panorama(output)

    save_panorama(output, output_path="./outputs/horse2.jpg")

if __name__ == '__main__':
    main()
