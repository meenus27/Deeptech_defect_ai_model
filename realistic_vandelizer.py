import cv2
import os
import numpy as np
import random

# --- CONFIG ---
SOURCE_ROOT = r'D:\dataset\raw_data'
DEST_ROOT = r'D:\dataset\processed_dataset'
IMG_SIZE = 128
CLASSES = ['Clean', 'Shorts', 'Opens', 'Scratches'] # Starting with the big 4

for cls in CLASSES:
    os.makedirs(os.path.join(DEST_ROOT, cls), exist_ok=True)

def apply_realistic_short(img):
    overlay = img.copy()
    # Draw a jagged, blurred line to look like a metal bridge
    pts = np.array([[30,40], [45,42], [60,55]], np.int32)
    cv2.polylines(overlay, [pts], False, (180), 3)
    # Blend it so it's not a "hard" line
    blurry = cv2.GaussianBlur(overlay, (3, 3), 0)
    return cv2.addWeighted(img, 0.5, blurry, 0.5, 0)

def apply_realistic_scratch(img):
    overlay = img.copy()
    # Thin, high-intensity line
    cv2.line(overlay, (10, 10), (110, 110), (220), 1)
    # Add a bit of "motion blur" to the scratch
    kernel = np.zeros((3, 3))
    kernel[1, :] = 1.0/3.0
    return cv2.filter2D(overlay, -1, kernel)

# (Add more defect functions here...)

print("🚀 Manufacturing Realistic Semiconductor Defects...")
# Loop through your raw images and save them into the new folders...