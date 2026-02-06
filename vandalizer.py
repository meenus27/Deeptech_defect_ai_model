import cv2
import os
import numpy as np
import random

# --- CONFIG ---
SOURCE_ROOT = r'D:\dataset\raw_data'
DEST_ROOT = r'D:\dataset\processed_dataset'
IMG_SIZE = 128
VARIANTS = 5 

CLASSES = ['Clean', 'Shorts', 'Opens', 'Scratches', 'Particles', 'LER', 'MalformedVia', 'Other', 'Corrosion']

# Ensure destination folders exist
for cls in CLASSES:
    os.makedirs(os.path.join(DEST_ROOT, cls), exist_ok=True)

def process_image(img_path, output_id):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return False
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # 1. Clean
    cv2.imwrite(os.path.join(DEST_ROOT, 'Clean', f"chip_{output_id}.png"), img)
    
    for i in range(VARIANTS):
        # Defect logic (Hyper-Visible)
        # Shorts
        s_img = img.copy()
        cv2.line(s_img, (20, 64), (100, 64), (255), 6) 
        cv2.imwrite(os.path.join(DEST_ROOT, 'Shorts', f"c_{output_id}_{i}.png"), s_img)

        # Opens
        o_img = img.copy()
        cv2.rectangle(o_img, (60, 10), (75, 110), (0), -1) 
        cv2.imwrite(os.path.join(DEST_ROOT, 'Opens', f"c_{output_id}_{i}.png"), o_img)

        # Scratches
        sc_img = img.copy()
        cv2.line(sc_img, (10, 10), (118, 118), (240), 2)
        cv2.imwrite(os.path.join(DEST_ROOT, 'Scratches', f"c_{output_id}_{i}.png"), sc_img)

        # Particles
        p_img = img.copy()
        for _ in range(8):
            cv2.circle(p_img, (random.randint(10,110), random.randint(10,110)), 4, (15), -1)
        cv2.imwrite(os.path.join(DEST_ROOT, 'Particles', f"c_{output_id}_{i}.png"), p_img)

        # LER
        noise = np.random.randint(0, 50, img.shape).astype(np.uint8)
        ler_img = cv2.add(img, noise)
        cv2.imwrite(os.path.join(DEST_ROOT, 'LER', f"c_{output_id}_{i}.png"), ler_img)

        # MalformedVia
        v_img = img.copy()
        cv2.circle(v_img, (64, 64), 15, (40), 5) 
        cv2.imwrite(os.path.join(DEST_ROOT, 'MalformedVia', f"c_{output_id}_{i}.png"), v_img)

        # Corrosion
        corr_img = img.copy()
        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        cv2.circle(mask, (random.randint(40,80), random.randint(40,80)), 40, (180), -1)
        mask = cv2.GaussianBlur(mask, (41, 41), 0)
        corr_img = cv2.subtract(corr_img, mask) 
        cv2.imwrite(os.path.join(DEST_ROOT, 'Corrosion', f"c_{output_id}_{i}.png"), corr_img)

        # Other
        ot_img = cv2.rotate(img, cv2.ROTATE_180)
        cv2.imwrite(os.path.join(DEST_ROOT, 'Other', f"c_{output_id}_{i}.png"), ot_img)
        
    return True

# --- NEW DEEP SCAN LOGIC ---
print(f"🚀 Scanning for images in: {SOURCE_ROOT}...")
success_count = 0

for root, dirs, files in os.walk(SOURCE_ROOT):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            full_path = os.path.join(root, file)
            # Use a unique ID based on the count to avoid filename collisions
            if process_image(full_path, success_count):
                success_count += 1
                if success_count % 100 == 0:
                    print(f"📦 Processed {success_count} images...")

print(f"\n✅ SUCCESS!")
print(f"Processed {success_count} raw images found in the directory tree.")
print(f"Created {success_count * 9} synthetic images in {DEST_ROOT}")