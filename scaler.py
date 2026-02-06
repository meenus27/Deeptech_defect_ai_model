import os
import shutil
import random

# --- CONFIG ---
SOURCE_DIR = r'D:\dataset\processed_dataset'
FINAL_DIR = r'D:\dataset\final_train'
IMAGES_PER_CLASS = 2000 # Optimized for high-contrast synthetic data

# 1. Clean Slate: Remove old final_train if it exists to avoid mixing data
if os.path.exists(FINAL_DIR):
    print(f"🧹 Removing old {FINAL_DIR}...")
    shutil.rmtree(FINAL_DIR)

os.makedirs(FINAL_DIR, exist_ok=True)

# 2. Process Folders
for cls in os.listdir(SOURCE_DIR):
    src_path = os.path.join(SOURCE_DIR, cls)
    if not os.path.isdir(src_path): continue
    
    dest_path = os.path.join(FINAL_DIR, cls)
    os.makedirs(dest_path, exist_ok=True)
    
    all_files = os.listdir(src_path)
    
    # Check if we have enough images
    num_to_copy = min(len(all_files), IMAGES_PER_CLASS)
    selected_files = random.sample(all_files, num_to_copy)
    
    print(f"📦 Copying {num_to_copy} images for class: {cls}...")
    for f in selected_files:
        shutil.copy(os.path.join(src_path, f), os.path.join(dest_path, f))

print(f"\n✅ SUCCESS! Your balanced dataset is ready at: {FINAL_DIR}")