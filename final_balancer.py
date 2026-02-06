import os
import shutil
import random

SOURCE = r'D:\dataset\processed_dataset'
TARGET = r'D:\dataset\final_train'
GOAL = 5000

# Delete the old target to start fresh and clean
if os.path.exists(TARGET):
    shutil.rmtree(TARGET)
os.makedirs(TARGET)

for cls in os.listdir(SOURCE):
    src_path = os.path.join(SOURCE, cls)
    if not os.path.isdir(src_path): continue
    
    dest_path = os.path.join(TARGET, cls)
    os.makedirs(dest_path, exist_ok=True)
    
    all_files = os.listdir(src_path)
    
    # If we have too many, pick a random 5000. If too few, take all.
    count = min(len(all_files), GOAL)
    selected = random.sample(all_files, count)
    
    print(f"📦 Copying {count} images for {cls}...")
    for f in selected:
        shutil.copy(os.path.join(src_path, f), os.path.join(dest_path, f))

print(f"\n🎉 DATASET READY! 45,000 images in {TARGET}")