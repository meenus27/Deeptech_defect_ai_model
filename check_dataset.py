import os

# --- PATHS ---
PROCESSED_PATH = r'D:\dataset\processed_dataset'
FINAL_PATH = r'D:\dataset\final_train'

def audit_directory(path, title):
    if not os.path.exists(path):
        print(f"\n❌ {title} folder not found at: {path}")
        return

    print(f"\n📊 --- {title} AUDIT ---")
    print(f"{'Class Name':<20} | {'File Count':<10}")
    print("-" * 35)
    
    total = 0
    # Sorting ensures 'Clean' to 'Shorts' are in order
    for folder in sorted(os.listdir(path)):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg'))])
            print(f"{folder:<20} | {count:<10}")
            total += count
            
    print("-" * 35)
    print(f"{'TOTAL IMAGES':<20} | {total:<10}")

# Run for both folders
audit_directory(PROCESSED_PATH, "SOURCE (Processed)")
audit_directory(FINAL_PATH, "TARGET (Final Train)")