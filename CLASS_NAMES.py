import os
BASE_PATH = r'D:\dataset\final_train'
folders = sorted([f for f in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, f))])
print("🚀 YOUR TRUE CLASS ORDER IS:")
for i, folder in enumerate(folders):
    print(f"{i}: {folder}")