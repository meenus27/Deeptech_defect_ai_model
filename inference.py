import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import random

# --- CONFIG ---
MODEL_PATH = "golden_brain_pro.pth"
BASE_DATASET = r'D:\dataset\processed_dataset' # Point to the main folder
CLASSES = ['Clean', 'Corrosion', 'LER', 'MalformedVia', 'Opens', 'Other', 'Particles', 'Scratches', 'Shorts']

class GoldenBrainCNN(nn.Module):
    def __init__(self):
        super(GoldenBrainCNN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 9) 
        )
    def forward(self, x): return self.main(x)

device = torch.device("cpu")
model = GoldenBrainCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(1), transforms.Resize((128, 128)),
    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
])

print(f"{'REAL FOLDER':<15} | {'AI PREDICTION':<15} | {'CONFIDENCE'}")
print("-" * 50)

# Pick one random image from every folder
for actual_class in CLASSES:
    folder_path = os.path.join(BASE_DATASET, actual_class)
    if not os.path.exists(folder_path): continue
    
    all_images = os.listdir(folder_path)
    if not all_images: continue
    
    random_img_name = random.choice(all_images)
    img_path = os.path.join(folder_path, random_img_name)
    
    img = Image.open(img_path)
    img_t = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_t)
        prob = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted = torch.max(prob, 0)
        pred_class = CLASSES[predicted.item()]
        
    print(f"{actual_class:<15} | {pred_class:<15} | {confidence.item()*100:.2f}%")