import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# --- 1. CONFIGURATION ---
MODEL_PATH = "D:\dataset\raw_data\golden_brain.pth"
# These must be in the EXACT same order as your training folders
CLASSES = ['Clean', 'Shorts', 'Opens', 'Scratches', 'Particles', 'LER', 'MalformedVia', 'Other']
DEVICE = torch.device("cpu") # Since we're on CPU

# --- 2. THE BRAIN STRUCTURE (Must match your Training script) ---
class GoldenBrainCNN(nn.Module):
    def __init__(self):
        super(GoldenBrainCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, 8) 
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# --- 3. LOAD THE TRAINED KNOWLEDGE ---
model = GoldenBrainCNN().to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Set to evaluation mode
    print("🧠 Model loaded successfully!")
else:
    print("❌ Error: 'golden_brain.pth' not found. Run training first!")

# --- 4. THE TEST FUNCTION ---
def predict_image(img_path):
    # Prepare the image just like we did during training
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    img = Image.open(img_path)
    img_tensor = transform(img).unsqueeze(0).to(DEVICE) # Add batch dimension
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    result = CLASSES[predicted_idx.item()]
    print(f"\n🔍 Image: {os.path.basename(img_path)}")
    print(f"✅ Prediction: {result} ({confidence.item()*100:.2f}% Confidence)")

# --- 5. RUN A TEST ---
# Point this to any image you want to test!
test_image = r'D:\dataset\final_train\Shorts\short_0_0.png'
if os.path.exists(test_image):
    predict_image(test_image)