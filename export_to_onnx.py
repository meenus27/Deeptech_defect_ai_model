import torch
import torch.nn as nn

# 1. ARCHITECTURE (Must be identical to your training script)
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

# 2. LOAD WEIGHTS
device = torch.device("cpu")
model = GoldenBrainCNN()
# Using weights_only=True to keep it clean and safe
model.load_state_dict(torch.load("golden_brain_pro.pth", map_location=device, weights_only=True))
model.eval() 

# 3. CREATE DUMMY INPUT (128x128 grayscale)
dummy_input = torch.randn(1, 1, 128, 128)

# 4. EXPORT TO ONNX
print("🚀 Freezing the Brain into ONNX format...")
torch.onnx.export(
    model, 
    dummy_input, 
    "SSAR_Final_Model.onnx", 
    export_params=True,      
    opset_version=11,        # Most stable version for NXP eIQ
    do_constant_folding=True,# Pre-calculates math for speed
    input_names=['input'], 
    output_names=['output']
)

import os
size_mb = os.path.getsize("SSAR_Final_Model.onnx") / (1024 * 1024)
print(f"✅ SUCCESS! 'SSAR_Final_Model.onnx' created.")
print(f"📦 Final Model Size: {size_mb:.2f} MB")