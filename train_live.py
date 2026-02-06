import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import time

# --- CONFIG ---
DATA_DIR = r'D:\dataset\final_train'
BATCH_SIZE = 64
EPOCHS = 10 
DEVICE = torch.device("cpu")

# --- DATA LOAD ---
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- MODEL ---
class SSAR_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, 9)
        )
    def forward(self, x): return self.main(x)

model = SSAR_Model().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- LIVE TRAINING LOOP ---
print(f"🚀 Training starting on {len(dataset)} images...")
print("⚠️ Monitor your laptop heat. Press Ctrl+C to stop safely after any Epoch.")

try:
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 50 == 0:
                print(f"Epoch {epoch+1} | Progress: [{i}/{len(train_loader)}] | Loss: {loss.item():.4f}")
        
        # SAVE PROGRESS AFTER EVERY EPOCH
        checkpoint_name = f"golden_brain_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_name)
        
        duration = (time.time() - start_time) / 60
        print(f"✅ EPOCH {epoch+1} FINISHED in {duration:.1f} mins.")
        print(f"💾 Saved progress as: {checkpoint_name}\n")

except KeyboardInterrupt:
    print("\n🛑 Training stopped by user. Your latest checkpoint is safe.")

print("🏁 Session ended.")