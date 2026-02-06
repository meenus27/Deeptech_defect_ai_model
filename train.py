import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# --- 1. CONFIGURATION ---
DATA_DIR = r'D:\dataset\final_train'
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Training on: {DEVICE}")

# --- 2. DATA PREPROCESSING ---
# We convert images to tensors and normalize them for the neural network
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # Keep it grayscale for NXP
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset and split into Train (80%) and Validation (20%)
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. THE ARCHITECTURE (The Brain) ---
class GoldenBrainCNN(nn.Module):
    def __init__(self):
        super(GoldenBrainCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), # 128 -> 64
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 32 -> 16
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 8) # 8 Classes: Clean, Shorts, etc.
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = GoldenBrainCNN().to(DEVICE)

# --- 4. LOSS AND OPTIMIZER ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. TRAINING LOOP ---
print("🧠 Starting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f}")

# --- 6. SAVE THE MODEL ---
torch.save(model.state_dict(), "golden_brain.pth")
print("✅ Training Complete! Model saved as 'golden_brain.pth'")