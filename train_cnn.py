import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ════════════════════════════════
#  CONFIG
# ════════════════════════════════
EPOCHS     = 15
BATCH_SIZE = 128
LR         = 0.001
SAVE_PATH  = "airwrite_cnn.pth"
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ════════════════════════════════
#  DOWNLOAD DATASET
# ════════════════════════════════
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("Downloading EMNIST Letters dataset...")
train_data = datasets.EMNIST(
    root="./emnist_data", split="letters",
    train=True, download=True, transform=transform
)
test_data = datasets.EMNIST(
    root="./emnist_data", split="letters",
    train=False, download=True, transform=transform
)

# EMNIST labels are 1-26, convert to 0-25
train_data.targets -= 1
test_data.targets  -= 1

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_data)}")
print(f"Test samples:  {len(test_data)}")

# ════════════════════════════════
#  CNN MODEL
# ════════════════════════════════
class AirWriteCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 26)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model = AirWriteCNN().to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ════════════════════════════════
#  TRAINING
# ════════════════════════════════
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

train_accs, val_accs = [], []
best_acc = 0.0

print("\nStarting training — this takes 15-25 mins on CPU...")
print("=" * 55)

for epoch in range(1, EPOCHS + 1):

    # Train
    model.train()
    correct = total = running_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct      += (outputs.argmax(1) == labels).sum().item()
        total        += labels.size(0)

    train_acc  = correct / total * 100
    train_loss = running_loss / len(train_loader)

    # Validate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total   += labels.size(0)

    val_acc = correct / total * 100
    scheduler.step(100 - val_acc)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch:02d}/{EPOCHS}  "
          f"Loss: {train_loss:.4f}  "
          f"Train: {train_acc:.1f}%  "
          f"Val: {val_acc:.1f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"           >>> Best model saved ({val_acc:.1f}%)")

print("=" * 55)
print(f"Done! Best accuracy: {best_acc:.1f}%")
print(f"Model saved as: {SAVE_PATH}")

# Plot
plt.figure(figsize=(8,4))
plt.plot(train_accs, label="Train")
plt.plot(val_accs,   label="Validation")
plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
plt.title("AirScript CNN Training")
plt.legend(); plt.tight_layout()
plt.savefig("training_history.png")
print("Chart saved as training_history.png")