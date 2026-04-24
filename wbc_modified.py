# #This module uses a modified VGG16 CNN to classify white blood cell types
# removed 5th block of conv layer
# added batch normalization after Conv layer
import os
import numpy as np
from pathlib import Path
print("hit6")
import random
print("hit9")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
print("hit0")


def save_checkpoint(path, model, optimizer, epoch, history, config):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': config
    }, path)
# =========================================================
# 1. Settings
# =========================================================
# CHANGE THIS to your dataset folder on Desktop
#dataset_dir = Path.home() / "Desktop" / "your_dataset_folder"
dataset_dir = Path.cwd() / "FullDataset"



# Training settings
batch_size = 32
num_epochs = 10
learning_rate = 1e-4
image_size = 224
val_split = 0.2
random_seed = 42

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("hit1")

# =========================================================
# 2. Transforms
# =========================================================
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# 3. Load dataset
# =========================================================
if not dataset_dir.exists():
    raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")


full_dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

if len(full_dataset) == 0:
    raise ValueError(f"No images found in dataset folder: {dataset_dir}")

class_names = full_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)
print("Total images:", len(full_dataset))

# Split into train and validation
val_size = int(len(full_dataset) * val_split)
train_size = len(full_dataset) - val_size

generator = torch.Generator().manual_seed(random_seed)
train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=generator
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Train images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")


print("hit2")

# =========================================================
# 4. Modified VGG16 Model
# =========================================================
class mod_VGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = mod_VGG16(num_classes=num_classes).to(device)
print(model)


# =========================================================
# 5. Loss and optimizer
# =========================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

config = {
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "num_epochs": num_epochs,
    "image_size": image_size,
    "val_split": val_split
}

def save_checkpoint(path, model, optimizer, epoch, history, config):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': config
    }, path)

# =========================================================
# 6. Training function
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# =========================================================
# 7. Validation function
# =========================================================
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# =========================================================
# 8. Training loop
# =========================================================
best_val_acc = 0.0

print("hit3")
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "fulldataset_vgg16_mod.pth")
        print("Saved best model to fulldataset_vgg16_mod.pth")

        save_checkpoint(
            "best_checkpoint_mod.pth",
            model,
            optimizer,
            epoch,
            history,
            config
        )


# =========================================================
# 9. Save final model
# =========================================================
save_checkpoint(
    "final_checkpoint_mod.pth",
    model,
    optimizer,
    num_epochs,
    history,
    config
)
print("Training complete.")
print("Saved final model to final_checkpoint_mod.pth")
