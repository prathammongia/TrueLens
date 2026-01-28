import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset_loader_fft import TrueLensFFTDataset
from model_fusion import TrueLensFusion

# ---------------- CONFIG ----------------
DATASET_DIR = "/content/drive/MyDrive/truelens"   # change only if path differs
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- DATA ----------------
dataset = TrueLensFFTDataset(DATASET_DIR)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Total images: {len(dataset)}")
print(f"Train: {train_size}, Val: {val_size}")

# ---------------- MODEL ----------------
model = TrueLensFusion().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------------- TRAINING LOOP ----------------
for epoch in range(EPOCHS):
    # -------- TRAIN --------
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for img, fft, labels in train_loader:
        img = img.to(DEVICE)
        fft = fft.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(img, fft)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = 100 * train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)

    # -------- VALIDATION --------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for img, fft, labels in val_loader:
            img = img.to(DEVICE)
            fft = fft.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(img, fft)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {avg_train_loss:.4f} "
        f"Train Acc: {train_acc:.2f}% "
        f"Val Acc: {val_acc:.2f}%"
    )

# ---------------- SAVE MODEL ----------------
torch.save(model.state_dict(), "truelens_fusion.pth")
print("✅ Improved model saved as truelens_fusion.pth")
