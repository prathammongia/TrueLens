# ===============================
# Point‑0: Multimodal Classifier
# HOG + LBP + CNN (VGG16) + LR
# ===============================

import os
import cv2
import numpy as np
from PIL import Image

from skimage.feature import hog, local_binary_pattern
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
from torchvision import models, transforms

# ---------------- CONFIG ----------------
DATASET_DIR = "/content/drive/MyDrive/truelens"  # must contain real/ and fake/
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- DATA LOADER ----------------
def load_image_paths():
    samples = []

    real_dir = os.path.join(DATASET_DIR, "real")
    fake_dir = os.path.join(DATASET_DIR, "fake")

    for fname in os.listdir(real_dir):
        samples.append((os.path.join(real_dir, fname), 0))  # Real = 0

    for fname in os.listdir(fake_dir):
        samples.append((os.path.join(fake_dir, fname), 1))  # Fake = 1

    return samples

# ---------------- FEATURE EXTRACTORS ----------------
def extract_hog(img_gray):
    return hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

def extract_lbp(img_gray):
    lbp = local_binary_pattern(img_gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=59, range=(0, 59))
    return hist / (hist.sum() + 1e-6)

# ---------------- CNN FEATURE (VGG16) ----------------
cnn = models.vgg16(pretrained=True)
cnn.classifier = cnn.classifier[:-1]  # remove final FC
cnn.eval().to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_cnn(img_pil):
    img = transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = cnn(img)
    return feat.cpu().numpy().flatten()

# ---------------- FULL FEATURE PIPELINE ----------------
def extract_features(img_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_pil = Image.open(img_path).convert("RGB")

    hog_f = extract_hog(img_gray)
    lbp_f = extract_lbp(img_gray)
    cnn_f = extract_cnn(img_pil)

    return np.concatenate([hog_f, lbp_f, cnn_f])

# ---------------- MAIN ----------------
print("🔰 Point‑0 Multimodal Training Started")

samples = load_image_paths()
print(f"Total images found: {len(samples)}")

X, y = [], []

for i, (img_path, label) in enumerate(samples):
    if i % 50 == 0:
        print(f"Processed {i}/{len(samples)} images")

    features = extract_features(img_path)
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

# ---------------- CLASSIFIER ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)

print("✅ Point‑0 Accuracy:", acc)
