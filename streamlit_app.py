import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

from model_fusion import TrueLensFusion
from fft_utils import fft_transform

# ---------------- CONFIG ----------------
BASELINE_MODEL_PATH = "truelens_baseline.pth"
FUSION_MODEL_PATH = "truelens_fusion.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["Real", "Fake"]

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_baseline():
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_fusion():
    model = TrueLensFusion()
    model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

baseline_model = load_baseline()
fusion_model = load_fusion()

# ---------------- UI ----------------
st.title("🕵️ TrueLens – AI Image Detection")
st.write("Detect whether an image is **AI‑generated or Real**, even after editing.")

model_choice = st.selectbox(
    "Select model",
    ["Baseline (RGB)", "Improved (RGB + Frequency)"]
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Check Image"):
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            if model_choice == "Baseline (RGB)":
                output = baseline_model(img_tensor)
            else:
                fft_img = fft_transform(img_tensor[0]).unsqueeze(0).to(DEVICE)
                output = fusion_model(img_tensor, fft_img)

            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1).item()

        st.success(f"Prediction: **{classes[pred]}**")
        st.info(f"Confidence: **{prob[0][pred].item():.2f}**")
