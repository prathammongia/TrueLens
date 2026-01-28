from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import models, transforms
from PIL import Image
import io

from model_fusion import TrueLensFusion
from fft_utils import fft_transform

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["Real", "Fake"]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# Load models ONCE
baseline = models.resnet18()
baseline.fc = torch.nn.Linear(baseline.fc.in_features, 2)
baseline.load_state_dict(torch.load("truelens_baseline.pth", map_location=DEVICE))
baseline.eval().to(DEVICE)

fusion = TrueLensFusion()
fusion.load_state_dict(torch.load("truelens_fusion.pth", map_location=DEVICE))
fusion.eval().to(DEVICE)

@app.post("/predict")
async def predict(file: UploadFile, model_type: str = Form(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if model_type == "baseline":
            output = baseline(img_tensor)
        else:
            fft = fft_transform(img_tensor[0]).unsqueeze(0).to(DEVICE)
            output = fusion(img_tensor, fft)

        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1).item()

    return JSONResponse({
        "prediction": classes[pred],
        "confidence": float(prob[0][pred])
    })
