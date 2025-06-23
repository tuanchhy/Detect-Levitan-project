from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

app = FastAPI()

# CORS: cho phép frontend (Live Server) gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo ResNet50 với 2 lớp output
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # output: 2 classes

# Load weights đã train
model.load_state_dict(torch.load('models/resnet50_best.pth', map_location='cpu'))
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Đọc ảnh từ request
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess theo chuẩn ImageNet
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    input_tensor = preprocess(image).unsqueeze(0)  # batch size = 1

    # Inference
    with torch.no_grad():
        logits = model(input_tensor)
    idx = torch.argmax(logits, dim=1).item()

    # Map index sang nhãn (theo thứ tự lớp khi train)
    labels = {
        0: "Levitan",
        1: "Non-Levitan"
    }
    return {"predicted_label": labels[idx]}
