import torch
from PIL import Image
from torchvision import transforms
from model_loader import model

classes = ["glioma", "meningioma", "no_tumor", "pituitary"]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

def predict_image(image_path: str):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(image)   # 🔴 IMPORTANT FIX
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)

    return {
        "class": classes[pred.item()],
        "confidence": round(conf.item() * 100, 2)
    }
