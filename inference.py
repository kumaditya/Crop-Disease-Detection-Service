import json
import os
import torch
from torchvision import models
from PIL import Image

from config import MODEL_PATH, CLASSES_PATH, TOP_K
from utils.image_utils import prepare_image

class CropDiseaseClassifier:
    def __init__(self, model_path=MODEL_PATH, classes_path=CLASSES_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")
        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"Classes not found: {classes_path}. Run train.py first.")

        with open(classes_path, "r", encoding="utf-8") as f:
            self.classes = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)

        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image):
        x = prepare_image(image).to(self.device)
        probabilities = torch.softmax(self.model(x), dim=1)[0]
        values, indices = torch.topk(probabilities, k=min(TOP_K, len(self.classes)))

        predictions = [
            {"disease": self.classes[i], "confidence": round(v, 4)}
            for v, i in zip(values.tolist(), indices.tolist())
        ]

        return {
            "top_prediction": predictions[0],
            "top_predictions": predictions,
            "device": str(self.device),
        }
