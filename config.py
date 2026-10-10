import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/crop_disease_model.pth")
CLASSES_PATH = os.getenv("CLASSES_PATH", "models/classes.json")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

IMAGE_SIZE = 224
TOP_K = 3
CONFIDENCE_THRESHOLD = 0.60
