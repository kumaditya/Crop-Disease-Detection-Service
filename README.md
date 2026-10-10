# Crop Disease Detection + LLM Explanation

A Python application that detects crop diseases from leaf images using a CNN (ResNet18), then uses an LLM to turn the structured prediction into a natural-language explanation.

Architecture:
Image -> preprocessing -> ResNet18 CNN -> disease probabilities -> LLM explanation -> Streamlit UI

The CNN performs classification. The LLM explains the classifier result; it is not used to invent or replace the diagnosis.

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Install:
```bash
pip install -r requirements.txt
```

Put a folder-per-class dataset under `data/dataset/`:

```text
data/dataset/
├── Apple___Apple_scab/
├── Apple___healthy/
├── Tomato___Early_blight/
├── Tomato___Late_blight/
└── Tomato___healthy/
```

Train:

```bash
python train.py --data_dir data/dataset --epochs 10
```

This creates:
- `models/crop_disease_model.pth`
- `models/classes.json`

Configure Gemini by copying `.env.example` to `.env` and adding your API key.

Run:

```bash
streamlit run app.py
```

The UI displays the uploaded image, disease prediction, confidence, top-3 predictions, and an LLM-generated explanation.

For production, validate the model on field images from the target crops, geography, cameras and lighting conditions. Laboratory-style datasets may not represent real field conditions.
