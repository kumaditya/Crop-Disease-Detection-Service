import streamlit as st
from PIL import Image

from inference import CropDiseaseClassifier
from llm_service import explain_prediction
from config import CONFIDENCE_THRESHOLD

st.set_page_config(page_title="Crop Disease Detection", page_icon="🌱", layout="wide")

st.title("🌱 Crop Disease Detection")
st.write(
    "Upload a crop leaf image. A CNN predicts the disease and an LLM "
    "converts the structured prediction into a natural-language explanation."
)

@st.cache_resource
def load_classifier():
    return CropDiseaseClassifier()

try:
    classifier = load_classifier()
except Exception as exc:
    st.error(str(exc))
    st.info("Train first: python train.py --data_dir data/dataset --epochs 10")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a leaf image", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    left, right = st.columns(2)

    with left:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with right:
        with st.spinner("Running disease detection..."):
            result = classifier.predict(image)

        top = result["top_prediction"]
        confidence = top["confidence"]

        st.subheader("CNN Prediction")
        st.metric("Detected disease", top["disease"], f"{confidence:.1%} confidence")

        if confidence < CONFIDENCE_THRESHOLD:
            st.warning("Low-confidence prediction. Consider a clearer image or expert inspection.")

        st.subheader("Top predictions")
        for item in result["top_predictions"]:
            st.write(f"**{item['disease']}** — {item['confidence']:.1%}")

    st.divider()
    st.subheader("🤖 Natural-language explanation")

    with st.spinner("Generating explanation..."):
        explanation = explain_prediction(result)

    st.markdown(explanation)

    st.caption(
        "AI screening only: field diagnosis can require expert inspection, "
        "additional images, laboratory testing, weather information and crop context."
    )
