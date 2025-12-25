import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

UPLOAD_DIR = "user_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ------------------------------
# Load trained model
# ------------------------------
MODEL_PATH = "mobilenetv2_skin_cancer.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (128, 128)

# ------------------------------
# Image preprocessing
# ------------------------------
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Skin Cancer Detection", layout="centered")

st.title("🩺 Skin Cancer Detection System")
st.write("Upload a skin lesion image to classify it as **Healthy** or **Unhealthy**.")

uploaded_file = st.file_uploader(
    "Upload Image (JPG / PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # save image
    save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    image.save(save_path)

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.success(f"Image saved to: {save_path}")

    # preprocess & predict
    img = preprocess_image(image)
    prob = model.predict(img)[0][0]

    if prob >= 0.5:
        label = "Unhealthy"
        label_id = 1
        confidence = prob
        st.error(f"⚠️ Prediction: {label} ({label_id})")
    else:
        label = "Healthy"
        label_id = 0
        confidence = 1 - prob
        st.success(f"✅ Prediction: {label} ({label_id})")

    st.write(f"**Confidence:** {confidence:.2f}")


    st.markdown("---")
    st.caption(
        "⚠️ This tool is for educational purposes only and does not replace professional medical diagnosis."
    )
