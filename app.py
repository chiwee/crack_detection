import streamlit as st
import cv2
import numpy as np
import joblib

# Load the trained model once
@st.cache_resource
def load_model():
    model = joblib.load("trained_model_LogisticRegression.pkl")  # adjust filename/model if needed
    return model

model = load_model()

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    resized = cv2.resize(edges, (64, 64))
    flattened = resized.flatten()
    return flattened.reshape(1, -1)

st.title("Image Classification with Pretrained Model")

uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, channels="BGR", caption="Uploaded Image")

    # Preprocess and predict
    features = preprocess_image(img)
    prediction = model.predict(features)[0]
    label = "Positive" if prediction == 1 else "Negative"

    st.markdown(f"### Prediction: **{label}**")
