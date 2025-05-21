import streamlit as st
import cv2
import numpy as np
import joblib
from fpdf import FPDF
import tempfile
from PIL import Image
import io

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("trained_model_LogisticRegression.pkl")

model = load_model()

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    resized = cv2.resize(edges, (64, 64))
    flattened = resized.flatten()
    return flattened.reshape(1, -1)

st.title("Batch Image Classification & PDF Report Generator")

uploaded_files = st.file_uploader("Upload multiple images...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

results = []

if uploaded_files:
    st.subheader("Preview and Prediction Results:")
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        features = preprocess_image(img)
        prediction = model.predict(features)[0]
        label = "Positive" if prediction == 1 else "Negative"

        st.image(img, channels="BGR", width=150, caption=f"{uploaded_file.name} - Prediction: {label}")

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        results.append((pil_img, uploaded_file.name, label))

    # Generate PDF only if button is clicked
    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        for img, name, label in results:
            pdf.add_page()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                img.save(tmpfile.name, "JPEG")
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, f"Image: {name}", ln=True)
                pdf.cell(200, 10, f"Prediction: {label}", ln=True)
                pdf.image(tmpfile.name, x=10, y=30, w=80)

        # Convert PDF to bytes
        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_output,
            file_name="prediction_report.pdf",
            mime="application/pdf"
        )
