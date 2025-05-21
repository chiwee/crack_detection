import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import tempfile
import io
from fpdf import FPDF
import matplotlib.pyplot as plt
import os

# Load trained model once
@st.cache_resource
def load_model():
    return joblib.load("trained_model_LogisticRegression.pkl")

model = load_model()

# Preprocessing function
def preprocess_image(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    resized = cv2.resize(edges, (64, 64))
    return resized.flatten().reshape(1, -1)

# Streamlit UI
st.title("🧠 Image Classification with Pretrained Model")
uploaded_files = st.file_uploader("Upload multiple images...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

results = []

if uploaded_files:
    st.subheader("🖼️ Image Predictions")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=uploaded_file.name, use_column_width=True)

        features = preprocess_image(image)
        prediction = model.predict(features)[0]
        label = "Positive" if prediction == 1 else "Negative"

        st.markdown(f"**Prediction for {uploaded_file.name}:** {label}")
        results.append((image, uploaded_file.name, label))

    # Pie chart
    st.subheader("📊 Overall Prediction Summary")
    labels_list = [label for _, _, label in results]
    positives = labels_list.count("Positive")
    negatives = labels_list.count("Negative")
    
    fig, ax = plt.subplots()
    ax.pie([positives, negatives], labels=["Positive", "Negative"], autopct='%1.1f%%', colors=["green", "red"])
    ax.axis("equal")
    st.pyplot(fig)

    # Save pie chart as image
    pie_path = "pie_chart.png"
    fig.savefig(pie_path)

    # PDF Generation
    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Add Pie Chart to PDF
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, "Overall Prediction Summary", ln=True)
        pdf.image(pie_path, x=10, y=30, w=180)

        # Add each image & prediction
        for img, name, label in results:
            pdf.add_page()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                img.save(tmpfile.name, "JPEG")
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, f"Image: {name}", ln=True)
                pdf.cell(200, 10, f"Prediction: {label}", ln=True)
                pdf.image(tmpfile.name, x=10, y=30, w=80)

        # Convert to BytesIO
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        pdf_output = io.BytesIO(pdf_bytes)

        # Download button
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_output,
            file_name="prediction_report.pdf",
            mime="application/pdf"
        )
