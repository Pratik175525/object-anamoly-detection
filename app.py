import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Define labels
class_labels = {
    0: "Defective",
    1: "Not Defective"
}

# Load TFLite model
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Streamlit UI
st.title("Bottle Defect Detection")

uploaded_file = st.file_uploader("Upload an image of a bottle", type=["jpg", "jpeg", "png","webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize and preprocess image
    size = (224, 224)  # or change based on model
    image = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_label = np.argmax(predictions)
    confidence = predictions[predicted_label]

    # Display raw scores and prediction
    st.write("Raw prediction scores:", predictions)
    st.success(f"Prediction: **{class_labels[predicted_label]}** (Confidence: {confidence:.2f})")
