import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 224

model = tf.keras.models.load_model("skin_model_generator.keras")

class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

st.title("🧴 Skin Disease Detection App")
st.write("Upload a dermatoscopic image and get the model's prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    pred_class = np.argmax(preds)
    confidence = preds[pred_class] * 100

    st.markdown(f"### 🔍 Prediction: `{class_names[pred_class]}` with {confidence:.2f}% confidence")

    st.subheader("Confidence Scores")
    for idx, score in enumerate(preds):
        st.write(f"{class_names[idx]}: {score:.2%}")
