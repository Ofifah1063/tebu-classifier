import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Ukuran gambar dan urutan kelas harus sama dengan saat training
IMG_SIZE = (128, 128)
KELAS = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]


@st.cache_resource
def muat_model():
    return tf.keras.models.load_model("model.keras")


model = muat_model()

st.title("Klasifikasi Penyakit Daun Tebu")
file = st.file_uploader("Upload foto daun tebu", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Gambar yang diunggah")

    # Tidak dibagi 255 di sini karena model sudah punya layer Rescaling
    x = np.array(img.resize(IMG_SIZE), dtype="float32")
    pred = model.predict(x[np.newaxis], verbose=0)[0]
    probs = tf.nn.softmax(pred).numpy()

    st.subheader("Prediksi: " + KELAS[int(np.argmax(probs))])
    st.write(f"Tingkat keyakinan: {probs.max():.1%}")
    st.bar_chart(dict(zip(KELAS, probs)))
