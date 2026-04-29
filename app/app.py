import streamlit as st
import numpy as np
import joblib
import librosa
import os
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = load_model(
    os.path.join(BASE_DIR, "models", "model.h5"),
    compile=False
)
le = joblib.load(os.path.join(BASE_DIR, "models", "encoder.pkl"))

# 🎯 Emotion Mapping
emotion_map = {
    "01": "Neutral 😐",
    "02": "Calm 😌",
    "03": "Happy 😊",
    "04": "Sad 😢",
    "05": "Angry 😡",
    "06": "Fearful 😨",
    "07": "Disgust 🤢",
    "08": "Surprised 😲"
}

def extract_features(file):
    audio, sr = librosa.load(file, duration=3, offset=0.5)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack((mfcc, delta, delta2)).T

    if features.shape[0] > 120:
        features = features[:120, :]
    else:
        pad = 120 - features.shape[0]
        features = np.pad(features, ((0, pad), (0, 0)))

    return features.reshape(1, 120, 120, 1)

# 🌟 UI Styling
st.set_page_config(page_title="Emotion Detection", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #00BFFF;'>🎤 Speech Emotion Recognition</h1>
    <hr>
""", unsafe_allow_html=True)

file = st.file_uploader("📂 Upload a WAV file", type=["wav"])

if file:
    st.audio(file)

    features = extract_features(file)
    pred = model.predict(features)[0]

    labels = le.classes_

    # Map encoded → emotion name
    mapped_labels = [emotion_map.get(lbl, lbl) for lbl in labels]

    predicted_emotion = emotion_map.get(labels[np.argmax(pred)], labels[np.argmax(pred)])

    # 🎯 Highlight result
    st.markdown(f"""
        <div style='background-color:#1E8449;padding:15px;border-radius:10px;text-align:center'>
        <h2 style='color:white;'>🎯 Predicted Emotion: {predicted_emotion}</h2>
        </div>
    """, unsafe_allow_html=True)

    # DataFrame
    df = pd.DataFrame({
        "Emotion": mapped_labels,
        "Probability": pred
    }).sort_values(by="Probability", ascending=False)

    st.markdown("## 📊 Emotion Analysis")

    col1, col2 = st.columns(2)

    # 📊 Bar Chart
    with col1:
        st.subheader("Bar Representation")
        st.bar_chart(df.set_index("Emotion"))

    # 🥧 Pie Chart
    with col2:
        fig = px.pie(
            df,
            names="Emotion",
            values="Probability",
            title="Emotion Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    # 📋 Legend / Key
    st.markdown("## 🔑 Emotion Key")
    for k, v in emotion_map.items():
        st.write(f"{k} → {v}")