import numpy as np
import joblib
from tensorflow.keras.models import load_model

from feature_extraction import extract_features
from preprocessing import pad_features

model = load_model("../models/model.keras")
le = joblib.load("../models/encoder.pkl")

def predict(file_path):
    features = extract_features(file_path)
    features = pad_features(features)
    features = features.reshape(1, 120, 120, 1)

    pred = model.predict(features)
    return le.inverse_transform([np.argmax(pred)])[0]