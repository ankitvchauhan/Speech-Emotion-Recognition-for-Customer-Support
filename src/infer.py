import sys
import json
import numpy as np
import joblib
import librosa
import os

from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.keras")
ENC_PATH = os.path.join(BASE_DIR, "models", "encoder.pkl")

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)

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

def main():
    wav_path = sys.argv[1]

    model = load_model(MODEL_PATH)
    le = joblib.load(ENC_PATH)

    x = extract_features(wav_path)
    pred = model.predict(x, verbose=0)[0]

    labels = le.classes_
    idx = int(np.argmax(pred))

    out = {
        "predicted": labels[idx],
        "probs": pred.tolist(),
        "labels": labels.tolist()
    }
    print(json.dumps(out))

if __name__ == "__main__":
    main()