import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

from tensorflow.keras.callbacks import EarlyStopping

from feature_extraction import extract_features
from preprocessing import pad_features
from model import build_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "ravdess")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

X = []
y = []

for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            emotion = file.split("-")[2]

            # ORIGINAL
            features = extract_features(file_path, augment=False)
            if features is not None:
                X.append(pad_features(features))
                y.append(emotion)

            # LIGHT AUGMENTATION (ONLY NOISE)
            features_aug = extract_features(file_path, augment=True)
            if features_aug is not None:
                X.append(pad_features(features_aug))
                y.append(emotion)

X = np.array(X)
y = np.array(y)

print("Loaded:", X.shape)

le = LabelEncoder()
y = le.fit_transform(y)

X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = build_model((120, 120, 1), len(set(y)))

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    epochs=25,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

loss, acc = model.evaluate(X_test, y_test)
print("Final Accuracy:", acc)

model.save(os.path.join(MODEL_DIR, "model.keras"))
joblib.dump(le, os.path.join(MODEL_DIR, "encoder.pkl"))

print("Model saved!")