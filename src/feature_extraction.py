import librosa
import numpy as np

def extract_features(file_path, augment=False):
    try:
        audio, sr = librosa.load(file_path, duration=3, offset=0.5)

        if augment:
            noise = 0.003 * np.random.randn(len(audio))
            audio = audio + noise

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        features = np.vstack((mfcc, delta, delta2))
        return features.T

    except:
        return None