import numpy as np

def pad_features(features, max_len=120):
    if features.shape[0] > max_len:
        return features[:max_len, :]
    else:
        pad_width = max_len - features.shape[0]
        return np.pad(features, ((0, pad_width), (0, 0)), mode='constant')