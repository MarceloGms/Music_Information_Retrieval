import librosa #https://librosa.org/    #sudo apt-get install -y ffmpeg (open mp3 files)
import librosa.display
import librosa.beat
import sounddevice as sd  #https://anaconda.org/conda-forge/python-sounddevice
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import skew, kurtosis

audio_folder = "../MER_audio_taffc_dataset/songs"

# 2.1.1
def extract_features(file_path):        
    y, sr = librosa.load(file_path)
    
    # features espectrais
    mfcc = librosa.feature.mfcc(y=y,n_mfcc=13)
    mfcc = calculate_stats(mfcc,1).flatten()
    
    spectral_centroid = librosa.feature.spectral_centroid(y=y)
    spectral_centroid = calculate_stats(spectral_centroid,1).flatten()
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y)
    spectral_bandwidth = calculate_stats(spectral_bandwidth,1).flatten()
    
    spectral_contrast = librosa.feature.spectral_contrast(y=y)
    spectral_contrast = calculate_stats(spectral_contrast,1).flatten()
    
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    spectral_flatness = calculate_stats(spectral_flatness,1).flatten()
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y)
    spectral_rolloff = calculate_stats(spectral_rolloff,1).flatten()
    
    # features temporais
    f0 = librosa.yin(y=y, fmin=20, fmax=11025)[0]
    f0 = calculate_stats(f0).flatten()
    
    rms = librosa.feature.rms(y=y)
    rms = calculate_stats(rms,1).flatten()
    
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    zero_crossing_rate = calculate_stats(zero_crossing_rate,1).flatten()
    
    # outras features
    tempo = librosa.feature.rhythm.tempo(y=y)
    
    features = np.concatenate([mfcc, spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_flatness, spectral_rolloff, f0, rms, zero_crossing_rate, tempo])
    
    return features

# 2.1.2
def calculate_stats(features, axis = None):
    mean = np.mean(features, axis = axis)
    std = np.std(features, axis = axis)
    skewness = skew(features, axis = axis)
    kurt = kurtosis(features, axis = axis)
    median = np.median(features, axis = axis)
    max_val = np.max(features,axis = axis)
    min_val = np.min(features, axis = axis)

    return np.array([mean, std, skewness, kurt, median, max_val, min_val])

# 2.1.3
def normalize_feats(features):
    min_values = np.min(features, axis = 0)
    max_values = np.max(features, axis = 0)
    normalized_features = (features - min_values) / (max_values - min_values)
    normalized_features = np.vstack((max_values, normalized_features))
    normalized_features = np.vstack((min_values, normalized_features))
    
    return normalized_features

# 2.1
def features():
    all_feats = []
    for file_name in os.listdir(audio_folder):
        if os.path.isfile(os.path.join(audio_folder, file_name)):
            file_path = os.path.join(audio_folder, file_name)
            # 2.1.1
            feats = extract_features(file_path)
            all_feats.append(feats)
            
    # 2.1.3
    norm_feats = normalize_feats(all_feats)
    # 2.1.4
    np.savetxt('../out/feats.csv', all_feats, delimiter=',', fmt="%.6f")
    np.savetxt('../out/norm_feats.csv', norm_feats, delimiter=',', fmt="%.6f")
    
    return np.asarray(all_feats), np.asarray(norm_feats)
    
if __name__ == "__main__":
    plt.close('all')
    # 2.1
    not_norm_feats, norm_feats = features()
    