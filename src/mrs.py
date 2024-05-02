import librosa #https://librosa.org/    #sudo apt-get install -y ffmpeg (open mp3 files)
import librosa.display
import librosa.beat
import sounddevice as sd  #https://anaconda.org/conda-forge/python-sounddevice
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import skew, kurtosis
import numpy.fft

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
    f0 = librosa.yin(y=y, fmin=20, fmax=sr/2)[0]
    f0[f0==sr/2]=0
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
def normalize_feats(features, min_values=None, max_values=None):
    if min_values is None:
        min_values = np.min(features, axis = 0)
    if max_values is None:
        max_values = np.max(features, axis = 0)
        normalized_features = (features - min_values) / (max_values - min_values)
    if min_values is None or max_values is None:
        normalized_features = np.vstack((max_values, normalized_features))
        normalized_features = np.vstack((min_values, normalized_features))
    
    return normalized_features, min_values, max_values

# 2.1
def features():
    all_feats = []
    count = 0
    for file_name in os.listdir(audio_folder):
        if os.path.isfile(os.path.join(audio_folder, file_name)):
            file_path = os.path.join(audio_folder, file_name)
            # 2.1.1
            feats = extract_features(file_path)
            all_feats.append(feats)
            manualCentroid(file_path)
            
    calculateManualCentroids()
    # 2.1.3
    norm_feats, min_vals, max_vals = normalize_feats(all_feats)
    # 2.1.4
    np.savetxt('../out/feats1.csv', all_feats, delimiter=',', fmt="%.6f")
    np.savetxt('../out/norm_feats1.csv', norm_feats, delimiter=',', fmt="%.6f")
    
    return np.asarray(all_feats), np.asarray(norm_feats), min_vals, max_vals

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))

def cosine_distance(x, y):
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return 1.0 - (dot_product / (norm_x * norm_y))

def manualCentroid(filename):
    sr = 22050
    mono = True
    warnings.filterwarnings("ignore")
    filePath = os.path.join(filename)
    y, fs = librosa.load(filename, sr=sr, mono = mono)
    n_fft = 2048
    hop_length = 512
    sc = np.zeros((len(y)-n_fft+1)//hop_length +1) #fazer (tamanho do array//hop - 3) em vez disto (é mais correto I guess idk)
    counter= 0
    df=sr/n_fft
    #freq=0,10,20,30,...
    freqs = np.arange(0, sr/2 + df, df)
    #hann window
    yw = np.hanning(n_fft)
    for i in range(0, len(y) - n_fft + 1, hop_length):
        #FFT
        yf = np.fft.rfft(y[i:i+n_fft]*yw)#usar rfft
        #Espectro de potencia
        magnitudes = np.abs(yf)
        #SC
        if np.sum(magnitudes)==0:
            sc[counter]=(0)
        else:
            sc[counter]=(np.sum(magnitudes*freqs)/np.sum(magnitudes))# se denominador for 0 centroid=0
        counter+=1
    librosa_sc=librosa.feature.spectral_centroid(y = y)[0][2:len(sc)+2]
    # print("Pearson Correlation: ",np.corrcoef(librosa_sc,sc)[0][1])
    # print("RMSE: ",np.sqrt(np.mean((librosa_sc-sc)**2)))
    return np.corrcoef(librosa_sc,sc)[0][1], np.sqrt(np.mean((librosa_sc-sc)**2))

def calculateManualCentroids():
    counter = 0
    allCentroids=np.zeros((900,2))
    for filename in os.listdir(audio_folder):    
        if filename.endswith(".mp3"):
            allCentroids[counter]=manualCentroid(f"{audio_folder}/{filename}")
        # print(counter)
        counter+=1
    #save csv de all Centroids
    np.savetxt("allCentroids.csv", allCentroids, delimiter=",", fmt="%f")#fmt=%f

# 3
def similarity_metrics(all_feats, min_vals, max_vals):
    query = "../Queries\MT0000414517.mp3"
    feats = extract_features(query)
    query_feats = normalize_feats(feats, min_vals, max_vals)
    print(query_feats)
    '''
    similarity_matrices = {}
    for metric in ['euclidean', 'manhattan', 'cosine']:
        similarity_matrix = np.zeros((len(all_feats), len(all_feats)))
        for i, feat1 in enumerate(all_feats):
            for j, feat2 in enumerate(all_feats):  # Iterate over all_feats for both feat1 and feat2
                if metric == 'euclidean':
                    similarity_matrix[i, j] = euclidean_distance(feat1, feat2)  # Calculate similarity between feat1 and feat2
                elif metric == 'manhattan':
                    similarity_matrix[i, j] = manhattan_distance(feat1, feat2)
                elif metric == 'cosine':
                    similarity_matrix[i, j] = cosine_distance(feat1, feat2)
        similarity_matrices[metric] = similarity_matrix  # Store similarity matrix for the current metric
    return similarity_matrices
'''
    
if __name__ == "__main__":
    plt.close('all')
    # 2.1
    not_norm_feats, norm_feats, min_values, max_values = features()
    # 3
    similarity_metrics(norm_feats, min_values, max_values)
