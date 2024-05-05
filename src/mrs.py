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
from scipy.spatial import distance

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
    f0 = librosa.yin(y=y, fmin=20, fmax=sr/2)
    f0[f0==sr/2]=0
    f0 = calculate_stats(f0).flatten()    
    
    rms = librosa.feature.rms(y=y)
    rms = calculate_stats(rms,1).flatten()
    
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    zero_crossing_rate = calculate_stats(zero_crossing_rate,1).flatten()
    
    # outras features
    tempo = librosa.feature.rhythm.tempo(y=y)
    
    features = np.concatenate([
        mfcc, 
        spectral_centroid, 
        spectral_bandwidth, 
        spectral_contrast, 
        spectral_flatness, 
        spectral_rolloff, 
        f0, 
        rms, 
        zero_crossing_rate, 
        tempo
    ])
    
    return features

# 2.1.2
def calculate_stats(features, axis = 0):
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
    features = np.array(features)
    min_values = np.min(features, axis=0)
    max_values = np.max(features, axis=0)
    
    # Features onde min == max vao ser 0
    zero_mask = min_values == max_values
    
    normalized_features = np.where(zero_mask, 0, features)
    
    for i in range(len(features[0])):
        if not zero_mask[i]:
            normalized_features[:, i] = (features[:, i] - min_values[i]) / (max_values[i] - min_values[i])
            
    normalized_features = np.vstack((max_values, normalized_features))
    normalized_features = np.vstack((min_values, normalized_features))
    
    return normalized_features

# 2.1
def features():
    all_feats = []
    counter = 0
    all_centroids=np.zeros((900,2))
    for file_name in os.listdir(audio_folder):
        if os.path.isfile(os.path.join(audio_folder, file_name)):
            file_path = os.path.join(audio_folder, file_name)
            # 2.1.1
            feats = extract_features(file_path)
            all_feats.append(feats)
            # 2.2.1
            all_centroids[counter]=manual_centroid(file_path)
            counter+=1
            
    all_feats = np.vstack(all_feats)
    # 2.1.3
    norm_feats = normalize_feats(all_feats)
    # 2.1.4
    np.savetxt('../out/feats.csv', all_feats, delimiter=',', fmt="%.6f")
    np.savetxt('../out/norm_feats.csv', norm_feats, delimiter=',', fmt="%.6f")
    # 2.2.3
    np.savetxt("../out/allCentroids.csv", all_centroids, delimiter=",", fmt="%f")
    
    return np.asarray(all_feats), np.asarray(norm_feats)

# 2.2.1
def manual_centroid(filename):
    sr = 22050
    mono = True
    warnings.filterwarnings("ignore")
    y, fs = librosa.load(filename, sr = sr, mono = mono)
    n_fft = 2048
    hop_length = 512
    sc = np.zeros((len(y)-n_fft+1)//hop_length + 1)
    counter= 0
    df=sr/n_fft
    #freq=0,10,20,30,...
    freqs = np.arange(0, sr/2 + df, df)
    #hann window
    yw = np.hanning(n_fft)
    for i in range(0, len(y) - n_fft + 1, hop_length):
        #FFT
        yf = np.fft.rfft(y[i:i+n_fft]*yw)
        #Espectro de potencia
        magnitudes = np.abs(yf)
        #SC
        if np.sum(magnitudes)==0:
            sc[counter]=(0)
        else:
            sc[counter]=(np.sum(magnitudes*freqs)/np.sum(magnitudes))# se denominador for 0 centroid=0
        counter+=1
    librosa_sc=librosa.feature.spectral_centroid(y = y)[0][2:len(sc)+2]

    return np.corrcoef(librosa_sc,sc)[0][1], np.sqrt(np.mean((librosa_sc-sc)**2))

# 3.1.1
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

# 3.1.2
def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))

# 3.1.3
def cosine_distance(x, y):
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return 1.0 - (dot_product / (norm_x * norm_y))

def normalize_query(query, min_max):
    normalized_query = np.zeros(190)
    for i in range(190):
        if min_max[1][i] == min_max[0][i]:
            normalized_query[i] = 0
        else:
            normalized_query[i] = (query[i] - min_max[0][i]) / (min_max[1][i] - min_max[0][i])
    return normalized_query

#3.1
def calc_similarity(norm_feats):
    euclidean = np.zeros(900)
    manhattan = np.zeros(900)
    cosine = np.zeros(900)
    
    for file_name in os.listdir("../Queries"):
        if os.path.isfile(os.path.join("../Queries", file_name)):
            file_path = os.path.join("../Queries", file_name)
            
            # 3.2.1 extrair e normalizar features da query
            query_features = extract_features(file_path)
            np.savetxt('../out/query.csv', query_features, delimiter=',', fmt="%.6f", newline=",")
            query_features = normalize_query(query_features, norm_feats[:2])
            np.savetxt('../out/norm_query.csv', np.vstack((norm_feats[:2], query_features)), delimiter=',', fmt="%.6f")
            
            # 3.2.2 calcular e guardar as distancias
            for i in range(900):
                euclidean[i] = euclidean_distance(norm_feats[i+2], query_features)
                manhattan[i] = manhattan_distance(norm_feats[i+2], query_features)
                cosine[i] = cosine_distance(norm_feats[i+2], query_features)

            np.savetxt('../out/euclidean.csv', euclidean, delimiter=',', fmt="%.6f")
            np.savetxt('../out/manhattan.csv', manhattan, delimiter=',', fmt="%.6f")
            np.savetxt('../out/cosine.csv', cosine, delimiter=',', fmt="%.6f")

            # 3.3 criar e guardar os rankings de similaridade
            euclidean_ranking = np.sort(euclidean)[:10]
            manhattan_ranking = np.sort(manhattan)[:10]
            cosine_ranking = np.sort(cosine)[:10]
            
            with open('../out/rankings.txt', 'w') as f:
                f.write(f"Ranking: Euclidean-------------\n{euclidean_ranking}\n\n")
                f.write(f"Ranking: Manhattan-------------\n{manhattan_ranking}\n\n")
                f.write(f"Ranking: Cosine-------------\n{cosine_ranking}\n")
    
if __name__ == "__main__":
    plt.close('all')
    # 2.1
    not_norm_feats, norm_feats = features()
    # 3
    #norm_feats = np.loadtxt('../out/norm_feats.csv', delimiter=',')
    calc_similarity(norm_feats)