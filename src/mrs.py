"""
Created on Tue Apr  6 13:03:06 2021

@author: rpp
"""

import librosa #https://librosa.org/    #sudo apt-get install -y ffmpeg (open mp3 files)
import librosa.display
import librosa.beat
import sounddevice as sd  #https://anaconda.org/conda-forge/python-sounddevice
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os

audio_folder = "../MER_audio_taffc_dataset/songs"

# Função para extrair todas as features necessárias de um arquivo de áudio
def extract_features():
    for file_name in os.listdir(audio_folder):
        if os.path.isfile(os.path.join(audio_folder, file_name)):
            file_path = os.path.join(audio_folder, file_name)
            
            features = []
            
            filey = librosa.load(file_path)
            features.append(librosa.feature.mfcc(filey))
            features.append(librosa.feature.spectral_centroid(filey))
            features.append(librosa.feature.spectral_bandwidth(filey))
            features.append(librosa.feature.spectral_contrast(filey))
            features.append(librosa.feature.spectral_flatness(filey))
            features.append(librosa.feature.spectral_rolloff(filey))
            features.append(librosa.feature.yin(filey))
            features.append(librosa.feature.rms(filey))
            features.append(librosa.feature.zero_crossing_rate(filey))
            features.append(librosa.beat.tempo(filey))
            
            print(features)


if __name__ == "__main__":
    plt.close('all')
    extract_features()
    
    #--- Load file
    '''fName = "Queries/MT0000414517.mp3"    
    sr = 22050
    mono = True
    warnings.filterwarnings("ignore")
    y, fs = librosa.load(fName, sr=sr, mono = mono)
    print(y.shape)
    print(fs)
    
    #--- Play Sound
    #sd.play(y, sr, blocking=False)
    
    #--- Plot sound waveform
    plt.figure()
    librosa.display.waveshow(y)
    
    #--- Plot spectrogram
    Y = np.abs(librosa.stft(y))
    Ydb = librosa.amplitude_to_db(Y, ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(Ydb, y_axis='linear', x_axis='time', ax=ax)
    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")'''
    