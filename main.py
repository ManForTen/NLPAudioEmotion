from typing import io

import streamlit as st
from st_audiorec import st_audiorec
import torchaudio
import torchaudio.transforms as T
import torch
import soundfile as sf
import os
import tempfile
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

st.write("""
# Лабораторная работа 6
Запишите свой голос
""")

wav_audio_data = st_audiorec()

def plot_audio_info_with_emotion(uploaded_file):
    try:
        # Преобразование байтов в аудиофайл
        y, sr = librosa.load(uploaded_file, sr=None)

        # Построение графика временного сигнала (waveplot)
        st.pyplot(plt.figure(figsize=(12, 4)))
        librosa.display.waveshow(y, sr=sr)
        plt.title('Waveplot')
        plt.xlabel('Время (сек.)')
        plt.ylabel('Amplitude')
        plt.title('Waveplot')

        # Построение спектрограммы
        st.pyplot(plt.figure(figsize=(10, 4)))
        librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(y), ref=np.max), y_axis='log', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')



    except Exception as e:
        st.error(f"Ошибка при обработке файла: {e}")


# Загрузка аудиофайлов и вывод графиков
uploaded_files = st.file_uploader("Загрузите записанный голос", accept_multiple_files=True)

for uploaded_file in uploaded_files:
    st.write("filename:", uploaded_file.name)
    plot_audio_info_with_emotion(uploaded_file)