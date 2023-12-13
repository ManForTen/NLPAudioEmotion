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

uploaded_file = st.file_uploader("Загрузите аудиофайл (допускаются файлы формата wav)", type=["wav"])

if uploaded_file:
    st.write("Файл успешно загружен!")

    # Преобразование байтов в аудиофайл
    y, sr = librosa.load(uploaded_file, sr=None)

    # Построение графика временного сигнала (waveplot)
    st.subheader("Waveplot:")
    fig_wave, ax_wave = plt.subplots(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax_wave)
    ax_wave.set_title('Waveplot')
    ax_wave.set_xlabel('Время (сек.)')
    ax_wave.set_ylabel('Amplitude')

    # Сохранение графика как изображения
    waveplot_image = "waveplot.png"
    fig_wave.savefig(waveplot_image)
    st.image(waveplot_image)

    # Построение спектрограммы
    st.subheader("Spectrogram:")
    fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(y), ref=np.max), y_axis='log', x_axis='time', ax=ax_spec)
    plt.colorbar(format='%+2.0f dB')
    ax_spec.set_title('Spectrogram')

    # Сохранение графика как изображения
    spectrogram_image = "spectrogram.png"
    fig_spec.savefig(spectrogram_image)
    st.image(spectrogram_image)
