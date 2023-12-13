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
from io import BytesIO

st.write("""
# Лабораторная работа 6
Запишите свой голос
""")

wav_audio_data = st_audiorec()

# Выбор загруженного .wav файла
uploaded_files = st.file_uploader("Загрузите записанный голос", accept_multiple_files=True)

# Для каждого загруженного файла
for uploaded_file in uploaded_files:
    # Чтение аудиофайла
    audio_tensor, sample_rate = torchaudio.load(uploaded_file)

    # Проверка, что аудио моно (1 канал)
    if audio_tensor.shape[0] == 2:
        audio_tensor = audio_tensor.mean(dim=0, keepdim=True)

    # Применение преобразований, например, вычисление спектрограммы
    transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=128)
    mel_spectrogram = transform(audio_tensor)

    # Нормализация значений спектрограммы в диапазон [0.0, 1.0]
    normalized_mel_spectrogram = (mel_spectrogram - mel_spectrogram.min()) / (mel_spectrogram.max() - mel_spectrogram.min())

    # Вывод спектрограммы
    st.write(f"Спектрограмма для файла: {uploaded_file.name}")
    st.image(normalized_mel_spectrogram[0].numpy(), caption='Мел-спектрограмма', use_column_width=True)