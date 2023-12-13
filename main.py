import streamlit as st
from st_audiorec import st_audiorec
import torchaudio
import torchaudio.transforms as T
import torch
import soundfile as sf
import os
import tempfile

st.write("""
# Лабораторная работа 6
Запишите свой голос
""")

wav_audio_data = st_audiorec()

uploaded_files = st.file_uploader("Загрузите записанный голос", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    # Чтение аудиофайла
    audio_tensor, sample_rate = torchaudio.load(uploaded_file)

    # Проверка, что аудио моно (1 канал)
    if audio_tensor.shape[0] == 2:
        audio_tensor = audio_tensor.mean(dim=0, keepdim=True)

    # Применение преобразований, например, вычисление спектрограммы
    transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=128)
    mel_spectrogram = transform(audio_tensor)

    # Вывод спектрограммы
    st.write(f"Спектрограмма для файла: {uploaded_file.name}")
    st.image(mel_spectrogram[0].numpy(), caption='Мел-спектрограмма', use_column_width=True)