import streamlit as st
from st_audiorec import st_audiorec
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import whisper

# Загрузка модели Whisper
model = whisper.load_model("base")

st.write("""
# Лабораторная работа 6
Запишите свой голос
""")

wav_audio_data = st_audiorec()

# Загрузка аудиофайла
uploaded_file = st.file_uploader("Загрузите аудиофайл (допускаются файлы формата wav)", type=["wav"])

if uploaded_file:
    st.write("Файл успешно загружен!")

    # Преобразование байтов в аудиофайл
    y, sr = librosa.load(uploaded_file, sr=None)

    st.audio(uploaded_file, format='audio/wav')

    # Построение графика временного сигнала (waveplot)
    st.subheader("Waveplot:")
    fig_wave, ax_wave = plt.subplots(figsize=(12, 4))
    ax_wave.plot(np.linspace(0, len(y) / sr, len(y)), y)
    ax_wave.set_title('Waveplot')
    ax_wave.set_xlabel('Время (сек.)')
    ax_wave.set_ylabel('Amplitude')
    st.pyplot(fig_wave)

    # Построение спектрограммы
    st.subheader("Spectrogram:")
    fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
    transform = T.MelSpectrogram(sample_rate=sr, n_fft=400, hop_length=160, n_mels=128)
    mel_spec = transform(torch.tensor(y).view(1, -1))
    log_mel_spec = T.AmplitudeToDB()(mel_spec)
    img = librosa.display.specshow(log_mel_spec[0].numpy(), y_axis='log', x_axis='time', ax=ax_spec, cmap='viridis')
    plt.colorbar(img, format='%+2.0f dB')
    ax_spec.set_title('Spectrogram')
    st.pyplot(fig_spec)

    st.subheader("Распознавание текста:")
    audio = whisper.load_audio(uploaded_file)
    audio = whisper.pad_or_trim(audio)

    transform = T.MelSpectrogram(sample_rate=sr, n_fft=400, hop_length=160, n_mels=128)
    mel_spec = transform(torch.tensor(audio).view(1, -1))
    log_mel_spec = T.AmplitudeToDB()(mel_spec)

    mel = log_mel_spec.to(model.device)
    _, probs = model.detect_language(mel)
    st.write(f"Язык аудио: {max(probs, key=probs.get)}")

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    st.write("Текст:", result.text)
