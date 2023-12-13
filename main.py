import streamlit as st
from st_audiorec import st_audiorec
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import whisper
import tempfile
import os

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

    # Create a temporary file and get the file path
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file_path = temp_file.name
    temp_file.write(uploaded_file.read())

    st.audio(temp_file_path, format='audio/wav')

    # Преобразование байтов в аудиофайл
    y, sr = librosa.load(temp_file_path, sr=None)

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
    img = librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(y), ref=np.max), y_axis='log', x_axis='time',
                                   ax=ax_spec, cmap='viridis')
    plt.colorbar(img, format='%+2.0f dB')
    ax_spec.set_title('Spectrogram')
    st.pyplot(fig_spec)

    st.subheader("Распознавание текста:")

    # Close the temporary file before loading it
    temp_file.close()

    # Load audio from the temporary file
    audio = whisper.load_audio(temp_file_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    st.write(f"Язык аудио: {max(probs, key=probs.get)}")
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    st.write("Текст:", result.text)

    os.remove(temp_file_path)  # Remove the temporary file after processing
