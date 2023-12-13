import streamlit as st
from st_audiorec import st_audiorec
import speech_recognition as sr
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

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
    img = librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(y), ref=np.max), y_axis='log', x_axis='time',
                                   ax=ax_spec, cmap='viridis')
    plt.colorbar(img, format='%+2.0f dB')
    ax_spec.set_title('Spectrogram')
    st.pyplot(fig_spec)

    # Определение эмоции с использованием speech_recognition
    st.subheader("Эмоция:")

    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(uploaded_file)

    with audio_file as source:
        audio_data = recognizer.record(source)

    try:
        emotion = recognizer.recognize_google(audio_data, language='ru-RU')
        st.write("Определенная эмоция: ", emotion)
    except sr.UnknownValueError:
        st.write("Не удалось определить эмоцию")
    except sr.RequestError as e:
        st.write("Ошибка сервиса распознавания речи; {0}".format(e))
