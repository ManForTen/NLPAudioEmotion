import streamlit as st
from st_audiorec import st_audiorec
import numpy as np
import librosa.display
import matplotlib.pyplot as plt


st.write("""
# Лабораторная работа 6
Запишите свой голос
""")

wav_audio_data = st_audiorec()

# Проверка, что запись прошла успешно
if wav_audio_data is not None:
    # Преобразование аудио в одномерный массив с плавающей точкой
    audio_array = np.squeeze(wav_audio_data)

    # Нормализация аудио, учитывая возможность деления на 0
    max_abs_value = np.max(np.abs(audio_array))
    audio_array = audio_array / max_abs_value if max_abs_value != 0 else audio_array

    # Создание спектрограммы с использованием librosa
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_array)), ref=np.max)

    # Отображение спектрограммы с использованием matplotlib
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=44100, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    st.pyplot()







