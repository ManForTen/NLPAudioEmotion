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

if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')

    # Создание графика амплитуды
    audio_array = np.squeeze(wav_audio_data)

    # Проверка, что audio_array не пуст
    if len(audio_array) > 0:
        time = np.arange(0, len(audio_array)) / 44100  # Временная ось в секундах

        plt.figure(figsize=(10, 4))
        plt.plot(time, audio_array)
        plt.title('График амплитуды')
        plt.xlabel('Время (сек)')
        plt.ylabel('Амплитуда')
        st.pyplot()
    else:
        st.warning("Аудио-данные пусты.")
else:
    st.warning("Аудио не было записано.")






