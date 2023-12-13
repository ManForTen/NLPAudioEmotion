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
    # Создание графика амплитуды
    audio_array = np.squeeze(wav_audio_data)
    time = np.arange(0, len(audio_array)) / 44100  # Временная ось в секундах

    plt.figure(figsize=(10, 4))
    plt.plot(time, audio_array)
    plt.title('График амплитуды')
    plt.xlabel('Время (сек)')
    plt.ylabel('Амплитуда')
    st.pyplot()





