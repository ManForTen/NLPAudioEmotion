import streamlit as st
from st_audiorec import st_audiorec
import torch
import torchaudio
from transformers import pipeline

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
    y, sr = torchaudio.load(uploaded_file, normalize=True)

    st.audio(uploaded_file, format='audio/wav')

    # Построение графика временного сигнала (waveplot)
    st.subheader("Waveplot:")
    st.line_chart(y[0].numpy())

    # Определение эмоции с использованием transformers
    st.subheader("Эмоция:")

    # Load pre-trained emotion classification model
    classifier = pipeline('audio-classification', model='maksimekin/emotion-audio')

    # Convert audio tensor to numpy array
    audio_np = y[0].numpy()

    # Classify emotion
    result = classifier(audio_np)

    # Display the result
    st.write(result)
