import librosa
import streamlit as st
from matplotlib import pyplot as plt
from st_audiorec import st_audiorec
import torch
import torchaudio
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

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

    # Построение спектрограммы
    st.subheader("Spectrogram:")
    fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(y), ref=np.max), y_axis='log', x_axis='time', ax=ax_spec, cmap='viridis')
    plt.colorbar(img, format='%+2.0f dB')
    ax_spec.set_title('Spectrogram')
    st.pyplot(fig_spec)

    # Определение эмоции
    st.subheader("Эмоция:")

    # Явная загрузка модели и токенизатора
    model_name = "maksimekin/emotion-audio"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Преобразование аудио в текст с использованием токенизатора
    inputs = tokenizer(" ".join(map(str, y[0].numpy())), return_tensors="pt", padding=True, truncation=True)

    # Классификация эмоции
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Отображение результата
    st.write(f"Predicted Emotion: {predicted_class}")
