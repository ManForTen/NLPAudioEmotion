import streamlit as st
from st_audiorec import st_audiorec
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import whisper
import torchaudio
import torch
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification
import tempfile
import os

# Load Hubert emotion prediction model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForSequenceClassification.from_pretrained("xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned")
num2emotion = {0: 'neutral', 1: 'angry', 2: 'positive', 3: 'sad', 4: 'other'}

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

    # Save the uploaded file to a temporary file
    temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    with open(temp_file_path, "wb") as temp_file:
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
    img = librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(y), ref=np.max), y_axis='log', x_axis='time', ax=ax_spec, cmap='viridis')
    plt.colorbar(img, format='%+2.0f dB')
    ax_spec.set_title('Spectrogram')
    st.pyplot(fig_spec)

    st.subheader("Распознавание текста:")

    # Load audio from the temporary file
    audio = whisper.load_audio(temp_file_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    st.write(f"Язык аудио: {max(probs, key=probs.get)}")
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    st.write("Текст:", result.text)

    # Predict emotion using the Hubert model
    waveform, sample_rate = torchaudio.load(temp_file_path, normalize=True)
    transform = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = transform(waveform)

    inputs = feature_extractor(
        waveform,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
        max_length=16000 * 10,
        truncation=True
    )

    logits = model(inputs['input_values'][0]).logits
    predictions = torch.argmax(logits, dim=-1)
    predicted_emotion = num2emotion[predictions.numpy()[0]]

    st.write("Предсказанная эмоция:", predicted_emotion)

    os.remove(temp_file_path)  # Remove the temporary file after processing
