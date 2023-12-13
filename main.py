import streamlit as st
from st_audiorec import st_audiorec
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

import whisper

st.write("""
# Лабораторная работа 6
Запишите свой голос
""")

wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    # Сохранение аудио в файл
    audio_filename = "recorded_audio.wav"
    sf.write(audio_filename, wav_audio_data, 44100)

    # Преобразование аудио в формат, поддерживаемый whisper
    audio, _ = whisper.read_wave(audio_filename)

    if audio is not None:
        # Используем конструктор WhisperASR
        model = whisper.WhisperASR(model_size="medium")

        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        st.write(f"Язык аудио: {max(probs, key=probs.get)}")

        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)

        st.write("Текст:", result.text)
    else:
        st.warning("Не удалось прочитать аудио.")
else:
    st.warning("Аудио не было записано.")






