import os

import streamlit as st
from st_audiorec import st_audiorec
import whisper
import soundfile as sf
import numpy as np
import tempfile

st.write("""
# Лабораторная работа 6
Запишите свой голос
""")

wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    # Создаем временный файл для сохранения аудио
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_audio_path = temp_audio_file.name

    try:
        # Сохраняем аудио во временный файл
        sf.write(temp_audio_path, wav_audio_data, 44100)

        # Используем конструктор WhisperASR
        model = whisper.WhisperASR(model_size="medium")

        # Читаем аудио из временного файла
        audio, sample_rate = whisper.read_wave(temp_audio_path)

        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        st.write(f"Язык аудио: {max(probs, key=probs.get)}")

        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)

        st.write("Текст:", result.text)
    except Exception as e:
        st.error(f"Произошла ошибка: {e}")
    finally:
        # Удаляем временный файл после использования
        os.remove(temp_audio_path)
else:
    st.warning("Аудио не было записано.")
