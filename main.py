import streamlit as st
from st_audiorec import st_audiorec
import whisper
import soundfile as sf
import numpy as np

st.write("""
# Лабораторная работа 6
Запишите свой голос
""")

wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    # Преобразование аудио в формат, поддерживаемый whisper
    audio, sample_rate = whisper.read_wave(wav_audio_data)

    if audio is not None:
        # Убедимся, что audio - это двумерный массив
        if len(audio.shape) == 2:
            # Сохранение аудио в файл
            audio_filename = "recorded_audio.wav"
            sf.write(audio_filename, audio, sample_rate)

            # Используем конструктор WhisperASR
            model = whisper.WhisperASR(model_size="medium")

            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            _, probs = model.detect_language(mel)
            st.write(f"Язык аудио: {max(probs, key=probs.get)}")

            options = whisper.DecodingOptions(fp16=False)
            result = whisper.decode(model, mel, options)

            st.write("Текст:", result.text)
        else:
            st.warning("Некорректный формат аудио. Ожидается двумерный массив.")
    else:
        st.warning("Не удалось прочитать аудио.")
else:
    st.warning("Аудио не было записано.")
