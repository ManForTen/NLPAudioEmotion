import streamlit as st
from st_audiorec import st_audiorec
import torchaudio
import torchaudio.transforms as T
import torch
import soundfile as sf
import os
import tempfile

st.write("""
# Лабораторная работа 6
Запишите свой голос
""")

wav_audio_data = st_audiorec()

wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    # Создаем временный файл для сохранения аудио
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_audio_path = temp_audio_file.name

    try:
        # Сохраняем аудио во временный файл
        sf.write(temp_audio_path, wav_audio_data, 44100)

        st.write("Аудио успешно сохранено. Путь к файлу:", temp_audio_path)
    except Exception as e:
        st.error(f"Произошла ошибка при сохранении аудио: {e}")
    finally:
        # Удаляем временный файл после использования
        os.remove(temp_audio_path)
else:
    st.warning("Аудио не было записано.")