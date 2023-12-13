import streamlit as st
from st_audiorec import st_audiorec
import torchaudio
import torchaudio.transforms as T
import torch

st.write("""
# Лабораторная работа 6
Запишите свой голос
""")

wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    # Преобразование аудио в тензор
    waveform, sample_rate = torchaudio.load(io.BytesIO(wav_audio_data), normalize=True)

    # Если аудио стерео, преобразуем его в моно
    if waveform.shape[0] == 2:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Применение преобразований, например, каскадного удаления шума
    transform = T.Compose([
        T.Resample(orig_freq=sample_rate, new_freq=16000),
        T.Vad(sample_rate=16000),
        T.MFCC(sample_rate=16000, n_mfcc=13)
    ])

    # Получение признаков
    features = transform(waveform)

    # Вывод результатов
    st.write("Форма вейвформы:", waveform.shape)
    st.write("Форма признаков:", features.shape)
else:
    st.warning("Аудио не было записано.")
