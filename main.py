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

uploaded_files = st.file_uploader("Загрузите записанный голос", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    st.write(bytes_data)