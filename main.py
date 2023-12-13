import streamlit as st
from st_audiorec import st_audiorec


st.write("""
# Лабораторная работа 6
Запишите свой голос
""")
wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')
a = str(wav_audio_data)
st.write(a)


