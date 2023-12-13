import streamlit as st
from st_audiorec import st_audiorec


st.write("""
# Лабораторная работа 6
Запишите свой голос
""")
wav_audio_data = st_audiorec()





