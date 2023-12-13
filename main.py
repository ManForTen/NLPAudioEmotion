# This is a sample Python script.
from streamlit_audiorec import st_audiorec
import streamlit as st


st.write("""
# My first app
Hello *world!*
""")
wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')

