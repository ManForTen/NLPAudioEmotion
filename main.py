# This is a sample Python script.
import streamlit as st
import numpy as np
from st_audiorec import st_audiorec

st.write("""
# My first app
Hello *world!*
""")

wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')

