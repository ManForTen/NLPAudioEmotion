import os

import streamlit as st
from st_audiorec import st_audiorec
import whisper
import soundfile as sf
import numpy as np
import tempfile
import speech_recognition as sr

st.write("""
# Лабораторная работа 6
Запишите свой голос
""")




def record_audio():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        st.write("Говорите что-нибудь...")
        audio = r.listen(source)

    st.write("Запись завершена. Текст:")

    try:
        text = r.recognize_google(audio, language="ru-RU")
        st.write(text)
    except sr.UnknownValueError:
        st.write("Не удалось распознать речь")
    except sr.RequestError as e:
        st.write(f"Ошибка при обращении к сервису распознавания речи: {e}")


def main():
    st.title("Голосовая запись в Streamlit")

    if st.button("Начать запись"):
        record_audio()


if __name__ == "__main__":
    main()
