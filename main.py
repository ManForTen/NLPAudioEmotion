# This is a sample Python script.
import streamlit as st
import sounddevice as sd
import numpy as np

st.write("""
# My first app
Hello *world!*
""")


def record_audio(duration=5, sample_rate=44100):
    st.text("Идет запись...")

    # Запись аудио
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()

    st.text("Запись завершена!")

    return audio_data

def main():
    st.title("Запись голоса с помощью Streamlit")

    duration = st.slider("Выберите длительность записи (в секундах)", min_value=1, max_value=10, value=5)

    if st.button("Начать запись"):
        audio_data = record_audio(duration)

        # Визуализация аудиоданных
        st.line_chart(np.squeeze(audio_data))

        # Сохранение аудио
        st.audio(audio_data, format="audio/wav", start_time=0)

if __name__ == "__main__":
    main()

