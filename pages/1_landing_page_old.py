import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Audio Denoising App", page_icon="🎧")

st.title("Audio Denoising Demo")
st.write("Upload a noisy audio file and choose a denoising model.")

model = st.selectbox(
    "Choose model",
    [
        "spectral_subtraction",
        "geometric_subtraction",
        "mlp",
        "unet",
    ],
)

uploaded_file = st.file_uploader(
    "Upload noisy audio",
    type=["wav", "flac", "mp3", "ogg"],
)

if uploaded_file is not None:
    st.subheader("Original audio")
    st.audio(uploaded_file, format="audio/wav")

    if st.button("Enhance audio"):
        with st.spinner("Denoising..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }

            response = requests.post(
                f"{API_URL}/enhance",
                params={"model": model},
                files=files,
            )

        if response.status_code == 200:
            enhanced_audio = response.content

            st.success("Enhancement complete!")

            st.subheader("Enhanced audio")
            st.audio(enhanced_audio, format="audio/wav")

            st.download_button(
                label="Download enhanced WAV",
                data=enhanced_audio,
                file_name=f"enhanced_{uploaded_file.name}",
                mime="audio/wav",
            )

            st.write("Model used:", response.headers.get("X-Model-Used"))
            st.write("Processing time:", response.headers.get("X-Processing-Time"), "seconds")

        else:
            st.error("Enhancement failed")
            st.json(response.json())