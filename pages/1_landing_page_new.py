import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Audio Denoising App", page_icon="🎧")

st.title("Audio Denoising Demo")
st.write("Upload a noisy audio file and choose a denoising model.")

mode = st.radio(
    "Choose task",
    ["Enhance audio", "Evaluate with clean reference"],
)

model = st.selectbox(
    "Choose model",
    [
        "spectral_subtraction",
        "geometric_subtraction",
        "mlp",
        "unet",
    ],
)

if mode == "Enhance audio":
    noisy_file = st.file_uploader(
        "Upload noisy audio",
        type=["wav", "flac", "mp3", "ogg"],
        key="noisy_file",
    )

clean_file = None
enhanced_file = None

if mode == "Evaluate with clean reference":
    noisy_file = None
    enhanced_file = st.file_uploader(
        "Upload enhanced audio",
        type=["wav", "flac", "mp3", "ogg"],
        key="enhanced_file",
    )

    clean_file = st.file_uploader(
        "Upload clean reference audio",
        type=["wav", "flac", "mp3", "ogg"],
        key="clean_file",
    )

if noisy_file is not None:
    st.subheader("Noisy audio")
    st.audio(noisy_file, format="audio/wav")

if clean_file is not None:
    st.subheader("Clean reference audio")
    st.audio(clean_file, format="audio/wav")

if enhanced_file is not None:
    st.subheader("Enhanced audio")
    st.audio(enhanced_file, format="audio/wav")

if mode == "Enhance audio":
    if noisy_file is not None:
        if st.button("Enhance audio"):
            with st.spinner("Denoising..."):
                files = {
                    "file": (
                        noisy_file.name,
                        noisy_file.getvalue(),
                        noisy_file.type,
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
                    file_name=f"enhanced_{noisy_file.name}",
                    mime="audio/wav",
                )

                st.write("Model used:", response.headers.get("X-Model-Used"))
                st.write(
                    "Processing time:",
                    response.headers.get("X-Processing-Time"),
                    "seconds",
                )

            else:
                st.error("Enhancement failed")
                try:
                    st.json(response.json())
                except Exception:
                    st.write(response.text)

elif mode == "Evaluate with clean reference":
    if enhanced_file is not None and clean_file is not None:
        if st.button("Evaluate model"):
            with st.spinner("Enhancing and evaluating..."):
                files = {
                    "noisy_file": (
                        enhanced_file.name,
                        enhanced_file.getvalue(),
                        enhanced_file.type,
                    ),
                    "clean_file": (
                        clean_file.name,
                        clean_file.getvalue(),
                        clean_file.type,
                    ),
                }

                response = requests.post(
                    f"{API_URL}/evaluate",
                    params={"model": model},
                    files=files,
                )

            if response.status_code == 200:
                result = response.json()

                st.success("Evaluation complete!")

                st.subheader("Summary")
                st.write("Model used:", result["model_used"])
                st.write(
                    "Input duration:",
                    result["input_duration_seconds"],
                    "seconds",
                )
                st.write(
                    "Processing time:",
                    result["processing_time_seconds"],
                    "seconds",
                )

                st.subheader("Metrics")
                metrics_df = pd.DataFrame(
                    list(result["metrics"].items()),
                    columns=["Metric", "Value"],
                )
                st.dataframe(metrics_df, use_container_width=True)

                st.subheader("Compared to noisy baseline")

                baseline_rows = []
                for metric, values in result["above_random_baseline"].items():
                    baseline_rows.append(
                        {
                            "Metric": metric,
                            "Noisy baseline": values["noisy_baseline"],
                            "Enhanced": values["enhanced"],
                            "Delta": values["delta"],
                            "Improved": values["improved"],
                        }
                    )

                baseline_df = pd.DataFrame(baseline_rows)
                st.dataframe(baseline_df, use_container_width=True)

            else:
                st.error("Evaluation failed")
                try:
                    st.json(response.json())
                except Exception:
                    st.write(response.text)

    else:
        st.info("Upload both an enhanced audio file and a clean reference file.")