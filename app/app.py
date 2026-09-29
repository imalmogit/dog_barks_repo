"""Simple, cautious interface for the trained bark baseline."""
from __future__ import annotations
import sys, tempfile
from pathlib import Path
import streamlit as st
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predict import predict_file

st.set_page_config(page_title="Happy Bark?", page_icon="🐶")
st.title("🐶 Happy Bark?")
st.write("Upload or record a short clip with one clear dog bark.")
mode = st.radio("Audio source", ["Upload audio", "Record"], horizontal=True)
audio = st.file_uploader("Choose WAV, MP3, or M4A", type=["wav", "mp3", "m4a"]) if mode == "Upload audio" else st.audio_input("Record a bark")
if audio is not None:
    suffix = Path(audio.name if hasattr(audio, "name") else "recording.wav").suffix or ".wav"
    st.audio(audio.getvalue())
    if st.button("Is the dog likely happy?", type="primary"):
        model_path = ROOT / "models/happy_bark_baseline.joblib"
        if not model_path.exists(): st.error("The model has not been trained yet. Run prepare_data.py, then train.py.")
        else:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
                temp.write(audio.getvalue()); temp_path = temp.name
            try:
                label, confidence = predict_file(temp_path, str(model_path))

                if label == "happy":
                    st.success("Yes — likely happy")
                else:
                    st.info("No — not likely happy")

                st.caption(
                    f"Model confidence: {confidence:.0%}. "
                    "This is an estimate, not certainty."
                )
            except Exception:
                st.error("We couldn’t process this audio clip. Please try another clear bark recording.")
            finally:
                Path(temp_path).unlink(missing_ok=True)
else: st.caption("For a first try, use a 2–10 second clip with minimal background noise.")
st.divider()
st.caption("Experimental research prototype. It predicts human-provided dataset labels from bark audio; it cannot determine a dog's true emotional state and is not veterinary advice.")
