"""Use the trained binary bark baseline on one audio file."""
from __future__ import annotations
import argparse
import joblib
from train import features

def predict_file(audio_path: str, model_path="models/happy_bark_baseline.joblib") -> tuple[str, float]:
    model = joblib.load(model_path)
    probabilities = model.predict_proba([features(audio_path)])[0]
    index = int(probabilities.argmax())
    return str(model.classes_[index]), float(probabilities[index])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("audio_path")
    label, confidence = predict_file(parser.parse_args().audio_path)
    print(f"{'Yes' if label == 'happy' else 'No'} — model confidence {confidence:.0%}")
