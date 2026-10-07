"""Predict Neutral/Negative for a clip assumed to have non-positive valence."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from train import features

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "neutral_negative" / "neutral_negative_baseline.joblib"


def predict_file(audio_path: str, model_path: str | Path = MODEL_PATH) -> tuple[str, float]:
    model = joblib.load(model_path)
    if set(model.classes_) != {"Neutral", "Negative"}:
        raise ValueError("Expected a Neutral/Negative model")
    vector = [features(audio_path)]
    label = str(model.predict(vector)[0])
    probabilities = model.predict_proba(vector)[0]
    index = list(model.classes_).index(label)
    return label, float(probabilities[index])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    label, confidence = predict_file(args.audio_path, args.model_path)
    print(f"{label} - model confidence {confidence:.0%}")
    print("Conditional Neutral/Negative prediction; this model cannot identify happy clips.")
