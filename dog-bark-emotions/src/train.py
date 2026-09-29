"""Train a lightweight MFCC + logistic-regression bark baseline."""
from __future__ import annotations
import json
from pathlib import Path
import joblib, librosa, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
SAMPLE_RATE = 16_000

def features(audio_path: str) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=10)
    if len(y) < 512: raise ValueError(f"Audio is too short or empty: {audio_path}")
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
def vectorise(frame: pd.DataFrame) -> np.ndarray: return np.vstack([features(path) for path in frame.audio_path])
def main() -> None:
    train, validation = pd.read_csv("data/processed/train.csv"), pd.read_csv("data/processed/validation.csv")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    model.fit(vectorise(train), train.binary_label)
    prediction = model.predict(vectorise(validation))
    metrics = {"validation_macro_f1": float(f1_score(validation.binary_label, prediction, average="macro")),
      "validation_report": classification_report(validation.binary_label, prediction, output_dict=True, zero_division=0),
      "feature": "20 MFCC mean + standard deviation; logistic regression",
      "label_definition": "happy=Positive valence; not_happy=Negative or Neutral valence",
      "warning": "Experimental label prediction, not a measurement of a dog's true emotion."}
    Path("models").mkdir(exist_ok=True); joblib.dump(model, "models/happy_bark_baseline.joblib")
    Path("models/validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Validation macro-F1: {metrics['validation_macro_f1']:.3f}")
    print("Do not use test.csv until the model choice is frozen; then run evaluate.py once.")
if __name__ == "__main__": main()
