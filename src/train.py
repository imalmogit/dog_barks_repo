"""Train and compare CPU-friendly audio classifiers on bark recordings."""
from __future__ import annotations
import json
from pathlib import Path
import joblib, librosa, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
SAMPLE_RATE = 16_000

def features(audio_path: str) -> np.ndarray:
    """MFCC, delta-MFCC, spectral and energy summaries for one clip."""
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=10)
    if len(y) < 512: raise ValueError(f"Audio is too short or empty: {audio_path}")
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20)
    if mfcc.shape[1] < 3:
        delta = np.zeros_like(mfcc)
    else:
        delta_width = min(
            9,
            mfcc.shape[1] if mfcc.shape[1] % 2 else mfcc.shape[1] - 1,
        )
        delta = librosa.feature.delta(
            mfcc, width=delta_width, mode="nearest"
        )
    chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE)
    spectral = np.vstack([
        librosa.feature.rms(y=y),
        librosa.feature.zero_crossing_rate(y),
        librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE),
        librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE),
        librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE),
        librosa.feature.spectral_flatness(y=y),
    ])
    def summarise(matrix: np.ndarray) -> np.ndarray:
        return np.concatenate([matrix.mean(axis=1), matrix.std(axis=1)])
    return np.concatenate([summarise(mfcc), summarise(delta), summarise(chroma), summarise(spectral)])
def vectorise(frame: pd.DataFrame) -> np.ndarray: return np.vstack([features(path) for path in frame.audio_path])
def main() -> None:
    train, validation = pd.read_csv("data/processed/train.csv"), pd.read_csv("data/processed/validation.csv")
    x_train, x_validation = vectorise(train), vectorise(validation)
    y_train, y_validation = train.binary_label, validation.binary_label
    candidates = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        "rbf_svm": make_pipeline(StandardScaler(), SVC(C=2.0, gamma="scale", class_weight="balanced", probability=True, random_state=42)),
    }
    results = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        predicted = model.predict(x_validation)
        results[name] = float(f1_score(y_validation, predicted, average="macro"))
    selected_name = max(results, key=results.get)
    model = candidates[selected_name]
    prediction = model.predict(x_validation)
    metrics = {"validation_macro_f1": results[selected_name],
      "validation_macro_f1_by_model": results,
      "selected_model": selected_name,
      "validation_report": classification_report(y_validation, prediction, output_dict=True, zero_division=0),
      "feature": "MFCC + delta-MFCC + chroma + spectral/energy summaries",
      "label_definition": "happy=Positive valence; not_happy=Negative or Neutral valence",
      "warning": "Experimental label prediction, not a measurement of a dog's true emotion."}
    Path("models").mkdir(exist_ok=True); joblib.dump(model, "models/happy_bark_baseline.joblib")
    Path("models/validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Selected model: {selected_name}")
    print(f"Validation macro-F1: {metrics['validation_macro_f1']:.3f}")
    print("Do not use test.csv until the model choice is frozen; then run evaluate.py once.")
if __name__ == "__main__": main()
