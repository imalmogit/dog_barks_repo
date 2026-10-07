"""Compare Neutral/Negative classifiers without changing the binary baseline."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from train import vectorise

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "models" / "neutral_negative"
MODEL_PATH = OUTPUT / "neutral_negative_baseline.joblib"
CLASSES = ["Neutral", "Negative"]


def load_split(name: str) -> pd.DataFrame:
    """Filter a saved split in memory; never recreate or write split CSVs."""
    frame = pd.read_csv(ROOT / "data" / "processed" / f"{name}.csv")
    required = {"audio_id", "audio_path", "valence"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{name} split needs columns {sorted(required)}")
    if not frame.valence.isin(["Positive", *CLASSES]).all():
        raise ValueError(f"Unexpected or missing valence in {name} split")
    if frame.audio_id.isna().any() or frame.audio_id.duplicated().any():
        raise ValueError(f"Missing or duplicate audio IDs in {name} split")
    frame = frame.loc[frame.valence.isin(CLASSES)].copy()
    if set(frame.valence) != set(CLASSES):
        raise ValueError(f"{name} split must contain both Neutral and Negative")
    frame["audio_path"] = frame.audio_path.map(
        lambda path: str(Path(path) if Path(path).is_absolute() else ROOT / path)
    )
    return frame


def class_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {label: int((frame.valence == label).sum()) for label in CLASSES}


def metrics_for(actual, predicted) -> dict:
    return {
        "macro_f1": float(f1_score(actual, predicted, labels=CLASSES, average="macro", zero_division=0)),
        "report": classification_report(actual, predicted, labels=CLASSES, output_dict=True, zero_division=0),
        "confusion_matrix_labels": CLASSES,
        "confusion_matrix": confusion_matrix(actual, predicted, labels=CLASSES).tolist(),
    }


def save_confusion(actual, predicted, path: Path) -> None:
    display = ConfusionMatrixDisplay.from_predictions(actual, predicted, labels=CLASSES)
    display.figure_.tight_layout()
    display.figure_.savefig(path, dpi=160)
    plt.close(display.figure_)


def main() -> None:
    train, validation = load_split("train"), load_split("validation")
    if set(train.audio_id) & set(validation.audio_id):
        raise ValueError("Train and validation splits overlap")
    print(f"Training class counts: {class_counts(train)}", flush=True)
    print(f"Validation class counts: {class_counts(validation)}", flush=True)
    print("Extracting the existing baseline audio features...", flush=True)
    x_train, x_validation = vectorise(train), vectorise(validation)
    y_train, y_validation = train.valence, validation.valence
    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
        ),
        "rbf_svm": CalibratedClassifierCV(
            estimator=make_pipeline(
                StandardScaler(),
                SVC(C=2.0, gamma="scale", class_weight="balanced", probability=False, random_state=42),
            ),
            method="sigmoid",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            ensemble=True,
        ),
    }
    majority = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    majority_metrics = metrics_for(y_validation, majority.predict(x_validation))
    majority_metrics["predicted_class"] = str(majority.predict(x_validation[:1])[0])
    results, predictions = {}, {}
    for name, model in candidates.items():
        print(f"Fitting {name}...", flush=True)
        model.fit(x_train, y_train)
        predictions[name] = model.predict(x_validation)
        results[name] = metrics_for(y_validation, predictions[name])
        print(f"{name} validation macro-F1: {results[name]['macro_f1']:.6f}", flush=True)
    selected = max(results, key=lambda name: results[name]["macro_f1"])
    metrics = {
        "train_class_counts": class_counts(train),
        "validation_class_counts": class_counts(validation),
        "majority_baseline": majority_metrics,
        "validation_macro_f1_by_model": {name: result["macro_f1"] for name, result in results.items()},
        "validation_results_by_model": results,
        "selected_model": selected,
        "validation_macro_f1": results[selected]["macro_f1"],
        "validation_report": results[selected]["report"],
        "feature": "Existing train.py features: 16 kHz mono, first 10 s, 116 MFCC/delta/chroma/spectral/energy summaries",
        "label_definition": "Original Neutral versus Negative valence; Positive excluded within each saved split",
        "svm_calibration": "Sigmoid, five shuffled stratified training folds, seed 42, ensemble=True; scaling fitted inside each fold",
        "selection": "Highest validation macro-F1; logistic regression wins exact ties; no refit on validation",
        "test_evaluated": False,
        "warning": "Experimental human-label prediction; clip-disjoint splits do not establish unseen-dog performance.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidates[selected], MODEL_PATH)
    (OUTPUT / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion(y_validation, predictions[selected], OUTPUT / "validation_confusion_matrix.png")
    print(f"Majority baseline ({majority_metrics['predicted_class']}) validation macro-F1: {majority_metrics['macro_f1']:.6f}")
    print(f"Selected model: {selected}")
    print(classification_report(y_validation, predictions[selected], labels=CLASSES, zero_division=0))
    print("Test evaluation was not run. Review and freeze validation choices before evaluating test.csv.")


if __name__ == "__main__":
    main()
