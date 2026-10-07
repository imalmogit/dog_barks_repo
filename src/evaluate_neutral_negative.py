"""Evaluate held-out Neutral/Negative clips only after validation choices freeze."""
from __future__ import annotations

import json

import joblib

from train import vectorise
from train_neutral_negative import MODEL_PATH, OUTPUT, class_counts, load_split, metrics_for, save_confusion


def main() -> None:
    model = joblib.load(MODEL_PATH)
    test = load_split("test")
    predicted = model.predict(vectorise(test))
    results = metrics_for(test.valence, predicted)
    results["test_macro_f1"] = results.pop("macro_f1")
    results["test_class_counts"] = class_counts(test)
    validation = json.loads((OUTPUT / "validation_metrics.json").read_text(encoding="utf-8"))
    results["selected_model"] = validation["selected_model"]
    majority_class = validation["majority_baseline"]["predicted_class"]
    results["majority_baseline"] = metrics_for(test.valence, [majority_class] * len(test))
    results["majority_baseline"]["predicted_class"] = majority_class
    (OUTPUT / "test_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_confusion(test.valence, predicted, OUTPUT / "test_confusion_matrix.png")
    print(f"Held-out Neutral/Negative test macro-F1: {results['test_macro_f1']:.6f}")


if __name__ == "__main__":
    main()
