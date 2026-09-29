"""Run after model choices are frozen: evaluate the held-out test set."""
from __future__ import annotations
import json
from pathlib import Path
import joblib, matplotlib.pyplot as plt, pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from train import vectorise

def main() -> None:
    test = pd.read_csv("data/processed/test.csv")
    predicted = joblib.load("models/happy_bark_baseline.joblib").predict(vectorise(test))
    score = f1_score(test.binary_label, predicted, average="macro")
    Path("models/test_metrics.json").write_text(json.dumps({"test_macro_f1":score, "report":classification_report(test.binary_label,predicted,output_dict=True,zero_division=0)}, indent=2), encoding="utf-8")
    ConfusionMatrixDisplay.from_predictions(test.binary_label, predicted); plt.tight_layout(); plt.savefig("models/test_confusion_matrix.png", dpi=160)
    print(f"Held-out test macro-F1: {score:.3f}")
if __name__ == "__main__": main()
