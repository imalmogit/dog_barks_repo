"""Download, validate, and reproducibly split the Barkopedia emotion data."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.model_selection import train_test_split

REPO_ID = "ArlingtonCL2/BarkopediaDogEmotionClassification_Data"

def locate_audio(root: Path, audio_id: str) -> Path | None:
    matches = list(root.rglob(f"{audio_id}.wav"))
    return matches[0] if len(matches) == 1 else None

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-size", type=int, default=0,
        help="Number of clips to use. Default 0 uses all official training clips.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-dir", default="data/raw/barkopedia")
    args = parser.parse_args()
    if args.subset_size and args.subset_size < 60:
        raise ValueError("--subset-size must be 0 (all clips) or at least 60.")
    root = Path(snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=args.download_dir,
        allow_patterns=["train/**/*.wav", "husky_train_labels.csv", "shiba_train_labels.csv"]))
    tables = []
    for breed in ("husky", "shiba"):
        csv_path = root / f"{breed}_train_labels.csv"
        table = pd.read_csv(csv_path)
        required = {"audio_id", "arousal", "valence"}
        if not required.issubset(table.columns):
            raise ValueError(f"{csv_path} needs {sorted(required)}; got {list(table.columns)}")
        table["breed"] = breed
        tables.append(table)
    labels = pd.concat(tables, ignore_index=True)
    labels["audio_path"] = labels.audio_id.map(lambda x: locate_audio(root, x))
    missing = labels.audio_path.isna().sum()
    if missing:
        raise RuntimeError(f"Could not match {missing} official audio_id values to WAV files.")
    labels["audio_path"] = labels.audio_path.map(str)
    labels["binary_label"] = labels.valence.map({"Positive": "happy", "Neutral": "not_happy", "Negative": "not_happy"})
    if labels.binary_label.isna().any():
        raise ValueError("Unexpected valence value in official labels.")
    counts = labels.valence.value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    print("Official training labels by valence:\n" + counts.to_string())
    if args.subset_size == 0:
        # Use all 1,000 official training clips. The classifier handles the
        # natural 429/571 binary imbalance with class-weighted training.
        subset = labels.sample(frac=1, random_state=args.seed)
    else:
        # A smaller study is balanced by binary class, never by the viewer's
        # generic breed label.
        per_class = args.subset_size // 2
        if per_class > min((labels.binary_label == "happy").sum(), (labels.binary_label == "not_happy").sum()):
            raise ValueError("Requested subset is larger than the smallest binary class.")
        happy = labels[labels.binary_label == "happy"].sample(per_class, random_state=args.seed)
        not_happy = labels[labels.binary_label == "not_happy"].sample(per_class, random_state=args.seed)
        subset = pd.concat([happy, not_happy], ignore_index=True).sample(frac=1, random_state=args.seed)
    train, remainder = train_test_split(subset, test_size=.30, stratify=subset.binary_label, random_state=args.seed)
    validation, test = train_test_split(remainder, test_size=.50, stratify=remainder.binary_label, random_state=args.seed)
    out = Path("data/processed"); out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        frame.sort_values("audio_id").to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(frame)} clips; {frame.binary_label.value_counts().to_dict()}")
    pd.DataFrame({"valence": counts.index, "count": counts.values}).to_csv(out / "official_train_valence_counts.csv", index=False)
    print(f"Wrote reproducible {len(subset)}-clip subset (seed={args.seed}).")

if __name__ == "__main__": main()
