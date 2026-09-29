"""Run from anywhere; no datasets or models are downloaded."""
import argparse
import importlib
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--ml", action="store_true", help="Also check ML dependencies")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
failed = False
print("Python:", sys.version.split()[0])
if sys.version_info < (3, 10):
    print("FAIL: Python 3.10+ is required; the guide uses 3.12.")
    failed = True
packages = ["streamlit"]
if args.ml:
    packages += ["torch", "huggingface_hub", "librosa", "soundfile", "numpy", "pandas", "sklearn", "joblib", "matplotlib"]
for name in packages:
    try:
        module = importlib.import_module(name)
        print("OK:", name, getattr(module, "__version__", ""))
    except Exception as error:
        failed = True
        print("FAIL:", name, str(error))
for name in ["data/raw", "data/processed", "notebooks", "src", "models", "app"]:
    exists = (root / name).is_dir()
    print("OK:" if exists else "FAIL:", name)
    failed = failed or not exists
print("Setup needs attention." if failed else "Setup checks passed. You can start the app.")
sys.exit(1 if failed else 0)
