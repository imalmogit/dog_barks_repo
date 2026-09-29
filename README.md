# Dog Bark Emotions — Steps 1 and 2

This starter defines the first version and sets up a Python project. It does not contain a trained model or download datasets. The app lets you upload or record audio and play it back; predictions stay disabled until a real model is implemented.

## Step 1: Define the first version

Goal: Given a short recording containing a dog bark, estimate the dataset's arousal and valence categories, with a separate model score for each output.

- Arousal = level of activation: Low / Medium / High.
- Valence = negative / neutral / positive emotional tone.
- A highly activated bark is not automatically negative: the two outputs answer different questions.
- These are predictions of human-annotated categories, not proof of a dog's internal emotional state or a veterinary diagnosis.
- Initially accept short WAV uploads or microphone recordings. Aim for 2–10 seconds as a practical user guideline, not a dataset requirement.
- First train on known dog vocalizations. Add automatic bark detection later, before accepting arbitrary everyday recordings as valid model inputs.
- Final flow: upload/record → check for dog vocalization → predict both targets → show model scores. A model score must not be presented as a guaranteed probability of being correct.

### Dataset decision

Start with ArlingtonCL2/BarkopediaDogEmotionClassification_Data:
https://huggingface.co/datasets/ArlingtonCL2/BarkopediaDogEmotionClassification_Data

The dataset card describes 1,000 training clips from huskies and shibas. Emotion labels are in husky_train_labels.csv and shiba_train_labels.csv. The automatically inferred `label` field visible in the preview describes breed, NOT emotion. When preparing data, join the CSV emotion annotations to the corresponding audio files and verify the actual column names first.

Keep DogSpeak for possible later pretraining and Barkopedia-Dog-Vocal-Detection for a later detector. Do not merge the three datasets as if they share emotion labels.

### How we will evaluate later

Report macro-F1 and a confusion matrix separately for arousal and valence, and compare against a simple baseline. Macro-F1 treats each class equally. Keep clips from the same dog/source recording together when IDs are available; split BEFORE creating crops or augmentations. If those IDs are absent, document possible overlap and avoid claiming performance on unseen dogs. Freeze the held-out test set before tuning. Generalization beyond the two training breeds needs additional evidence.

Step 1 is complete when you can explain the input, both outputs, initial dataset, and the limits above.

## Step 2: Set up on Windows with VS Code

1. Install Python 3.12, 64-bit, from https://www.python.org/downloads/ if needed. Python 3.11 is also suitable for this starter. Install VS Code and its Microsoft Python extension if needed.
2. Extract the ZIP. In VS Code choose File > Open Folder and select the inner `dog-bark-emotions` folder containing this README.
3. Choose Terminal > New Terminal. The commands below work in PowerShell from that folder.
4. Check Python: `py --version`. If the launcher is unavailable but `python --version` works with Python 3.11 or 3.12, use `python` instead of `py` in the next command.
5. Create a private environment: `py -m venv .venv`. If several Python versions are installed, use `py -3.12 -m venv .venv` to select Python 3.12.
6. Install the starter dependency:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

A virtual environment is a separate box of Python packages for this project. These commands use its Python directly, so you do not need to activate it or change PowerShell execution policy.

7. In VS Code, press Ctrl+Shift+P, select Python: Select Interpreter, and choose `.venv`. If it is missing, browse to `.venv\Scripts\python.exe`.
8. Check setup and run the app:

```powershell
.\.venv\Scripts\python.exe src/check_setup.py
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

Open the Local URL printed in the terminal if your browser does not open automatically. Keep the terminal running. Stop with Ctrl+C.

9. Upload a WAV file or select Record and allow microphone access. Play back the audio. The app should say that emotion prediction is not available yet. This is expected.

### Install the ML tools

After the starter works, install packages for the later data and training steps:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv\Scripts\python.exe src/check_setup.py --ml
```

This is a default local setup. It does not configure GPU acceleration. For a hardware-specific PyTorch installation use the official selector: https://pytorch.org/get-started/locally/ . No GPU is needed to run this starter interface.

When installation works, optionally record exact installed versions:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > requirements-lock.txt
```

The requirements files list intended dependencies; they are not a fully tested version lock.

### Mac or Linux

Use `python3 -m venv .venv` to create the environment. Replace `.\.venv\Scripts\python.exe` with `.venv/bin/python` in each command above.

## What each part does

| Path | Purpose |
| --- | --- |
| data/raw/ | Original downloaded audio and label CSVs, later |
| data/processed/ | Cleaned metadata and prepared data, later |
| notebooks/ | Experiments and listening to samples |
| src/prepare_data.py | Placeholder for linking audio to emotion labels |
| src/train.py | Placeholder for learning from labeled examples |
| src/evaluate.py | Placeholder for measuring performance on held-out audio |
| src/predict.py | Placeholder for using a trained model on new audio |
| src/check_setup.py | Checks Python dependencies and project folders |
| models/ | Saved trained models, later |
| app/app.py | Working upload/record/playback interface |
| requirements.txt | Packages needed for the starter interface |
| requirements-ml.txt | Additional packages for later ML work |

## Package roles

- Streamlit: the browser interface, written in Python.
- PyTorch: train neural networks.
- Transformers: access pretrained models.
- Datasets and huggingface-hub: access datasets and repository files.
- Librosa and SoundFile: read and process sound.
- pandas and NumPy: tables and numeric arrays.
- scikit-learn: baseline models and evaluation metrics.

## Troubleshooting

- `py` is not recognized: try `python --version`; otherwise install Python and reopen VS Code.
- Cannot find requirements.txt: open the folder that contains this README in VS Code.
- Microphone unavailable: allow microphone access or use a WAV upload.
- No predictions: expected at this stage; no model has been trained.
- Installation fails: keep the exact error and the output of `py --version` to diagnose it.

## Completion checklist

- [ ] Understand arousal and valence.
- [ ] Agree on the initial scope and dataset.
- [ ] Create the environment and install dependencies.
- [ ] Run the setup checker successfully.
- [ ] Open the app and upload or record playable audio.

Next: inspect actual CSV columns and audio paths, verify labels and class counts, and plan a leakage-resistant split before training.

## Verification of this starter

Python syntax and ZIP contents were checked during creation. Dependency installation and an interactive browser run were not performed in the creation environment. Run the setup checker and playback check on your machine.
