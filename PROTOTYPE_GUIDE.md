# Happy Bark? — first prototype

## What this version does

This is an experimental **audio** classifier.  A user uploads or records a short bark, presses one button, and receives:

- **Yes**: the model classifies the bark as likely happy.
- **No**: the model does not classify the bark as likely happy.

The optional percentage is the model's confidence score for its chosen class. It is not a probability that the dog truly feels that emotion. This project is not veterinary or medical advice.

## Where the labels come from

Dataset: `ArlingtonCL2/BarkopediaDogEmotionClassification_Data`.

Do **not** use the generic `label` field shown by the Hugging Face viewer: it represents the audio-folder/breed label rather than emotion. The source of truth is the two official CSV files supplied alongside the audio:

| Audio clips | Emotion-label file | Required columns |
| --- | --- | --- |
| `train/husky/*.wav` | `husky_train_labels.csv` | `audio_id`, `arousal`, `valence` |
| `train/shiba/*.wav` | `shiba_train_labels.csv` | `audio_id`, `arousal`, `valence` |

`prepare_data.py` checks those columns, finds each WAV by `audio_id`, and maps only the official `valence` value:

| Original valence | Binary label |
| --- | --- |
| Positive | `happy` |
| Neutral or Negative | `not_happy` |

The Hugging Face release documents Positive/Neutral/Negative valence and Low/Medium/High arousal. The related official EmotionalCanines repository hosts the same label-file scheme.

## Why use all 1,000 training clips?

The 1,000 official training clips contain **429 Positive**, **340 Neutral**, and **231 Negative** labels; after conversion this is **429 happy** and **571 not_happy**. A random 150-clip subset would preserve that imbalance (about 64 happy / 86 not-happy). A deliberately balanced 150 subset is possible (75/75), but a 70/15/15 split produces only about 22–23 clips per validation/test class—too unstable for a meaningful baseline.

This improved version uses **all 1,000 official training clips**. It keeps their natural 429/571 binary distribution and uses class-weighted models to reduce the effect of that imbalance. The fixed seed creates a stratified 70/15/15 train/validation/test split (700/150/150), which gives a far more stable validation and test result than 150 or 240 clips.

The public label CSVs do not include a dog identity. Consequently, this script cannot enforce dog-disjoint splitting and does not claim generalisation to new individual dogs. It does keep a frozen test CSV that `train.py` never reads. The original dataset paper reports dog-level splits for its release, but this small random prototype should be described as clip-disjoint only unless identity metadata is later obtained.

## Run it

From the project folder, create/activate an environment and install the ML dependencies:

```bash
python -m pip install -r requirements-ml.txt
python src/prepare_data.py --seed 42
python src/train.py
python -m streamlit run app/app.py
```

Use the validation macro-F1 to make one small modelling decision if needed. Only after the approach is fixed, run this once:

```bash
python src/evaluate.py
```

It writes the held-out macro-F1 and a confusion matrix into `models/`. Keep the test score out of model selection.

## Model choice

The improved CPU baseline uses MFCCs, their time variation, chroma, and spectral/energy features. It compares class-weighted logistic regression with an RBF support-vector machine on the validation split, then saves the better one. This makes a stronger baseline, but does not guarantee any target score; the held-out test set remains the final check.

## Limits to state in a report

- The labels are human-provided valence categories, not direct access to a dog's inner state.
- Training data covers Husky and Shiba Inu clips; it is not evidence of broad breed generalisation.
- The interface assumes a clear bark; it does not yet detect barks or reject unrelated environmental audio.
- A confidence score shows the model's output distribution, not clinical certainty.

## Next experiment: Neutral versus Negative

This separate classifier distinguishes the two original valence categories grouped
as `not_happy`. It excludes Positive clips using ground-truth `valence`, rather
than predictions from the happy classifier. It does not change the existing app,
binary classifier, saved binary results, or split CSVs.

Use the existing prepared data. **Do not rerun `prepare_data.py` or the binary
training/evaluation commands for this experiment.** The new scripts filter rows
in memory without resampling or moving clips between splits.

| Split | Neutral | Negative | Total used |
| --- | --- | --- | --- |
| Train | 240 | 160 | 400 |
| Validation | 52 | 33 | 85 |
| Test (reserved) | 48 | 38 | 86 |

Run training from the project folder with the ML dependencies already installed:

```powershell
.\.venv\Scripts\python.exe -B src/train_neutral_negative.py
```

`src/train_neutral_negative.py` imports the unchanged feature extraction from
`src/train.py`: 16 kHz mono, up to the first 10 seconds, and 116 features formed
from MFCC, delta-MFCC, chroma, and spectral/energy means and standard deviations.
It compares class-weighted logistic regression with an RBF SVM using validation
macro-F1. Exact ties select logistic regression. Models are not refitted on
validation data.

The SVM uses `SVC(probability=False)` inside `CalibratedClassifierCV`, with sigmoid
calibration and five shuffled stratified folds drawn only from training data
(seed 42, `ensemble=True`). Scaling is inside the calibrated pipeline, so each
fold fits its scaler on that fold's training rows. Validation is used only for
model comparison, not calibration. The saved calibrated ensemble is the same
model evaluated on validation.

A `DummyClassifier(strategy="most_frequent")` learns its constant prediction
from the training labels. Reports include its validation macro-F1 alongside both
candidate scores, per-class precision/recall/F1/support, and confusion matrices
with explicit class order `[Neutral, Negative]`.

All new artifacts go under `models/neutral_negative/`:

- `neutral_negative_baseline.joblib`: selected fitted classifier.
- `validation_metrics.json`: class counts, majority baseline, both candidate
  reports/scores, selected model, and feature/calibration details.
- `validation_confusion_matrix.png`: selected model's validation confusion matrix.

Predict on a clip assumed to have non-positive valence:

```powershell
.\.venv\Scripts\python.exe -B src/predict_neutral_negative.py path/to/bark.wav
```

This classifier always chooses Neutral or Negative, including for a happy clip
or unrelated sound. Its score is conditional on these two classes and does not
measure the dog's true emotional state. Prediction uses the model's `predict`
decision and reports the probability score for that same class.

**Test evaluation must wait until the validation result has been reviewed and
the model choices frozen.** At that later stage only, run:

```powershell
.\.venv\Scripts\python.exe -B src/evaluate_neutral_negative.py
```

That script writes `test_metrics.json` and `test_confusion_matrix.png` within
`models/neutral_negative/`, including the training-derived majority baseline.
Training never reads `test.csv`. The existing files directly under `models/`
remain the happy/not_happy baseline artifacts. Both sets of artifacts are ignored
by Git, so preserve their local copies separately.

This experiment measures Neutral/Negative separation on known non-positive
clips. Combining it with the happy classifier would require separate three-class
evaluation of the full two-stage system, including first-stage errors. The
existing splits are clip-disjoint; dog-disjoint generalization is unverified.
