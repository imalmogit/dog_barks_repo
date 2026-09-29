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

## Why 240 clips instead of exactly 150?

The 1,000 official training clips contain **429 Positive**, **340 Neutral**, and **231 Negative** labels; after conversion this is **429 happy** and **571 not_happy**. A random 150-clip subset would preserve that imbalance (about 64 happy / 86 not-happy). A deliberately balanced 150 subset is possible (75/75), but a 70/15/15 split produces only about 22–23 clips per validation/test class—too unstable for a meaningful baseline.

This project therefore uses **240 clips**, sampled reproducibly with seed 42: 120 happy and 120 not-happy. It yields a balanced 168/36/36 train/validation/test split (84/84, 18/18, 18/18). This is still lightweight, but makes the held-out scores less noisy.

The public label CSVs do not include a dog identity. Consequently, this script cannot enforce dog-disjoint splitting and does not claim generalisation to new individual dogs. It does keep a frozen test CSV that `train.py` never reads. The original dataset paper reports dog-level splits for its release, but this small random prototype should be described as clip-disjoint only unless identity metadata is later obtained.

## Run it

From the project folder, create/activate an environment and install the ML dependencies:

```bash
python -m pip install -r requirements-ml.txt
python src/prepare_data.py --subset-size 240 --seed 42
python src/train.py
python -m streamlit run app/app.py
```

Use the validation macro-F1 to make one small modelling decision if needed. Only after the approach is fixed, run this once:

```bash
python src/evaluate.py
```

It writes the held-out macro-F1 and a confusion matrix into `models/`. Keep the test score out of model selection.

## Model choice

The baseline uses 20 MFCC acoustic features (their mean and variation over the clip) and logistic regression. It is intentionally simple, fast on a CPU, and appropriate for a first 240-clip study. A pretrained audio embedding model may be a useful later comparison, but should not replace this interpretable baseline before it is measured.

## Limits to state in a report

- The labels are human-provided valence categories, not direct access to a dog's inner state.
- Training data covers Husky and Shiba Inu clips; it is not evidence of broad breed generalisation.
- The interface assumes a clear bark; it does not yet detect barks or reject unrelated environmental audio.
- A confidence score shows the model's output distribution, not clinical certainty.
