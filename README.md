# Melanoma Classification (PyTorch)

Binary classification of dermoscopic skin-lesion images as **Benign** or **Malignant**,
built with PyTorch transfer learning. The repository covers the full loop: data
loading with a stratified split, training with mixed precision and class-imbalance
handling, threshold-aware evaluation, Grad-CAM explanations, ONNX export, and a small
Flask demo.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-0.17%2B-ee4c2c)](https://pytorch.org/vision/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000)](https://flask.palletsprojects.com/)
[![Tests](https://img.shields.io/badge/tests-pytest-0a9edc)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Medical disclaimer

**This project is a research and educational demonstration. It is not a medical device,
it has not been clinically validated, and it must never be used to make, support or
defer a medical decision.** The model is trained on a public dataset of curated
dermoscopic images that does not represent the distribution of lesions a clinician
sees in practice, and its outputs carry no diagnostic weight. If you are concerned
about a skin lesion, consult a qualified dermatologist. The same disclaimer is shown
on every page of the web demo and attached to every API response.

---

## Features

- **Four interchangeable backbones** - `resnet18`, `resnet50`, `efficientnet_b0`,
  `densenet121` - loaded through torchvision's current `weights=` enum API.
- **Stratified, seeded train/validation split** by index, so the class ratio is
  preserved and the split is reproducible across runs and machines. The `test/`
  folder is held out and touched only by `evaluate.py`.
- **Class-imbalance handling** with either a `WeightedRandomSampler` or an
  inverse-frequency weighted `CrossEntropyLoss` (`--balance {sampler,weights,none}`).
- **Mixed-precision training** (`torch.amp`) with gradient clipping, AdamW, linear
  warmup into cosine annealing, early stopping on validation macro-F1, best/last
  checkpointing, resume, and TensorBoard scalars.
- **`--dry-run` smoke test** that pushes two batches through the whole pipeline in
  seconds, so configuration errors surface before a long run.
- **Threshold-aware evaluation**: accuracy, precision, recall, F1, ROC-AUC, average
  precision, confusion matrix and a full classification report, plus two tuned
  operating points (see [Threshold tuning](#threshold-tuning)).
- **Grad-CAM from scratch** using forward and full-backward hooks - no third-party
  CAM dependency.
- **ONNX export** with a dynamic batch axis and an optional `onnxruntime` numerical
  check that degrades gracefully when the runtime is not installed.
- **Flask demo** with an application factory, Bootstrap 5 UI, Grad-CAM overlay,
  `POST /api/predict` and `GET /api/health`, that starts cleanly even before any
  model has been trained.

## Tech stack

| Layer | Choice |
| --- | --- |
| Deep learning | PyTorch 2.2+, torchvision 0.17+ (`transforms.v2`) |
| Metrics | scikit-learn |
| Plotting | matplotlib (Agg backend, headless safe) |
| Configuration | PyYAML dataclasses + environment overrides, python-dotenv |
| Experiment tracking | TensorBoard |
| Web demo | Flask 3, Jinja2, Bootstrap 5 (CDN) |
| Packaging | pip (`requirements.txt` and `pyproject.toml`) |
| Tests | pytest |

## Project structure

```
melanoma-classification-pytorch/
├── app/
│   ├── __init__.py              # Flask application factory
│   ├── routes.py                # UI routes + /api/predict, /api/health
│   ├── static/css/app.css
│   └── templates/
│       ├── base.html            # layout, nav, medical disclaimer banner
│       ├── index.html           # upload form
│       └── result.html          # prediction + Grad-CAM overlay
├── checkpoints/                 # best.pt / last.pt land here (git-ignored)
├── configs/
│   ├── default.yaml
│   ├── resnet50.yaml
│   └── efficientnet_b0.yaml
├── docs/images/                 # screenshots referenced by this README
├── reports/
│   └── figures/                 # confusion matrix, ROC, PR curve
├── runs/                        # TensorBoard event files (git-ignored)
├── samples/                     # 4 test-split images from the dataset, for smoke tests
│   ├── benign_01.jpg
│   ├── benign_02.jpg
│   ├── malignant_01.jpg
│   └── malignant_02.jpg
├── scripts/
│   └── download_data.py         # kagglehub / Kaggle CLI download, or import a local copy
├── src/
│   ├── __init__.py
│   ├── config.py                # dataclass config, YAML + env overrides
│   ├── dataset.py               # ImageFolder, stratified split, dataloaders
│   ├── transforms.py            # torchvision v2 train/eval pipelines
│   ├── model.py                 # build_model() and head replacement
│   ├── engine.py                # train_one_epoch / validate with AMP
│   ├── train.py                 # training CLI
│   ├── evaluate.py              # test metrics, figures, threshold tuning
│   ├── predict.py               # single image or directory inference
│   ├── gradcam.py               # Grad-CAM via hooks
│   ├── export_onnx.py           # ONNX export + onnxruntime verification
│   └── utils.py                 # seeding, device, checkpoints, AverageMeter
├── tests/
│   ├── conftest.py
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_engine.py
│   ├── test_app.py
│   └── test_download_data.py
├── run.py                       # development server entry point
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

## Prerequisites

- Python 3.10 or newer.
- pip 23+.
- Roughly 2 GB of disk space for the dataset and pretrained weights.
- A CUDA GPU is optional. Everything runs on CPU, just slower; Apple Silicon is
  supported through the MPS backend. Mixed precision is enabled only on CUDA.
- Internet access on the first run, to download the ImageNet weights. Use
  `--no-pretrained` to stay fully offline.

### Dataset

The project expects this layout under `DATA_DIR`:

```
DATA_DIR/
├── train/
│   ├── Benign/       *.jpg
│   └── Malignant/    *.jpg
└── test/
    ├── Benign/       *.jpg
    └── Malignant/    *.jpg
```

The reference dataset is
[`ailearner-researchlab/melanoma-skin-cancer-dataset-benign-vs-malignant`](https://www.kaggle.com/datasets/ailearner-researchlab/melanoma-skin-cancer-dataset-benign-vs-malignant)
(224x224 JPEGs: 11,879 training images - 6,289 Benign / 5,590 Malignant - and 2,000
test images, 1,000 per class). `scripts/download_data.py` gets it into place in one of
three ways:

```bash
# 1. kagglehub (default). Needs a Kaggle API token: ~/.kaggle/kaggle.json
#    (%USERPROFILE%\.kaggle\kaggle.json on Windows) or KAGGLE_USERNAME / KAGGLE_KEY.
python scripts/download_data.py                  # -> ../_datasets/melanoma_skin_cancer

# 2. The Kaggle CLI instead of kagglehub
pip install kaggle
python scripts/download_data.py --method cli
python scripts/download_data.py --print-command  # just show the kaggle command

# 3. A copy you already have: a folder (verified and used in place) or the .zip (extracted)
python scripts/download_data.py --local D:\sample_projects\_datasets\melanoma_skin_cancer
python scripts/download_data.py --local D:\Downloads\archive.zip --data-dir D:\sample_projects\_datasets\melanoma_skin_cancer
```

The destination is `--data-dir`, else `DATA_DIR`, else `../_datasets/melanoma_skin_cancer`
(relative to the repository root). With the repository cloned to
`D:\sample_projects\melanoma-classification-pytorch`, the default resolves to
`D:\sample_projects\_datasets\melanoma_skin_cancer`, so a dataset already at that path
needs no configuration at all. Otherwise set `DATA_DIR` in `.env`, for example
`DATA_DIR=D:\sample_projects\_datasets\melanoma_skin_cancer`, or pass `--data-dir` to
the train and evaluate scripts. The script prints the image count per folder and the
`DATA_DIR` value to use.

Any dataset that matches the folder layout above will work; class names are read from
the directory names, and `Benign` sorts before `Malignant` so index 1 is the positive
(malignant) class throughout.

## Installation

**Windows (PowerShell or cmd):**

```bat
git clone https://github.com/your-org/melanoma-classification-pytorch.git
cd melanoma-classification-pytorch
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

**macOS / Linux:**

```bash
git clone https://github.com/your-org/melanoma-classification-pytorch.git
cd melanoma-classification-pytorch
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

For a CUDA build of PyTorch, install torch and torchvision first from the official
index for your CUDA version (see [pytorch.org](https://pytorch.org/get-started/locally/)),
then run `pip install -r requirements.txt`.

Then edit `.env` so that `DATA_DIR` points at your dataset.

## Configuration

Values are resolved in this order, later sources winning: dataclass defaults ->
YAML file -> environment variables -> command-line flags.

| Environment variable | Description | Default |
| --- | --- | --- |
| `DATA_DIR` | Dataset root containing `train/` and `test/`. Relative paths resolve against the repository root. | `../_datasets/melanoma_skin_cancer` |
| `CHECKPOINT_PATH` | Checkpoint used by `evaluate`, `predict`, `gradcam`, `export_onnx` and the web app. | `checkpoints/best.pt` |
| `CONFIG_PATH` | YAML config loaded by `load_config()`. | `configs/default.yaml` |
| `DEVICE` | `auto`, `cuda`, `mps` or `cpu`. `auto` prefers CUDA, then MPS, then CPU. | `auto` |
| `NUM_WORKERS` | DataLoader worker processes. Set to `0` on Windows if you hit multiprocessing errors. | `4` |
| `RANDOM_SEED` | Seed for `random`, `numpy` and `torch`; also seeds the train/val split. | `42` |
| `FLASK_ENV` | `development` enables the Flask debugger and reloader. | `production` |
| `SECRET_KEY` | Flask session and flash-message key. Change it for any deployment. | `change-me-in-production` |
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. | `INFO` |
| `BATCH_SIZE` | Overrides `data.batch_size`. | from YAML (`32`) |
| `IMAGE_SIZE` | Overrides `data.image_size`. | from YAML (`224`) |
| `MODEL_NAME` | Overrides `model.name`. | from YAML (`resnet50`) |
| `EPOCHS` | Overrides `train.epochs`. | from YAML (`15`) |
| `LEARNING_RATE` | Overrides `train.lr`. | from YAML (`3e-4`) |
| `AMP` | `true`/`false`; mixed precision is applied only on CUDA regardless. | from YAML (`true`) |
| `DECISION_THRESHOLD` | Malignant probability at which the web demo flags an image. | `0.5` |

Three YAML presets ship in `configs/`: `default.yaml` (ResNet-50, 15 epochs, weighted
loss), `resnet50.yaml` (longer schedule, `WeightedRandomSampler`) and
`efficientnet_b0.yaml` (lighter and faster, larger batch).

### Windows and `num_workers`

Windows spawns rather than forks DataLoader workers. Every entry point in this
repository is guarded by `if __name__ == "__main__":`, so `num_workers > 0` is safe
from the command line. Inside a notebook, or if you see `BrokenPipeError` or a long
stall at the start of each epoch, pass `--num-workers 0` or set `NUM_WORKERS=0`.

## Usage

### Smoke test (no training)

```bash
python -m src.train --dry-run
```

Runs two training batches and two validation batches, logs the metrics and exits
without writing a checkpoint. This is the fastest way to confirm that `DATA_DIR`,
the transforms and the model all line up.

### Quick end-to-end run on a subset (CPU friendly)

`--limit N` trains on a stratified sample of N training images (validation is capped
at N/4) and `evaluate --limit N` scores N test images. This exercises the full loop -
checkpoints, metrics, figures, prediction, Grad-CAM and the web demo - in about a
minute on a laptop CPU:

```bash
python -m src.train --model resnet18 --epochs 1 --limit 256 --batch-size 16 --num-workers 0
python -m src.evaluate --limit 200 --num-workers 0
python -m src.predict --input samples/
python run.py
```

The resulting model is only a pipeline check, not a useful classifier; train on the
full dataset (ideally on a GPU) for real results.

### Training

```bash
python -m src.train                                    # configs/default.yaml
python -m src.train --config configs/efficientnet_b0.yaml
python -m src.train --model resnet50 --epochs 25 --balance sampler
python -m src.train --freeze-backbone --lr 1e-3        # linear probing
python -m src.train --resume checkpoints/last.pt
python -m src.train --no-pretrained --device cpu       # fully offline
```

| Flag | Description |
| --- | --- |
| `--config` | YAML config file. |
| `--data-dir` | Override the dataset root. |
| `--model` | `resnet18`, `resnet50`, `efficientnet_b0` or `densenet121`. |
| `--epochs`, `--batch-size`, `--lr`, `--weight-decay` | Optimisation overrides. |
| `--balance {sampler,weights,none}` | Class-imbalance strategy. |
| `--freeze-backbone` | Train only the classifier head. |
| `--no-pretrained` | Start from random weights. |
| `--no-amp` | Disable mixed precision. |
| `--num-workers`, `--device`, `--seed` | Runtime overrides. |
| `--resume` | Continue from a checkpoint (model, optimizer, scheduler, scaler). |
| `--checkpoint-dir`, `--log-dir`, `--patience` | Output and early-stopping controls. |
| `--limit N` | Train on a stratified subset of N images (smoke runs). |
| `--dry-run` | Two batches, then exit. |

Checkpoints are written to `checkpoints/best.pt` (best validation macro-F1) and
`checkpoints/last.pt` (most recent epoch). Follow training with:

```bash
tensorboard --logdir runs
```

### Evaluation

```bash
python -m src.evaluate --checkpoint checkpoints/best.pt
python -m src.evaluate --target-sensitivity 0.98 --no-figures
python -m src.evaluate --limit 200          # quick check on 200 test images
```

Writes `reports/metrics.json` plus `confusion_matrix.png`, `roc_curve.png` and
`pr_curve.png` under `reports/figures/`.

#### Threshold tuning

`evaluate.py` reports three operating points on the held-out test split:

| Operating point | Definition | Why it is there |
| --- | --- | --- |
| `default` | Fixed threshold of 0.50. | The naive argmax decision; a reference, not a recommendation. |
| `best_f1` | Threshold maximising F1 on the Malignant class. | The point a balanced-accuracy view would pick. |
| `target_sensitivity` | Highest threshold whose recall on Malignant is still at or above `eval.target_sensitivity` (0.95 by default). | The clinically meaningful one. |

In a screening context the two error types are not symmetric: a false positive costs
a patient an unnecessary review, while a false negative is a melanoma sent home. A
model tuned to maximise accuracy or F1 will happily trade recall for precision,
because both metrics treat the two errors alike. Fixing the sensitivity first and
reading off the resulting precision and specificity states the trade-off in the terms
that actually matter, and makes it obvious when a model cannot reach the required
recall at any threshold - in which case `evaluate.py` records
`"achieved": false` rather than quietly reporting the closest point.

> **No performance numbers are quoted anywhere in this README.** All metrics are
> placeholders until you run `python -m src.evaluate`, which writes the real values
> for your training run to `reports/metrics.json`.

The JSON looks like this (values shown are illustrative placeholders, **not**
measured results):

```jsonc
{
  "model_name": "resnet50",
  "num_test_samples": 0,
  "metrics": { "accuracy": null, "roc_auc": null, "average_precision": null, "macro_f1": null },
  "confusion_matrix": [[0, 0], [0, 0]],
  "operating_points": {
    "default":            { "threshold": 0.5,  "recall": null, "precision": null, "specificity": null },
    "best_f1":            { "threshold": null, "recall": null, "precision": null, "specificity": null },
    "target_sensitivity": { "threshold": null, "target_recall": 0.95, "achieved": null }
  }
}
```

### Prediction

```bash
python -m src.predict --input samples/malignant_01.jpg
python -m src.predict --input samples/ --csv reports/predictions.csv
python -m src.predict --input D:/images --recursive --threshold 0.35
```

The four images in `samples/` are copied unmodified from the dataset's `test/` split
(two per class) and remain under the dataset's own terms.

Prints one line per image and optionally writes a CSV with `path`, `filename`,
`predicted_class`, `confidence`, `malignant_probability` and `threshold`.

### Grad-CAM

```bash
python -m src.gradcam --input samples/malignant_01.jpg --output docs/images/cam.png
python -m src.gradcam --input samples/ --output reports/figures
python -m src.gradcam --input samples/benign_01.jpg --class-index 1 --alpha 0.6
```

Hooks the last convolutional layer, weights its feature maps by the spatially averaged
gradient of the target logit, and blends the ReLU'd result over the original image.
`--class-index 1` always explains the malignant logit, which is usually what you want
even when the prediction is benign.

### ONNX export

```bash
python -m src.export_onnx --checkpoint checkpoints/best.pt --output checkpoints/model.onnx
python -m src.export_onnx --opset 18 --no-verify
```

The graph takes `input: (batch_size, 3, H, W)` (ImageNet-normalised floats) and returns
`logits: (batch_size, num_classes)`; apply softmax downstream. If `onnxruntime` is
installed, the export is checked numerically against the PyTorch model; if it is not,
the export still succeeds and a note is logged.

### Web demo

```bash
python run.py
python run.py --host 0.0.0.0 --port 8080 --debug
```

Then open <http://127.0.0.1:5000>. The demo starts even with no checkpoint present:
the form is disabled, `/api/health` reports `"model_loaded": false`, and
`/api/predict` returns 503 with an instruction to train a model first.

`run.py` is the Werkzeug development server. For anything else, serve the factory:

```bash
waitress-serve --port 8000 --call app:create_app     # Windows friendly
gunicorn "app:create_app()" --bind 0.0.0.0:8000      # Linux / macOS
```

## API reference

| Method | Path | Request | Success | Failure |
| --- | --- | --- | --- | --- |
| `GET` | `/` | - | `200` upload page | - |
| `POST` | `/predict` | `multipart/form-data`, field `image` | `200` result page with Grad-CAM overlay | redirect to `/` with a flash message |
| `POST` | `/api/predict` | `multipart/form-data`, field `image` | `200` JSON | `400` missing or undecodable file, `413` over 8 MB, `415` unsupported extension, `503` no checkpoint |
| `GET` | `/api/health` | - | `200` JSON with `status`, `model_loaded`, `checkpoint_path`, `classes`, `device` | - |

```bash
curl -F "image=@samples/malignant_01.jpg" http://127.0.0.1:5000/api/predict
curl http://127.0.0.1:5000/api/health
```

A successful `/api/predict` response:

```json
{
  "predicted_class": "Malignant",
  "confidence": 0.0,
  "threshold": 0.5,
  "probabilities": { "Benign": 0.0, "Malignant": 0.0 },
  "malignant_probability": 0.0,
  "filename": "malignant_01.jpg",
  "disclaimer": "Research demo only. This output is not a diagnosis and must not be used for clinical decisions."
}
```

(The numbers above are structural placeholders showing the response shape.)

## Screenshots

Screenshots are not committed. Drop your own into `docs/images/` and they will render
here:

- `docs/images/app-upload.png` - the upload form
- `docs/images/app-result.png` - prediction with the Grad-CAM overlay
- `docs/images/confusion-matrix.png` - copied from `reports/figures/` after evaluation
- `docs/images/tensorboard.png` - training curves

```markdown
![Upload form](docs/images/app-upload.png)
![Prediction result](docs/images/app-result.png)
```

## Testing

```bash
pip install pytest
pytest                      # whole suite
pytest -v tests/test_model.py
pytest -k stratified
```

The suite runs on CPU in seconds and needs no dataset, no GPU and no trained
checkpoint. It covers:

- transform output shape, dtype and determinism, and the denormalisation round trip;
- the stratified split preserving class ratios, being reproducible under a fixed seed,
  changing under a different seed, and rejecting invalid fractions;
- class weights and the weighted sampler on an imbalanced label list;
- `build_model` producing the correct head width for all four architectures, plus
  freezing, the Grad-CAM target-layer lookup and the error paths;
- a training step on a synthetic two-sample tensor dataset actually updating weights,
  `max_batches` stopping early, and validation leaving parameters untouched;
- Grad-CAM producing a normalised heatmap for the ResNet, EfficientNet and DenseNet
  families;
- the Flask app: `/api/health`, the missing-checkpoint path on both the HTML and JSON
  routes, and upload validation;
- `scripts/download_data.py` with a local folder, a local `.zip`, and a mocked
  kagglehub download.

Model tests always pass `pretrained=False`, so no weights are downloaded during a
test run.

## Roadmap

- [ ] Test-time augmentation (horizontal/vertical flip averaging) in `evaluate.py`.
- [ ] Stratified k-fold cross-validation with ensembled predictions.
- [ ] Calibration: reliability diagrams and temperature scaling, so the reported
      probabilities can be read as probabilities.
- [ ] Bootstrap confidence intervals on the reported metrics.
- [ ] A batch endpoint for the API, and a Dockerfile for the demo.
- [ ] Hair and ruler-mark removal as an optional preprocessing step.
- [ ] Metadata fusion (age, site, sex) where the dataset provides it.

## Limitations

- **Not a diagnostic tool.** See the disclaimer at the top. No clinical validation of
  any kind has been performed.
- **Dataset shift.** The reference dataset is curated, cropped and resized to 224x224.
  Phone photographs, different dermatoscopes, other skin tones and other lesion types
  are all out of distribution, and performance on them is unknown.
- **Skin-tone representation.** Public dermoscopic datasets are heavily skewed towards
  lighter skin. Any model trained here is likely to perform worse on darker skin, and
  this repository does not measure that gap.
- **Two classes only.** Real dermatology involves many lesion types; collapsing them
  into benign/malignant discards clinically important distinctions.
- **Grad-CAM explains the model, not the biology.** A plausible-looking heatmap is not
  evidence that the model learned a dermatologically meaningful feature; CAMs are
  low-resolution and known to be sensitive to the chosen layer.
- **Probabilities are uncalibrated.** A softmax output of 0.9 does not mean a 90%
  chance of malignancy. Calibration is on the roadmap.
- **Single-split evaluation.** Metrics come from one train/val split and one test
  folder, with no confidence intervals and no repeated runs.
- **The demo server is not hardened.** `run.py` is the development server, uploads are
  held in memory, and there is no authentication or rate limiting.

## License

Released under the MIT License. See [LICENSE](LICENSE).

The dataset is distributed under its own terms on Kaggle; review them before use. The
pretrained ImageNet weights are subject to torchvision's licensing.
