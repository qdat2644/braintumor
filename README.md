# Brain MRI Classification AI Demo

## Overview

This repository contains a research/demo brain MRI image classification project. The model predicts one of four classes:

- glioma
- meningioma
- pituitary
- notumor

The project is built with PyTorch, timm, and Streamlit. It is intended for machine learning experimentation and portfolio demonstration only. It is not a medical diagnosis system.

## Key Features

- Dataset audit pipeline for class counts, corrupted images, image sizes, and duplicate hashes.
- Manifest-based duplicate handling without deleting or modifying raw images.
- EfficientNet-B0 baseline training pipeline.
- DenseNet121 and ConvNeXt-Tiny experiment support.
- Medical-risk evaluation focused on tumor-to-notumor errors.
- Grad-CAM hard-case debugging utilities.
- Binary tumor-vs-notumor safety gate experiment.
- Ensemble safety override using ConvNeXt-Tiny and DenseNet121.
- Streamlit demo UI with safety-first wording.

## Demo App

The demo app is implemented in `app.py`.

The selected demo logic uses:

- Primary model: ConvNeXt-Tiny
- Safety override model: DenseNet121

If ConvNeXt-Tiny predicts `notumor` but DenseNet121 predicts a tumor class, the UI displays an uncertain result and recommends medical review. The app avoids definitive medical wording and does not claim that a patient has or does not have a tumor.

## Dataset

The raw dataset is not included in this repository due to size and licensing constraints.

Download the dataset manually from Kaggle and place it locally as:

```text
data/extracted/Training
data/extracted/Testing
```

The expected class folders are:

```text
glioma
meningioma
notumor
pituitary
```

## Project Structure

```text
brain-tumor/
├── app.py
├── src/
├── docs/
├── outputs/
├── data/
├── requirements.txt
├── README.md
└── .gitignore
```

`data/` is ignored because raw datasets should not be committed. Model checkpoints and heavy generated artifacts are also ignored.

## Environment Setup

On Windows with Conda:

```powershell
conda create -n dat python=3.10 -y
conda activate dat
```

Install PyTorch with CUDA using the official PyTorch install selector for your GPU and CUDA version:

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Do not install a CPU-only PyTorch build if you intend to use GPU acceleration.

Then install the remaining dependencies:

```powershell
pip install -r requirements.txt
```

## How to Run the Demo

```powershell
streamlit run app.py
```

The demo expects the selected model checkpoints to exist locally under `outputs/checkpoints/`.

## How to Reproduce the Pipeline

Run the main data and baseline pipeline:

```powershell
python src\audit_dataset.py
python src\create_manifest.py
python src\train.py
python src\evaluate.py
```

Run an example ConvNeXt-Tiny experiment:

```powershell
python src\run_experiment.py --model-name convnext_tiny --image-size 224 --pad-to-square --no-horizontal-flip --experiment-name convnext_tiny_pad224_nohflip
```

## Results Summary

| Model | Accuracy | Macro F1 | Glioma Recall | Tumor-to-Notumor | Notes |
|---|---:|---:|---:|---:|---|
| EfficientNet-B0 manifest baseline | 0.9520 | 0.9505 | 0.8238 | 22 | Baseline |
| DenseNet121 pad224 no horizontal flip | 0.9545 | 0.9532 | 0.8316 | 21 | Lower tumor-to-notumor count |
| ConvNeXt-Tiny pad224 no horizontal flip | 0.9609 | 0.9598 | 0.8627 | 24 | Best overall metrics |

Ensemble safety evaluation:

- Caught 14/22 original tumor-to-notumor cases.
- Still missed 8/22.
- Flagged 2 notumor samples unnecessarily.

## Known Limitations

- This project is not clinically validated.
- Some glioma cases are still predicted as no-tumor-like.
- Known hard cases include:
  - `Te-gl_143`
  - `Te-gl_341`
  - `Te-gl_74`
- A no-tumor-like output does not confirm absence of disease.
- The dataset is limited and may not generalize to external hospitals, scanners, acquisition protocols, or patient populations.

## Medical Disclaimer

This project is for research and educational demo purposes only. It is not a diagnostic tool. MRI images must be reviewed by a qualified radiologist or medical professional.

## Future Work

- More robust medical image preprocessing.
- External validation on independent datasets.
- Better probability calibration.
- More robust uncertainty estimation.
- Improved handling of hard glioma cases.
- Clinical expert review of model errors and Grad-CAM outputs.
