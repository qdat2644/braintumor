# Usage

## 1. Create Environment

```powershell
conda create -n dat python=3.10 -y
conda activate dat
```

Install PyTorch from the official selector for your CUDA setup:

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Then install project dependencies:

```powershell
pip install -r requirements.txt
```

## 2. Prepare Dataset

Download the Kaggle dataset manually and extract it to:

```text
data/extracted/Training
data/extracted/Testing
```

Raw data is intentionally ignored by Git.

## 3. Audit and Build Manifests

```powershell
python src\audit_dataset.py
python src\create_manifest.py
```

## 4. Train and Evaluate

```powershell
python src\train.py
python src\evaluate.py
```

## 5. Run an Experiment

```powershell
python src\run_experiment.py --model-name convnext_tiny --image-size 224 --pad-to-square --no-horizontal-flip --experiment-name convnext_tiny_pad224_nohflip
```

## 6. Run Streamlit Demo

```powershell
streamlit run app.py
```

The demo requires local checkpoints under `outputs/checkpoints/`.
