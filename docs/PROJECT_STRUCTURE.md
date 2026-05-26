# Project Structure

## Root Files

- `app.py`: Streamlit demo app using ConvNeXt-Tiny as the primary classifier and DenseNet121 as a safety override.
- `README.md`: GitHub-facing project overview, setup instructions, results summary, and disclaimer.
- `requirements.txt`: Minimal Python package list.
- `.gitignore`: Keeps raw data, checkpoints, and heavy generated artifacts out of Git.

## Source Files

- `src/audit_dataset.py`: Audits dataset folders, class distribution, corrupted images, duplicates, and image sizes.
- `src/create_manifest.py`: Creates cleaned train, validation, and test CSV manifests without modifying raw files.
- `src/dataset.py`: Loads ImageFolder or manifest-based datasets with configurable preprocessing.
- `src/model.py`: Builds supported timm models for 4-class classification.
- `src/train.py`: Trains the 4-class classifier and saves experiment-specific checkpoints.
- `src/evaluate.py`: Evaluates the 4-class classifier and exports medical-risk reports.
- `src/train_binary.py`: Trains the binary tumor-vs-notumor gate experiment.
- `src/evaluate_binary.py`: Evaluates the binary tumor-vs-notumor model.
- `src/evaluate_ensemble_safety.py`: Evaluates ensemble safety strategies across trained models.
- `src/run_experiment.py`: Runs one configured model/preprocessing experiment.
- `src/evaluate_hard_cases.py`: Evaluates trained experiments on the hard-case audit set only.

## Documentation

- `docs/RESULTS.md`: Concise model and safety evaluation results.
- `docs/SAFETY.md`: Safety notes, wording rules, and known limitations.
- `docs/PROJECT_STRUCTURE.md`: Repository layout and file responsibilities.

## Data and Outputs

- `data/raw/`: Optional local raw download location. Ignored by Git.
- `data/extracted/`: Extracted Kaggle dataset location. Ignored by Git.
- `data/processed/`: CSV manifests. These can be regenerated from the local dataset.
- `outputs/checkpoints/`: Model checkpoints. Ignored by Git.
- `outputs/reports/`: Generated evaluation reports. Small reports may be committed selectively, but they can also be regenerated.
- `outputs/gradcam/`: Heavy Grad-CAM artifacts. Ignored by Git.
- `outputs/high_risk_samples/`: Exported hard-case samples. Ignored by Git.
- `outputs/combined_gate_missed_samples/`: Exported combined-gate miss samples. Ignored by Git.

Raw data and checkpoints should not be committed. Heavy generated artifacts should remain local.
