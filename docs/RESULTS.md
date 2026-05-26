# Results Summary

## 1. Dataset Audit

- Training: 5600 original images.
- Testing: 1600 original images.
- Corrupted images: 0.
- Cross-split duplicates: 0.
- Internal duplicates existed and were handled through manifests.

## 2. Manifest Cleaning

- Cleaned train, validation, and test manifests were created.
- Raw dataset files were not modified.
- Train manifest: 4615 images.
- Validation manifest: 814 images.
- Test manifest: 1584 images.

## 3. Model Comparison

| Model | Accuracy | Macro F1 | Glioma Recall | Tumor→Notumor | Notes |
|---|---:|---:|---:|---:|---|
| EfficientNet-B0 | 0.9520 | 0.9505 | 0.8238 | 22 | Baseline |
| DenseNet121 | 0.9545 | 0.9532 | 0.8316 | 21 | Lower tumor→notumor count |
| ConvNeXt-Tiny | 0.9609 | 0.9598 | 0.8627 | 24 | Best overall metrics |

## 4. Hard-Case Evaluation

- 16 hard cases were identified where both the 4-class model and binary tumor-vs-notumor gate failed.
- All 16 hard cases were glioma.
- ConvNeXt-Tiny recovered 8/16.
- DenseNet121 recovered 5/16.
- Some hard cases remained predicted as no-tumor-like.

## 5. Ensemble Safety

- Ensemble safety evaluation caught 14/22 original tumor-to-notumor errors.
- It still missed 8/22.
- It flagged 2 notumor samples unnecessarily.
- `Te-gl_143`, `Te-gl_341`, and `Te-gl_74` were still not caught.

## 6. Interpretation

Overall accuracy is not enough for a medical AI prototype. The highest-risk failure mode is a tumor image being predicted as no-tumor-like. This project therefore emphasizes safety-aware evaluation, hard-case debugging, and conservative UI wording rather than clinical deployment.
