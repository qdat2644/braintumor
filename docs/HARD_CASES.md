# Hard Case Analysis

## Overview

A focused hard-case set was identified during safety evaluation. These samples are useful for auditing model behavior, but they must not be used as training data.

The hard-case analysis focuses on tumor images that were predicted as no-tumor-like by the selected model and safety setup.

## Known Hard Cases

Known examples:

- `Te-gl_143`
- `Te-gl_341`
- `Te-gl_74`

These are glioma samples that were still predicted as no-tumor-like by the selected model/safety setup.

## Why This Matters

False negatives are the highest-risk failure mode for this project. A no-tumor-like prediction does not confirm absence of disease.

Overall accuracy can hide dangerous class-specific failures. A model can have strong aggregate metrics while still missing clinically important tumor-like patterns.

## What Was Tried

The following evaluation and mitigation steps were implemented:

- EfficientNet-B0 baseline
- Manifest cleaning
- Medical-risk report
- Grad-CAM
- Binary tumor-vs-notumor gate
- DenseNet121
- ConvNeXt-Tiny
- Ensemble safety evaluation

## Current Status

ConvNeXt-Tiny had the best overall metrics among the tested 4-class models.

Ensemble safety reduced some tumor-to-notumor errors, but some hard cases still remain. This is not clinical validation, and the model must not be treated as a diagnostic system.

## Future Work

- MRI-specific preprocessing
- External validation dataset
- Expert review
- Calibration and uncertainty estimation
- More robust hard-case handling
