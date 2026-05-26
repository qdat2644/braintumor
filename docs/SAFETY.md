# Safety Notes

## 1. Not a Medical Device

This project is not clinically validated and must not be used for diagnosis. It is a research and educational demo only.

## 2. Why No-Tumor-Like Output Is Risky

A no-tumor-like prediction does not confirm absence of disease. False negatives are especially dangerous because a tumor image may be incorrectly treated as low risk by the model.

## 3. Known High-Risk Failure Mode

Glioma images can be predicted as no-tumor-like. Known examples include:

- `Te-gl_143`
- `Te-gl_341`
- `Te-gl_74`

## 4. Safety-Oriented UI Wording

Allowed wording:

- "AI classification result"
- "No-tumor-like pattern"
- "Medical review recommended"
- "Research/demo purposes only"

Forbidden wording:

- "Diagnosis"
- "Cancer detected"
- "No tumor confirmed"
- "You are healthy"
- "Bạn bị u não"
- "Chẩn đoán"
- "Ung thư"
- "Không có u"
- "Không phát hiện bệnh"
- "Bạn khỏe mạnh"

## 5. Ensemble Safety Logic

The Streamlit demo uses ConvNeXt-Tiny as the primary model and DenseNet121 as a safety override. If the primary model predicts `notumor` while DenseNet121 predicts a tumor class, the app displays uncertain/review wording instead of a decisive no-tumor-like result.

## 6. Remaining Limitation

The ensemble still misses some hard cases. All model outputs must be reviewed by qualified medical professionals before any real-world medical interpretation.
