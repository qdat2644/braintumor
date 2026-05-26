from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import timm
import torch
from PIL import Image, ImageOps
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

CONVNEXT_CHECKPOINT = CHECKPOINT_DIR / "best_model_convnext_tiny_pad224_nohflip.pth"
DENSENET_CHECKPOINT = CHECKPOINT_DIR / "best_model_densenet121_pad224_nohflip.pth"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
TUMOR_CLASSES = {"glioma", "meningioma", "pituitary"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DISCLAIMER = (
    "Kết quả chỉ phục vụ mục đích nghiên cứu/demo, không phải chẩn đoán y khoa. "
    "Vui lòng tham khảo ý kiến bác sĩ hoặc chuyên gia chẩn đoán hình ảnh."
)
SHORT_NOTE = "Kết quả này chỉ phục vụ mục đích nghiên cứu/demo, không phải chẩn đoán y khoa."

CLASS_DISPLAY_NAMES = {
    "glioma": "Nghi ngờ u thần kinh đệm",
    "meningioma": "Nghi ngờ u màng não",
    "pituitary": "Nghi ngờ u tuyến yên",
    "notumor": "Dạng giống không thấy u",
}


@dataclass(frozen=True)
class Prediction:
    predicted_class: str
    confidence: float
    probabilities: list[float]


class PadToSquare:
    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        right = side - width - left
        bottom = side - height - top
        return ImageOps.expand(image, border=(left, top, right, bottom), fill=0)


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            PadToSquare(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def class_text(class_name: str) -> str:
    if class_name == "uncertain_tumor_review_recommended":
        return "Mẫu ảnh chưa đủ chắc chắn. Khuyến nghị bác sĩ/chuyên gia chẩn đoán hình ảnh xem xét lại."
    return CLASS_DISPLAY_NAMES[class_name]


def final_output_message(safe_output: str) -> str:
    if safe_output == "uncertain_tumor_review_recommended":
        return "Mẫu ảnh chưa đủ chắc chắn. Khuyến nghị bác sĩ/chuyên gia chẩn đoán hình ảnh xem xét lại."
    if safe_output == "notumor":
        return "Dạng giống không thấy u trên ảnh. Kết quả này không xác nhận chắc chắn là không có bệnh."
    return class_text(safe_output)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"] {
            background: #0b0f17;
            color: #e5e7eb;
        }
        [data-testid="stHeader"] {
            background: rgba(11, 15, 23, 0);
        }
        .block-container {
            padding-top: 0.7rem;
            padding-bottom: 1rem;
            max-width: 1240px;
            margin: 0 auto;
        }
        h1 {
            color: #f9fafb;
            font-size: 32px !important;
            line-height: 1.12 !important;
            margin: 0 0 0.12rem 0 !important;
        }
        h2, h3 {
            color: #f9fafb;
            margin-top: 0.1rem !important;
            margin-bottom: 0.45rem !important;
        }
        .subtitle {
            color: #9ca3af;
            font-size: 14px;
            margin-bottom: 0.55rem;
        }
        .top-disclaimer {
            background: #151b2a;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            color: #d1d5db;
            font-size: 13px;
            line-height: 1.35;
            margin-bottom: 0.85rem;
            padding: 10px 12px;
        }
        .dashboard-card, .safe-output-card {
            background: #121826;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            margin-bottom: 12px;
            padding: 14px;
        }
        .safe-output-card {
            background: #151b2a;
            border-color: rgba(148, 163, 184, 0.22);
        }
        .card-title {
            color: #9ca3af;
            font-size: 13px;
            font-weight: 650;
            margin-bottom: 6px;
        }
        .safe-result {
            color: #f9fafb;
            font-size: 19px;
            font-weight: 750;
            line-height: 1.35;
            margin-bottom: 6px;
        }
        .small-muted {
            color: #9ca3af;
            font-size: 13px;
            line-height: 1.35;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px 12px;
        }
        .metric-label {
            color: #9ca3af;
            font-size: 12px;
            margin-bottom: 2px;
        }
        .metric-value {
            color: #f3f4f6;
            font-size: 14px;
            font-weight: 650;
            line-height: 1.3;
        }
        div[data-testid="stImage"] img {
            max-height: 420px;
            object-fit: contain;
            width: 100%;
        }
        div[data-testid="stImage"] {
            background: #121826;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 8px;
        }
        .stFileUploader {
            margin-bottom: 0.45rem;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            overflow: hidden;
        }
        div[data-testid="stAlert"] {
            margin-bottom: 0.5rem;
            padding: 0.55rem 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model(checkpoint_path: str, model_name: str) -> torch.nn.Module:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint}")

    model = timm.create_model(model_name, pretrained=False, num_classes=len(CLASS_NAMES))
    state = torch.load(checkpoint, map_location="cpu")
    class_names = state.get("class_names", CLASS_NAMES)
    if class_names != CLASS_NAMES:
        raise ValueError(f"Thứ tự lớp trong checkpoint không đúng: {class_names}")

    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


@st.cache_resource
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model: torch.nn.Module, image: Image.Image, device: torch.device) -> Prediction:
    tensor = build_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    model = model.to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0).cpu()

    confidence, predicted_idx = probabilities.max(dim=0)
    return Prediction(
        predicted_class=CLASS_NAMES[predicted_idx.item()],
        confidence=confidence.item(),
        probabilities=probabilities.tolist(),
    )


def final_safe_output(primary: Prediction, safety: Prediction) -> str:
    if primary.predicted_class == "notumor" and safety.predicted_class in TUMOR_CLASSES:
        return "uncertain_tumor_review_recommended"
    return primary.predicted_class


def probability_table(primary: Prediction) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Nhóm": [CLASS_DISPLAY_NAMES[name] for name in CLASS_NAMES],
            "Xác suất": primary.probabilities,
        }
    )


def render_probability_chart(table: pd.DataFrame) -> None:
    chart_data = table.sort_values("Xác suất", ascending=True)
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    fig.patch.set_facecolor("#121826")
    ax.set_facecolor("#121826")
    ax.barh(chart_data["Nhóm"], chart_data["Xác suất"], color="#60a5fa")
    ax.set_xlim(0, 1)
    ax.tick_params(colors="#e5e7eb", labelsize=8)
    ax.xaxis.label.set_color("#9ca3af")
    ax.yaxis.label.set_color("#9ca3af")
    ax.grid(axis="x", color="white", alpha=0.08)
    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.08))
    ax.set_xlabel("Xác suất")
    fig.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_result_panel(primary: Prediction, safety: Prediction, safe_output: str) -> None:
    st.markdown("### 2. Kết quả AI")
    st.markdown(
        f"""
        <div class="safe-output-card">
            <div class="card-title">Kết luận hiển thị an toàn</div>
            <div class="safe-result">{escape(final_output_message(safe_output))}</div>
            <div class="small-muted">{escape(SHORT_NOTE)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="dashboard-card">
            <div class="card-title">Tóm tắt mô hình</div>
            <div class="metric-grid">
                <div>
                    <div class="metric-label">Mô hình chính</div>
                    <div class="metric-value">ConvNeXt-Tiny</div>
                </div>
                <div>
                    <div class="metric-label">Kết quả chính</div>
                    <div class="metric-value">{escape(class_text(primary.predicted_class))}</div>
                </div>
                <div>
                    <div class="metric-label">Độ tin cậy</div>
                    <div class="metric-value">{primary.confidence:.4f}</div>
                </div>
                <div>
                    <div class="metric-label">Mô hình đối chứng</div>
                    <div class="metric-value">DenseNet121</div>
                </div>
                <div>
                    <div class="metric-label">Kết quả kiểm tra an toàn</div>
                    <div class="metric-value">{escape(class_text(safety.predicted_class))}</div>
                </div>
                <div>
                    <div class="metric-label">Kết luận hiển thị an toàn</div>
                    <div class="metric-value">{escape(class_text(safe_output))}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="dashboard-card">
            <div class="card-title">Xác suất từng nhóm</div>
        """,
        unsafe_allow_html=True,
    )
    table = probability_table(primary)
    st.dataframe(table, hide_index=True, use_container_width=True, height=160)
    render_probability_chart(table)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Phân loại ảnh MRI não bằng AI", layout="wide")
    inject_css()

    st.markdown("# Phân loại ảnh MRI não bằng AI")
    st.markdown(
        '<div class="subtitle">Ứng dụng demo nghiên cứu hỗ trợ phân loại ảnh MRI não. Kết quả không phải chẩn đoán y khoa.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="top-disclaimer">{escape(DISCLAIMER)}</div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([0.9, 1.1], gap="large")

    with left_col:
        st.markdown("### 1. Tải ảnh MRI")
        uploaded_file = st.file_uploader(
            "Tải ảnh MRI lên",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Chọn một ảnh MRI não để hệ thống phân tích.",
        )

        if uploaded_file is None:
            st.markdown(
                """
                <div class="dashboard-card">
                    <div class="card-title">Trạng thái</div>
                    <div class="metric-value">Chưa có ảnh được tải lên.</div>
                    <div class="small-muted">Chọn một ảnh MRI não để hệ thống phân tích.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ảnh đã tải lên", use_container_width=True)

    with right_col:
        try:
            device = get_device()
            primary_model = load_model(str(CONVNEXT_CHECKPOINT), "convnext_tiny")
            safety_model = load_model(str(DENSENET_CHECKPOINT), "densenet121")

            with st.spinner("Đang chạy mô hình AI..."):
                primary_prediction = predict(primary_model, image, device)
                safety_prediction = predict(safety_model, image, device)
                safe_output = final_safe_output(primary_prediction, safety_prediction)
        except Exception as exc:
            st.error(f"Không thể chạy ứng dụng demo: {exc}")
            return

        render_result_panel(primary_prediction, safety_prediction, safe_output)


if __name__ == "__main__":
    main()
