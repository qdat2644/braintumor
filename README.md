# Demo AI phân loại ảnh MRI não

## 1. Tổng quan

Đây là dự án demo/nghiên cứu phân loại ảnh MRI não bằng học sâu. Mô hình dự đoán một trong bốn lớp:

- glioma
- meningioma
- pituitary
- notumor

Dự án được xây dựng bằng PyTorch, timm và Streamlit. Mục tiêu chính là thực hành pipeline machine learning, đánh giá rủi ro mô hình và trình bày một demo portfolio. Dự án này không phải hệ thống chẩn đoán y khoa.

## 2. Tính năng chính

- Kiểm tra dataset: số lượng ảnh theo lớp, ảnh lỗi, kích thước ảnh và ảnh trùng lặp.
- Làm sạch bằng manifest CSV mà không xóa, di chuyển hoặc sửa ảnh gốc.
- Baseline EfficientNet-B0.
- Thử nghiệm DenseNet121 và ConvNeXt-Tiny.
- Đánh giá rủi ro y khoa, tập trung vào lỗi tumor-to-notumor.
- Debug hard case bằng Grad-CAM.
- Thử nghiệm bộ lọc nhị phân tumor-vs-notumor.
- Ensemble safety override giữa ConvNeXt-Tiny và DenseNet121.
- Giao diện demo Streamlit bằng tiếng Việt với wording thận trọng.

## 3. Demo app

Ứng dụng demo nằm ở:

```text
app.py
```

Logic demo hiện tại:

- Mô hình chính: ConvNeXt-Tiny.
- Mô hình kiểm tra đối chứng: DenseNet121.
- Nếu ConvNeXt-Tiny dự đoán `notumor` nhưng DenseNet121 dự đoán một lớp tumor, app hiển thị kết quả không chắc chắn và khuyến nghị chuyên gia xem xét lại.

App không đưa ra khẳng định y khoa chắc chắn và không dùng wording kiểu xác nhận người dùng có hoặc không có bệnh.

## 4. Dataset

Dataset gốc không được đưa vào repository vì dung lượng và giới hạn bản quyền/giấy phép.

Tải dataset thủ công từ Kaggle, sau đó đặt vào cấu trúc sau:

```text
data/extracted/Training
data/extracted/Testing
```

Các thư mục lớp mong đợi:

```text
glioma
meningioma
notumor
pituitary
```

## 5. Cấu trúc dự án

```text
brain-tumor/
|-- app.py
|-- src/
|-- docs/
|-- outputs/
|-- data/
|-- requirements.txt
|-- README.md
`-- .gitignore
```

Ghi chú:

- `data/` không nên commit vì chứa dataset.
- `outputs/checkpoints/` không nên commit vì chứa model checkpoint.
- Các artifact nặng như Grad-CAM, ảnh debug và file `.pth` được ignore.

## 6. Cài đặt môi trường

Trên Windows với Conda:

```powershell
conda create -n dat python=3.10 -y
conda activate dat
```

Cài PyTorch CUDA theo đúng GPU/CUDA bằng trang chính thức của PyTorch:

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Không dùng bản CPU-only nếu muốn chạy GPU.

Sau đó cài các thư viện còn lại:

```powershell
pip install -r requirements.txt
```

## 7. Chạy demo

```powershell
streamlit run app.py
```

Demo cần các checkpoint đã train sẵn trong:

```text
outputs/checkpoints/
```

## 8. Tái chạy pipeline

Kiểm tra dataset và tạo manifest:

```powershell
python src\audit_dataset.py
python src\create_manifest.py
```

Train và evaluate baseline:

```powershell
python src\train.py
python src\evaluate.py
```

Ví dụ chạy thí nghiệm ConvNeXt-Tiny:

```powershell
python src\run_experiment.py --model-name convnext_tiny --image-size 224 --pad-to-square --no-horizontal-flip --experiment-name convnext_tiny_pad224_nohflip
```

## 9. Tóm tắt kết quả

| Model | Accuracy | Macro F1 | Glioma Recall | Tumor-to-Notumor | Ghi chú |
|---|---:|---:|---:|---:|---|
| EfficientNet-B0 manifest baseline | 0.9520 | 0.9505 | 0.8238 | 22 | Baseline |
| DenseNet121 pad224 no horizontal flip | 0.9545 | 0.9532 | 0.8316 | 21 | Ít lỗi tumor-to-notumor hơn ConvNeXt |
| ConvNeXt-Tiny pad224 no horizontal flip | 0.9609 | 0.9598 | 0.8627 | 24 | Metric tổng thể tốt nhất |

Đánh giá ensemble safety:

- Bắt được 14/22 lỗi tumor-to-notumor ban đầu.
- Vẫn bỏ sót 8/22 lỗi tumor-to-notumor.
- Flag nhầm 2 mẫu notumor.

## 10. Hạn chế đã biết

- Dự án chưa được kiểm định lâm sàng.
- Một số ca glioma vẫn bị dự đoán theo hướng no-tumor-like.
- Các hard case đã biết:
  - `Te-gl_143`
  - `Te-gl_341`
  - `Te-gl_74`
- Kết quả no-tumor-like không xác nhận chắc chắn là không có bệnh.
- Dataset còn giới hạn và có thể không tổng quát tốt cho dữ liệu từ bệnh viện, máy chụp, protocol hoặc nhóm bệnh nhân khác.

## 11. Lưu ý an toàn

Dự án này chỉ phục vụ mục đích nghiên cứu, học tập và demo portfolio. Đây không phải công cụ chẩn đoán. Ảnh MRI cần được bác sĩ hoặc chuyên gia chẩn đoán hình ảnh có chuyên môn xem xét.

## 12. Hướng phát triển tiếp theo

- Tiền xử lý ảnh y khoa tốt hơn.
- Đánh giá trên dataset độc lập bên ngoài.
- Calibration xác suất.
- Ước lượng uncertainty tốt hơn.
- Cải thiện xử lý hard case glioma.
- Review lỗi mô hình và Grad-CAM với chuyên gia y khoa.
