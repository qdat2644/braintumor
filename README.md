# Demo AI phân loại ảnh MRI não

## 1. Tổng quan

Đây là dự án demo/nghiên cứu phân loại ảnh MRI não bằng học sâu. Mô hình phân loại ảnh vào một trong bốn lớp:

- glioma
- meningioma
- pituitary
- notumor

Dự án được xây dựng bằng PyTorch, timm và Streamlit. Mục tiêu là trình bày một pipeline machine learning có kiểm tra dữ liệu, đánh giá lỗi rủi ro cao và giao diện demo an toàn. Dự án này không phải hệ thống chẩn đoán y khoa.

## 2. Quick Start

```powershell
conda create -n dat python=3.10 -y
conda activate dat
pip install -r requirements.txt
streamlit run app.py
```

Lưu ý: nếu chưa có checkpoint trong `outputs/checkpoints/`, app sẽ hiển thị cảnh báo và dừng an toàn.

## 3. Tính năng chính

- Kiểm tra dataset: số lượng ảnh theo lớp, ảnh lỗi, kích thước ảnh và ảnh trùng lặp.
- Làm sạch bằng manifest CSV mà không xóa, di chuyển hoặc sửa ảnh gốc.
- Baseline EfficientNet-B0.
- Thử nghiệm DenseNet121 và ConvNeXt-Tiny.
- Đánh giá rủi ro y khoa, tập trung vào lỗi tumor-to-notumor.
- Debug hard case bằng Grad-CAM.
- Thử nghiệm bộ lọc nhị phân tumor-vs-notumor.
- Ensemble safety override giữa ConvNeXt-Tiny và DenseNet121.
- Giao diện demo Streamlit bằng tiếng Việt với wording thận trọng.

## 4. Demo app

Ứng dụng demo nằm ở:

```text
app.py
```

Logic demo hiện tại:

- Mô hình chính: ConvNeXt-Tiny.
- Mô hình kiểm tra đối chứng: DenseNet121.
- Nếu ConvNeXt-Tiny dự đoán `notumor` nhưng DenseNet121 dự đoán một lớp tumor, app hiển thị kết quả không chắc chắn và khuyến nghị chuyên gia xem xét lại.
- Các trường hợp còn lại hiển thị kết quả từ ConvNeXt-Tiny bằng wording an toàn.

App không đưa ra khẳng định y khoa chắc chắn và không xác nhận người dùng có hoặc không có bệnh.

## 5. Checkpoint Notice

Model checkpoints không được commit vào repository vì dung lượng lớn.

Các checkpoint cần thiết phải được đặt trong:

```text
outputs/checkpoints/
```

Các file bắt buộc cho demo:

```text
best_model_convnext_tiny_pad224_nohflip.pth
best_model_densenet121_pad224_nohflip.pth
```

Người dùng có hai lựa chọn:

- Train model locally bằng các script trong `src/`.
- Tải checkpoint từ GitHub Releases nếu repo owner cung cấp, sau đó đặt vào `outputs/checkpoints/`.

## 6. Dataset Notice

Raw MRI dataset không được đưa vào repository.

Người dùng cần tải dataset thủ công và đặt đúng cấu trúc:

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

## 7. Demo Screenshot

![Demo UI](docs/images/demo_home.png)

Demo screenshots can be added under `docs/images/`.

## 8. Cấu trúc dự án

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

## 9. Cài đặt môi trường

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

## 10. Chạy demo

```powershell
streamlit run app.py
```

## 11. Tái chạy pipeline

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

## 12. Tóm tắt kết quả

| Model | Accuracy | Macro F1 | Glioma Recall | Tumor-to-Notumor | Ghi chú |
|---|---:|---:|---:|---:|---|
| EfficientNet-B0 manifest baseline | 0.9520 | 0.9505 | 0.8238 | 22 | Baseline |
| DenseNet121 pad224 no horizontal flip | 0.9545 | 0.9532 | 0.8316 | 21 | Ít lỗi tumor-to-notumor hơn ConvNeXt |
| ConvNeXt-Tiny pad224 no horizontal flip | 0.9609 | 0.9598 | 0.8627 | 24 | Metric tổng thể tốt nhất |

Đánh giá ensemble safety:

- Bắt được 14/22 lỗi tumor-to-notumor ban đầu.
- Vẫn bỏ sót 8/22 lỗi tumor-to-notumor.
- Flag nhầm 2 mẫu notumor.

## 13. Known Hard Cases

Một số ca glioma vẫn bị dự đoán theo hướng no-tumor-like bởi setup demo hiện tại:

- `Te-gl_143`
- `Te-gl_341`
- `Te-gl_74`

Chi tiết thêm nằm trong [docs/HARD_CASES.md](docs/HARD_CASES.md).

## 14. Safety Disclaimer

Dự án này chỉ phục vụ mục đích nghiên cứu/demo và không phải công cụ chẩn đoán y khoa.

Kết quả no-tumor-like không xác nhận chắc chắn là không có bệnh. Một số hard case glioma vẫn bị dự đoán theo hướng no-tumor-like. Ảnh MRI cần được bác sĩ hoặc chuyên gia chẩn đoán hình ảnh có chuyên môn xem xét.

## 15. Hướng phát triển tiếp theo

- Tiền xử lý ảnh MRI chuyên biệt hơn.
- Đánh giá trên dataset độc lập bên ngoài.
- Calibration xác suất.
- Ước lượng uncertainty tốt hơn.
- Cải thiện xử lý hard case glioma.
- Review lỗi mô hình và Grad-CAM với chuyên gia y khoa.
