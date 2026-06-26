## 📋 Mục Lục

| # | Phần | Mô Tả |
|---|------|--------|
| 1 | Tổng Quan Dự Án | Vấn đề, mục tiêu và hướng tiếp cận |
| 2 | Quy Trình ML | Sơ đồ workflow từ đầu đến cuối |
| 3 | Cấu Trúc Dự Án | Bố cục thư mục và file |
| 4 | Bộ Dữ Liệu | Nguồn dữ liệu, đặc trưng và tải xuống |
| 5 | Hướng dẫn cài đặt và Chạy dự án | Hướng dẫn cài đặt và yêu cầu |
| 6 | Kết Quả & Đánh Giá | So sánh đầy đủ các chỉ số |
| 7 | Phát Hiện Chính | Những gì dữ liệu cho thấy |
| 8 | Hạn Chế | Đánh giá trung thực về các điểm yếu |
| 9 | Hướng Phát Triển | Lộ trình cải thiện |
| 10 | Tài Liệu Tham Khảo | Nguồn dữ liệu và trích dẫn |

# 1. Tổng quan dự án
dự án này được đặt tên là dự án creditcard fraud detection. Đây là Dự án  xây dựng một hệ thống học máy (Machine Learning) toàn diện nhằm phát hiện các giao dịch gian lận thẻ tín dụng. Bộ dữ liệu được sử dụng chứa các giao dịch được thực hiện bằng thẻ tín dụng vào tháng 9 năm 2013 bởi các chủ thẻ châu Âu. Bộ dữ liệu này bị mất cân bằng lớp cực kỳ nghiêm trọng, đòi hỏi các kỹ thuật tiền xử lý và mô hình hóa đặc thù.

## 🤖 Hướng Tiếp Cận

Dự án này tiếp cận bài toán phát hiện gian lận như một bài toán **phân loại nhị phân** sử dụng Machine Learning có giám sát:

| Lớp | Nhãn | Ý Nghĩa |
|-----|------|---------|
| Hợp lệ | 0 | Giao dịch bình thường |
| Gian lận | 1 | Giao dịch đáng ngờ |

### ⚠️ Thách Thức Cốt Lõi: Mất Cân Bằng Dữ Liệu Nghiêm Trọng

Bộ dữ liệu chứa **284.807 giao dịch**, trong đó chỉ có **492 giao dịch (0,17%)** là gian lận. Một mô hình đơn giản luôn dự đoán "không gian lận" sẽ đạt độ chính xác **99,83%** nhưng **không phát hiện được bất kỳ giao dịch gian lận nào** — điều này khiến độ chính xác (accuracy) trở thành một chỉ số hoàn toàn gây hiểu lầm trong bài toán này.

Dự án giải quyết vấn đề mất cân bằng dữ liệu bằng **SMOTE** và đánh giá mô hình dựa trên **Recall, Precision, F1-Score và ROC-AUC** thay vì accuracy.

## 2. 🔄 Quy Trình Dự Án

### Bước 1 — Khám Phá Dữ Liệu Ban Đầu (EDA)
- **Mã nguồn:** `scripts/buoc_1.py`
- Đọc dữ liệu, kiểm tra cấu trúc và tìm giá trị khuyết thiếu
- Phân tích sự mất cân bằng giữa giao dịch bình thường (Class 0) và gian lận (Class 1)
- **Đầu ra:** `output/class_distribution.png`

### Bước 2 — Phân Tích Mô Tả Chi Tiết (Descriptive Analysis)
- **Mã nguồn:** `scripts/buoc_2.py`
- Thống kê mô tả số tiền giao dịch (`Amount`) và thời gian (`Time`)
- Vẽ ma trận tương quan giữa tất cả các đặc trưng
- **Đầu ra:** `output/amount_distribution.png`, `output/time_distribution.png`, `output/correlation_matrix.png`

### Bước 3 — Tiền Xử Lý Dữ Liệu (Preprocessing)
- **Mã nguồn:** `scripts/buoc_3.py`
- Chuẩn hóa dữ liệu bằng `RobustScaler`
- Phân chia tập Train/Test theo tỷ lệ **80/20**
- Áp dụng **SMOTE** để cân bằng tập huấn luyện (từ 394 → 227,451 mẫu gian lận)

### Bước 4 — Huấn Luyện và Đánh Giá Mô Hình (Modelling)
- **Mã nguồn:** `scripts/buoc_4.py`
- Huấn luyện 3 mô hình: **Logistic Regression**, **Random Forest**, **XGBoost**
- Đánh giá dựa trên: Recall, Precision, F1-Score, ROC-AUC
- **Đầu ra:** `output/confusion_matrices.png`

### Bước 5 — Tạo Kết Quả Dự Đoán (Submission)
- **Mã nguồn:** `scripts/buoc_5_submission.py`
- Huấn luyện mô hình đề xuất **Logistic Regression** trên toàn bộ dữ liệu
- **Đầu ra:** `output/submission_logistic_regression.csv`

## 3. 📁 Cấu Trúc Dự Án

```
credit-card-fraud-detection/
├── README.md
├── requirements.txt
│
├── data/
│   └── creditcard.csv
│
├── scripts/
│   ├── buoc_1.py
│   ├── buoc_2.py
│   ├── buoc_3.py
│   ├── buoc_4.py
│   └── buoc_5_submission.py
│
├── output/
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── time_distribution.png
│   ├── correlation_matrix.png
│   ├── confusion_matrices.png
│   └── submission_logistic_regression.csv
│
└── notebooks/ (optional)
    └── fraud_detection_full.ipynb
```

## 4. Bộ dữ liệu
Do giới hạn dung lượng tệp tin của GitHub (<100MB), tệp dữ liệu gốc `creditcard.csv` (150MB) không được đẩy lên kho lưu trữ này. 
- Bạn có thể tải bộ dữ liệu trực tiếp từ liên kết: [GeeksforGeeks creditcard.csv](https://media.geeksforgeeks.org/wp-content/uploads/20240904104950/creditcard.csv)
- Sau khi tải về, hãy đặt tệp tin `creditcard.csv` vào thư mục gốc của dự án.


## 5. 🛠️ Hướng dẫn cài đặt và Chạy dự án

### 1. Cài đặt các thư viện cần thiết
Đảm bảo bạn đã cài đặt Python. Cài đặt các thư viện phụ thuộc bằng lệnh sau:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

### 2. Chạy lần lượt các bước
Để chạy các đoạn mã trong môi trường Windows PowerShell, vui lòng thiết lập bảng mã UTF-8 bằng cách thêm tiền tố lệnh:

- **Chạy Bước 1 (EDA ban đầu)**:
  ```powershell
  $env:PYTHONIOENCODING='utf-8'; python buoc_1.py
  ```
- **Chạy Bước 2 (Phân tích chi tiết)**:
  ```powershell
  $env:PYTHONIOENCODING='utf-8'; python buoc_2.py
  ```
- **Chạy Bước 3 (Tiền xử lý & SMOTE)**:
  ```powershell
  $env:PYTHONIOENCODING='utf-8'; python buoc_3.py
  ```
- **Chạy Bước 4 (Huấn luyện & Đánh giá)**:
  ```powershell
  $env:PYTHONIOENCODING='utf-8'; python buoc_4.py
  ```
- **Chạy Bước 5 (Tạo kết quả dự đoán - Submission)**:
  ```powershell
  $env:PYTHONIOENCODING='utf-8'; python buoc_5_submission.py
  ```

## 6. Kết quả và đánh giá
## 12. 📈 Kết Quả & Đánh Giá

### So Sánh Hiệu Năng 3 Mô Hình

| Mô Hình | Accuracy | Precision | Recall | F1-Score | Gian lận bắt được |
|---------|----------|-----------|--------|----------|-------------------|
| **Logistic Regression** | 97.47% | 5.91% | **91.84%** | 11.10% | **90/98** |
| **XGBoost** | 99.68% | 33.85% | 88.78% | 49.01% | 87/98 |
| **Random Forest** | 99.75% | **39.25%** | 85.71% | **53.85%** | 84/98 |

### Ma Trận Nhầm Lẫn (Confusion Matrix)

![Confusion Matrix](output/confusion_matrices.png)

| | Logistic Regression | Random Forest | XGBoost |
|--|--|--|--|
| Dự đoán đúng bình thường (TN) | 55,430 | 56,734 | 56,694 |
| Báo nhầm gian lận (FP) | 1,434 | 130 | 170 |
| Bỏ sót gian lận (FN) | 8 | 14 | 11 |
| Bắt đúng gian lận (TP) | **90** | 84 | 87 |

### Nhận Xét
- **Logistic Regression** bắt được nhiều gian lận nhất (90/98) nhưng báo nhầm nhiều (1,434 trường hợp)
- **Random Forest** báo nhầm ít nhất (130 FP) nhưng bỏ sót nhiều gian lận hơn
- **XGBoost** cân bằng tốt nhất giữa Recall và Precision

> ⚠️ Trong thực tế, **bỏ sót gian lận (FN) nguy hiểm hơn báo nhầm (FP)**, do đó **Logistic Regression** là lựa chọn phù hợp nhất.

## 🤝 Đóng góp
Nếu bạn có bất kỳ đề xuất cải tiến nào về mô hình (như Hyperparameter Tuning hoặc Grid Search), hãy tạo Pull Request hoặc Issues trên repo này!

## 7. Những phát hiện chính
## 13. 🔍 Những Phát Hiện Chính

### 📊 Về Dữ Liệu
- Dữ liệu mất cân bằng nghiêm trọng: chỉ **0.17%** giao dịch là gian lận (492/284,807)
- Giao dịch gian lận có số tiền **thấp hơn** giao dịch bình thường (trung vị ~10 USD vs ~20 USD)
  → Kẻ gian lận thường thực hiện nhiều giao dịch nhỏ để tránh bị phát hiện
- Giao dịch gian lận **không theo chu kỳ thời gian** — hoạt động cả vào ban đêm (3h–4h sáng)
  khi giao dịch bình thường ở mức thấp nhất
- Các đặc trưng V1–V28 (kết quả PCA) **gần như độc lập với nhau**
  → Không có đặc trưng đơn lẻ nào đủ mạnh để phát hiện gian lận

### 🤖 Về Mô Hình
- **SMOTE** cải thiện đáng kể khả năng phát hiện gian lận so với không xử lý mất cân bằng
- Không có mô hình nào hoàn hảo — mỗi mô hình có sự đánh đổi riêng giữa Recall và Precision
- **Logistic Regression** được khuyến nghị vì Recall cao nhất (91.84%), bỏ sót ít gian lận nhất

### 🚀 Hướng Phát Triển Tiếp Theo
- Tinh chỉnh ngưỡng phân loại (threshold tuning) để tối ưu Recall và Precision cùng lúc
- Kết hợp các mô hình (Ensemble Methods) để cải thiện tổng thể
- Thu thập thêm dữ liệu gian lận thực tế để giảm mức độ mất cân bằng





