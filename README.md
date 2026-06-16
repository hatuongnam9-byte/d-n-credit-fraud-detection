
# Phát hiện Gian lận Thẻ Tín dụng (Credit Card Fraud Detection)

Dự án này xây dựng một hệ thống học máy (Machine Learning) toàn diện nhằm phát hiện các giao dịch gian lận thẻ tín dụng. Bộ dữ liệu được sử dụng chứa các giao dịch được thực hiện bằng thẻ tín dụng vào tháng 9 năm 2013 bởi các chủ thẻ châu Âu. Bộ dữ liệu này bị mất cân bằng lớp cực kỳ nghiêm trọng, đòi hỏi các kỹ thuật tiền xử lý và mô hình hóa đặc thù.
# 🔍 Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-orange.svg)
Dự án phát hiện giao dịch thẻ tín dụng gian lận sử dụng các thuật toán Machine Learning. 
Dataset sử dụng: [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),  
gồm **284,807 giao dịch** với **492 giao dịch gian lận (0.17%)**
---

## 📌 Bộ dữ liệu (Dataset)
Do giới hạn dung lượng tệp tin của GitHub (<100MB), tệp dữ liệu gốc `creditcard.csv` (150MB) không được đẩy lên kho lưu trữ này. 
- Bạn có thể tải bộ dữ liệu trực tiếp từ liên kết: [GeeksforGeeks creditcard.csv](https://media.geeksforgeeks.org/wp-content/uploads/20240904104950/creditcard.csv)
- Sau khi tải về, hãy đặt tệp tin `creditcard.csv` vào thư mục gốc của dự án.

---

## 📂 Cấu trúc thư mục dự án
Dự án được chia thành các bước rõ ràng thông qua các tệp tin mã nguồn (có cả bản tiếng Việt có dấu và bản không dấu để tránh lỗi mã hóa trên Windows PowerShell):

1. **Bước 1: Khám phá và phân tích dữ liệu ban đầu (EDA)**
   - Mã nguồn: `buoc_1.py` hoặc `Bước 1 Khám phá và phân tích dữ liệu ban đầu (EDA).py`
   - Nhiệm vụ: Đọc dữ liệu, kiểm tra cấu trúc, tìm giá trị khuyết thiếu và phân tích sự mất cân bằng giữa giao dịch bình thường (Class 0) và gian lận (Class 1).
   - Biểu đồ đầu ra: `class_distribution.png` (Biểu thị tỷ lệ mất cân bằng lớp).
   - Kết quả biểu đồ class distribution:
     ![Class Distribution](outputs/class_distribution.png)
<img width="2400" height="1800" alt="image" src="https://github.com/user-attachments/assets/09679287-0525-454e-b0a1-6e63d1300316" />

2. **Bước 2: Phân tích mô tả chi tiết (Descriptive Analysis)**
   - Mã nguồn: `buoc_2.py` hoặc `Bước 2 Phân tích mô tả chi tiết (Descriptive Analysis).py`
   - Nhiệm vụ: Thống kê mô tả chi tiết thuộc tính số tiền giao dịch (`Amount`), phân bố thời gian giao dịch (`Time`), và vẽ ma trận tương quan giữa tất cả các đặc trưng.
   - Biểu đồ đầu ra: `amount_distribution.png` (Box plot số tiền), `time_distribution.png` (KDE plot thời gian), `correlation_matrix.png` (Heatmap ma trận tương quan).

3. **Bước 3: Tiền xử lý dữ liệu (Data Preprocessing)**
   - Mã nguồn: `buoc_3.py` hoặc `Bước 3 Tiền xử lý dữ liệu (Data Preprocessing).py`
   - Nhiệm vụ: Chuẩn hóa dữ liệu bằng `RobustScaler` (chống chịu tốt với ngoại trị), phân chia tập Train/Test theo tỷ lệ 80/20, áp dụng thuật toán **SMOTE** (Oversampling) để cân bằng tập huấn luyện từ 394 mẫu gian lận lên 227,451 mẫu.

4. **Bước 4: Huấn luyện và Đánh giá Mô hình**
   - Mã nguồn: `buoc_4.py` hoặc `Bước 4 Huấn luyện và Đánh giá Mô hình.py`
   - Nhiệm vụ: Huấn luyện 3 mô hình phân loại: **Logistic Regression**, **Random Forest Classifier**, và **XGBoost Classifier**. Thực hiện dự đoán trên tập kiểm thử (giữ nguyên tỷ lệ mất cân bằng thực tế) và so sánh hiệu năng.
   - Biểu đồ đầu ra: `confusion_matrices.png` (Ma trận nhầm lẫn của cả 3 mô hình).

---

## 📈 Kết quả huấn luyện và So sánh mô hình
Dưới đây là bảng so sánh hiệu năng của các mô hình (được xếp thứ tự theo độ nhạy **Recall** lớp 1):

| Mô hình (Model) | Accuracy | Precision (Lớp 1) | Recall (Lớp 1) | F1-Score (Lớp 1) | Số giao dịch gian lận bắt được |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 97.47% | 5.91% | **91.84%** | 11.10% | **90 / 98** |
| **XGBoost** | 99.68% | 33.85% | **88.78%** | 49.01% | **87 / 98** |
| **Random Forest** | 99.75% | **39.25%** | **85.71%** | **53.85%** | **84 / 98** |

### Nhận xét quan trọng:
- **Logistic Regression** đạt Recall cao nhất (91.84%) nhưng Precision vô cùng thấp (5.91%), tạo ra quá nhiều báo động giả (False Positives), gây phiền hà cho khách hàng trong thế giới thực.
- **XGBoost** là mô hình có sự cân bằng tối ưu nhất cho bài toán này, đạt Recall **88.78%** và Precision tăng vượt bậc lên **33.85%** (F1-score 49.01%).
- **Random Forest** cho số lượng báo động giả ít nhất với Precision cao nhất là **39.25%**, tuy nhiên bỏ sót nhiều giao dịch gian lận hơn (Recall 85.71%).
> ⚠️ **Lưu ý:** Trong bài toán phát hiện gian lận, **bỏ sót giao dịch gian lận (False Negative) nguy hiểm hơn báo nhầm (False Positive)**. Do đó, **Logistic Regression** là lựa chọn phù hợp nhất cho môi trường thực tế.
## 🎯 Kết luận

Qua quá trình thử nghiệm 3 mô hình trên bộ dữ liệu mất cân bằng nghiêm trọng (0.17% gian lận), kết quả cho thấy:

- Không có mô hình nào hoàn hảo — mỗi mô hình có sự đánh đổi riêng giữa Recall và Precision, lựa chọn chỉ số nào nên dùng phụ thuộc và yêu cầu của dự án
- Kỹ thuật **SMOTE** giúp cải thiện đáng kể khả năng phát hiện gian lận so với không xử lý mất cân bằng
- Dựa trên bố cảnh và yêu cầu của bài toán,chỉ số recall sẽ là chỉ số quan trọng nhất trong việc xác định và hạn chế bỏ sót khách hàng gian lận. phương pháp **Logistic Regression** được khuyến nghị cho môi trường thực tế vì bỏ sót ít giao dịch gian lận nhất (90/98) với chỉ số recall cao nhất 91,84%

> Hướng phát triển tiếp theo: Tinh chỉnh ngưỡng phân loại (threshold tuning) và kết hợp các mô hình (Ensemble) để cải thiện cả Recall lẫn Precision cùng lúc.
```
---

## 🛠️ Hướng dẫn cài đặt và Chạy dự án

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

---

## 🤝 Đóng góp
Nếu bạn có bất kỳ đề xuất cải tiến nào về mô hình (như Hyperparameter Tuning hoặc Grid Search), hãy tạo Pull Request hoặc Issues trên repo này!
