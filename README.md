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

Mô hình sẽ dựa trên những đặc điểm của khách hàng và tìm ra mô hình phù hợp cho việc xác định được những đối tượng có khả năng gian lận và tìm ra những khách hàng minh bạch uy tín dựa trên các đặc điểm ấy.

### ⚠️ Thách Thức Cốt Lõi: Mất Cân Bằng Dữ Liệu Nghiêm Trọng

Bộ dữ liệu chứa **284.807 giao dịch**, trong đó chỉ có **492 giao dịch (0,17%)** là gian lận. Một mô hình đơn giản luôn dự đoán "không gian lận" sẽ đạt độ chính xác **99,83%** nhưng **không phát hiện được bất kỳ giao dịch gian lận nào** — điều này khiến độ chính xác (accuracy) trở thành một chỉ số hoàn toàn gây hiểu lầm trong bài toán này.

Dự án giải quyết vấn đề mất cân bằng dữ liệu bằng **SMOTE** và đánh giá mô hình dựa trên **Recall, Precision, F1-Score và ROC-AUC** thay vì accuracy.

## 2. 🔄 Quy Trình Dự Án

### Bước 1 — Khám Phá Dữ Liệu Ban Đầu (EDA)
- **Mã nguồn:** `scripts/buoc_1.py`
- Đọc dữ liệu, kiểm tra cấu trúc và tìm giá trị khuyết thiếu
- Phân tích sự mất cân bằng giữa giao dịch bình thường (Class 0) và gian lận (Class 1)
- **Đầu ra:** ![Biểu đồ phân phối các lớp](output/class_distribution.png)

### Bước 2 — Phân Tích Mô Tả Chi Tiết (Descriptive Analysis) để tìm ra những phát hiện chính
- **Mã nguồn:** `scripts/buoc_2.py`
- Thống kê mô tả số tiền giao dịch (`Amount`) và thời gian (`Time`)
- Vẽ ma trận tương quan giữa tất cả các đặc trưng
- **Đầu ra:** ![Biểu đồ phân phối số tiền giao dịch](output/amount_distribution.png)

![Biểu đồ phân phối thời gian giao dịch](output/time_distribution.png)

![Ma trận tương quan giữa các đặc trưng](output/correlation_matrix.png)


### Bước 3 — Tiền Xử Lý Dữ Liệu (Preprocessing)
- **Mã nguồn:** `scripts/buoc_3.py`
- Chuẩn hóa dữ liệu bằng `RobustScaler`
- Phân chia tập Train/Test theo tỷ lệ **80/20**
- Áp dụng **SMOTE** để cân bằng tập huấn luyện (từ 394 → 227,451 mẫu gian lận)

### Bước 4 — Huấn Luyện và Đánh Giá Mô Hình (Modelling)
- **Mã nguồn:** `scripts/buoc_4.py`
- Huấn luyện 3 mô hình: **Logistic Regression**, **Random Forest**, **XGBoost**
- Đánh giá dựa trên: Recall, Precision, F1-Score, ROC-AUC
- **Đầu ra:** ![Ma trận nhầm lẫn của mô hình](output/confusion_matrices.png)

### Bước 5 — Tạo Kết Quả Dự Đoán (Submission)
- **Mã nguồn:** `scripts/buoc_5_submission.py`
- Huấn luyện mô hình đề xuất **Logistic Regression** trên toàn bộ dữ liệu
- **Đầu ra:** [Tải về file kết quả dự đoán tại đây](output/submission_logistic_regression.csv)

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

## 7. Những phát hiện chính dựa trên kết quả đầu ra

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

## 8. Hạn chế
- Bộ dữ liệu bị thiếu cân bằng và phương pháp SMOTE chỉ là phương pháp tạm thời vì nó là phương pháp tự nhân bản dữ liệu để giải quyết vấn đề chứ chưa thật sự đem lại sự tin tưởng tuyệt đối cho kết quả đầu ra. Cần phải thu thập thêm các dữ liệu nên cũng rất dễ bị rủi ro mô hình bị học lệch dẫn đến khi vào thực tế mô hình không thể dự đoán chính xác
- Việc áp dụng phương pháp PCA sẽ dẫn đến hạn chế trong việc biết rõ tại sao và những yếu tố nào thật sự giúp xác định các đối tượng gian lân vì ý nghĩa các biến đã được thay thế bằng biến V1,V2,V3 nhưng thực chất ý nghĩa biến đó là gì thì không biết. Dẫn đến giả sử phòng risk operation muốn hiểu ý nghĩa các biến để tư vấn cho khách hàng thì không làm được.
- Bộ dữ liệu này bản chất là bộ dữ liệu cố định, các biến đều là cố định trong khi hành vi con người luôn thay đổi theo ngày, tháng nên mô hình dự đoán sẽ dễ bị lỗi thời nếu không cập nhật liên tục.
- Mô hình logistic regression tuy không bỏ lọt gian lận nhiều nhưng lại để việc bắt nhầm khách hàng gian lận là tương đối nhiều điều này dẫn đén tăng chi phí giải quyết vấn đề này của ngân hàng. Dự án chưa thật sự tối ưu bài toán chi phí kinh tế của ngân hàng

## 9. Hướng phát triển
-  thu thập thêm dữ liệu khách hàng gian lận để cân bằng lại dữ liệu thay thế phương pháp nhân bản dữ liệu SMOTE
- Tìm ra giải pháp khác thay thế phương pháp PCA vừa giúp làm rõ ý nghĩa các biến, vừa đảm bảo tính bảo mật đúng như yêu cầu từ ngân hàng
- Cập nhật và thu thập bộ dữ liệu mới giúp xác định sự thay đổi hành vi của khách hàng hằng ngày để giúp cho mô hình nâng cao khả năng dự đoán
- Kết hợp mô hình logistic regression với một mô hình khác giúp ngân hàng giảm thiểu vấn đề báo nhầm gian lận

## 10. Tài liệu tham khảo
- Dal Pozzolo, A., Caelen, O., Johnson, R.A. and Bontempi, G., 2015. Calibrating probability with undersampling for unbalanced classification. In Symposium on Computational Intelligence and Data Mining (CIDM). IEEE.

- Dal Pozzolo, A., Caelen, O., Le Borgne, Y.A., Waterschoot, S. and Bontempi, G., 2014. Learned lessons in credit card fraud detection from a practitioner perspective. Expert Systems with Applications, 41(10), pp. 4915-4928.

- Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C. and Bontempi, G., 2018. Credit card fraud detection: a realistic modeling and a novel learning strategy. IEEE Transactions on Neural Networks and Learning Systems, 29(8), pp. 3784-3797.

- Dal Pozzolo, A., 2015. Adaptive machine learning for credit card fraud detection. PhD thesis. Université Libre de Bruxelles (ULB), Machine Learning Group (MLG).

- Carcillo, F., Dal Pozzolo, A., Le Borgne, Y.A., Caelen, O., Mazzer, Y. and Bontempi, G., 2018. Scarff: a scalable framework for streaming credit card fraud detection with Spark. Information Fusion, 41, pp. 182-194.

- Carcillo, F., Le Borgne, Y.A., Caelen, O. and Bontempi, G., 2018. Streaming active learning strategies for real-life credit card fraud detection: assessment and visualization. International Journal of Data Science and Analytics, 5(4), pp. 285-300.

- Lebichot, B., Le Borgne, Y.A., He, L., Oblé, F. and Bontempi, G., 2019. Deep-learning domain adaptation techniques for credit cards fraud detection. In INNSBDDL 2019: Recent Advances in Big Data and Deep Learning (pp. 78-88). Springer.

- Carcillo, F., Le Borgne, Y.A., Caelen, O., Oblé, F. and Bontempi, G., 2019. Combining unsupervised and supervised learning in credit card fraud detection. Information Sciences, 504, pp. 285-300. (Lưu ý: Đã bổ sung số volume và trang chuẩn xác của bài báo này trên Elsevier).
