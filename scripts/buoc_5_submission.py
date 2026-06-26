import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 1. Tạo thư mục output nếu chưa tồn tại
os.makedirs('output', exist_ok=True)

# 2. Đọc file dữ liệu
print("Đang đọc file dữ liệu 'creditcard.csv'...")
df = pd.read_csv('creditcard.csv')
print(f"Đọc dữ liệu thành công! Kích thước: {df.shape}\n")

# 保存原始 Index làm ID giao dịch để tham chiếu sau này
df['Original_Index'] = df.index

# 3. Tiền xử lý dữ liệu
print("Đang tiền xử lý dữ liệu...")
robust_scaler = RobustScaler()
df['scaled_amount'] = robust_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = robust_scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Tách X, y và giữ lại cột Original_Index riêng
X = df.drop(['Class', 'Time', 'Amount', 'Original_Index'], axis=1)
y = df['Class']
indices = df['Original_Index']

# 4. Chia tập dữ liệu Train/Test (80/20) có phân tầng
print("Đang phân chia tập dữ liệu thành Train và Test (80/20)...")
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X, y, indices, test_size=0.2, random_state=42, stratify=y
)

# 5. Áp dụng SMOTE trên tập Train
print("Đang áp dụng SMOTE trên tập Train...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Kích thước tập Train sau SMOTE: {X_train_res.shape}\n")

# 6. Huấn luyện mô hình Logistic Regression
print("Đang huấn luyện mô hình Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_res, y_train_res)
print("Huấn luyện thành công!\n")

# 7. Dự đoán trên tập kiểm thử (Test Set)
print("=== DỰ ĐOÁN TRÊN TẬP KIỂM THỬ (TEST SET) ===")
y_pred_test = model.predict(X_test)
y_prob_test = model.predict_proba(X_test)[:, 1]

# Tính toán các chỉ số
print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ:")
print(classification_report(y_test, y_pred_test))
cm = confusion_matrix(y_test, y_pred_test)
print("Ma trận nhầm lẫn:")
print(cm)
print(f"Bắt đúng gian lận (TP)        : {cm[1,1]} / {sum(y_test)}")
print(f"Số lượng báo nhầm gian lận (FP): {cm[0,1]}\n")

# Tạo file submission cho tập kiểm thử
submission_test_df = pd.DataFrame({
    'Original_Index': indices_test.values,
    'Actual_Class': y_test.values,
    'Predicted_Class': y_pred_test,
    'Fraud_Probability': y_prob_test
}).sort_values(by='Original_Index')

submission_test_path = 'output/submission_logistic_regression.csv'
submission_test_df.to_csv(submission_test_path, index=False)
print(f"Đã lưu kết quả dự đoán tập Test vào: {submission_test_path}")
print(f"Kích thước file kết quả tập Test: {submission_test_df.shape}\n")

# 8. Xóa file dự đoán toàn bộ tập dữ liệu cũ nếu tồn tại (để chỉ giữ lại dự đoán tập Test 20%)
submission_full_path = 'output/full_predictions_logistic_regression.csv'
if os.path.exists(submission_full_path):
    os.remove(submission_full_path)
    print(f"Đã xóa file dự đoán toàn bộ dữ liệu cũ: {submission_full_path}\n")

print("Hoàn tất quá trình tạo kết quả submission chỉ dành cho tập Test 20%!")
