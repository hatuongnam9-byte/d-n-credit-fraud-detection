import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

# 1. Đọc file dữ liệu bằng pandas
print("Đang đọc file dữ liệu 'creditcard.csv'...")
df = pd.read_csv('creditcard.csv')
print("Đọc dữ liệu thành công!\n")

# 2. Chuẩn hóa đặc trưng (Feature Scaling) cho Time và Amount bằng RobustScaler
print("Đang chuẩn hóa đặc trưng 'Time' và 'Amount' bằng RobustScaler...")
robust_scaler = RobustScaler()

# Tạo cột mới đã chuẩn hóa và loại bỏ các cột cũ
df['scaled_amount'] = robust_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = robust_scaler.fit_transform(df['Time'].values.reshape(-1, 1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)

# Đưa hai cột vừa chuẩn hóa lên đầu hoặc giữ nguyên vị trí (ở đây chúng ta xếp lại vị trí)
scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']
df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

print("Đã chuẩn hóa xong. 5 dòng dữ liệu đầu tiên sau chuẩn hóa:")
print(df.head())
print("\n" + "="*50 + "\n")

# 3. Tách thuộc tính đặc trưng (X) và nhãn mục tiêu (y)
X = df.drop('Class', axis=1)
y = df['Class']

# 4. Phân chia tập dữ liệu thành Train và Test (tỷ lệ 80% Train, 20% Test)
print("Đang phân chia tập dữ liệu thành Train và Test (tỷ lệ 80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Kích thước tập huấn luyện ban đầu (X_train): {X_train.shape}")
print(f"Kích thước tập kiểm thử (X_test): {X_test.shape}")
print(f"Phân phối lớp trong tập Train ban đầu: {Counter(y_train)}")
print(f"Phân phối lớp trong tập Test: {Counter(y_test)}")
print("\n" + "="*50 + "\n")

# 5. Áp dụng SMOTE trên tập Train để cân bằng lớp dữ liệu
print("Đang áp dụng SMOTE trên tập Train để cân bằng lớp...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Áp dụng SMOTE thành công!")
print(f"Kích thước tập Train sau khi SMOTE (X_train_res): {X_train_res.shape}")
print(f"Phân phối lớp trong tập Train sau khi SMOTE: {Counter(y_train_res)}")
print("\n" + "="*50 + "\n")

# 6. Xác nhận tính cân bằng và lưu trữ thông tin kích thước dữ liệu để bước sau sử dụng
print("=== TỔNG KẾT KÍCH THƯỚC CÁC TẬP DỮ LIỆU ĐÃ SẴN SÀNG HUẤN LUYỆN ===")
print(f"Tập Train Đặc trưng (X_train_res): {X_train_res.shape}")
print(f"Tập Train Nhãn (y_train_res)    : {y_train_res.shape} (Cân bằng: {Counter(y_train_res)[0]} Bình thường vs {Counter(y_train_res)[1]} Gian lận)")
print(f"Tập Test Đặc trưng (X_test)      : {X_test.shape}")
print(f"Tập Test Nhãn (y_test)          : {y_test.shape}")
