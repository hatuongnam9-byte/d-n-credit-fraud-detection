import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Đọc file dữ liệu bằng pandas
print("Đang đọc file dữ liệu 'creditcard.csv'...")
df = pd.read_csv('creditcard.csv')
print("Đọc dữ liệu thành công!\n")

# 2. Hiển thị 10 dòng dữ liệu đầu tiên
print("=== 10 DÒNG DỮ LIỆU ĐẦU TIÊN ===")
print(df.head(10))
print("\n" + "="*50 + "\n")

# 3. Hiển thị thống kê mô tả (describe) của bộ dữ liệu
print("=== THỐNG KÊ MÔ TẢ CHUNG CỦA BỘ DỮ LIỆU ===")
# Đặt tùy chọn hiển thị đầy đủ các cột để không bị ẩn dấu chấm lửng
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.describe())
print("\n" + "="*50 + "\n")

# 4. Phân tích mô tả chi tiết cho trường Amount (Số tiền giao dịch) theo từng lớp
print("=== THỐNG KÊ MÔ TẢ SỐ TIỀN GIAO DỊCH (AMOUNT) THEO LỚP ===")
normal_amount = df[df['Class'] == 0]['Amount']
fraud_amount = df[df['Class'] == 1]['Amount']

print("--- Giao dịch BÌNH THƯỜNG (Normal - Class 0) ---")
print(normal_amount.describe())
print(f"Median (Trung vị): {normal_amount.median():.2f}")
print("\n--- Giao dịch GIAN LẬN (Fraud - Class 1) ---")
print(fraud_amount.describe())
print(f"Median (Trung vị): {fraud_amount.median():.2f}")
print("\n" + "="*50 + "\n")

# 5. Vẽ biểu đồ hộp (Box Plot) cho Amount phân nhóm theo Class
print("Đang vẽ biểu đồ hộp so sánh số tiền giao dịch...")
plt.figure(figsize=(8, 6))
# Sử dụng thang đo log(Amount + 1) để tránh việc lệch quá nhiều do các giao dịch có số tiền cực lớn
sns.boxplot(x='Class', y='Amount', data=df, palette='Set2')
plt.yscale('log') # Thang đo logarit để dễ quan sát phân phối hộp
plt.title('Biểu đồ hộp số tiền giao dịch theo Lớp (Thang đo Log)')
plt.xlabel('Lớp (0: Bình thường, 1: Gian lận)')
plt.ylabel('Số tiền giao dịch (Log Scale - USD)')
plt.tight_layout()
plt.savefig('amount_distribution.png', dpi=300)
print("Đã lưu biểu đồ hộp số tiền giao dịch vào file 'amount_distribution.png'")

# 6. Vẽ biểu đồ mật độ (KDE Plot) phân bố giao dịch theo thời gian
print("Đang vẽ biểu đồ phân bố thời gian giao dịch...")
plt.figure(figsize=(10, 6))
sns.kdeplot(df[df['Class'] == 0]['Time'], label='Bình thường (Class 0)', shade=True, color='blue')
sns.kdeplot(df[df['Class'] == 1]['Time'], label='Gian lận (Class 1)', shade=True, color='red')
plt.title('Phân bố thời gian giao dịch (Time) theo Lớp')
plt.xlabel('Thời gian (Giây)')
plt.ylabel('Mật độ phân bố (Density)')
plt.legend()
plt.tight_layout()
plt.savefig('time_distribution.png', dpi=300)
print("Đã lưu biểu đồ thời gian giao dịch vào file 'time_distribution.png'")

# 7. Vẽ ma trận tương quan (Correlation Matrix Heatmap)
print("Đang tính toán và vẽ ma trận tương quan...")
plt.figure(figsize=(14, 10))
corr = df.corr()
# Chỉ vẽ tương quan dạng heatmap đơn giản, không ghi số vì quá nhiều biến (31 biến)
sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, annot=False)
plt.title('Ma trận tương quan giữa các đặc trưng')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300)
print("Đã lưu ma trận tương quan vào file 'correlation_matrix.png'")
