import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Đọc file dữ liệu bằng pandas
print("Đang đọc file dữ liệu 'creditcard.csv'...")
df = pd.read_csv('creditcard.csv')
print("Đọc dữ liệu thành công!\n")

# 2. Hiển thị thông tin tổng quan của dữ liệu
print("=== 5 DÒNG DỮ LIỆU ĐẦU TIÊN ===")
print(df.head())
print("\n" + "="*50 + "\n")

print("=== THÔNG TIN CẤU TRÚC DỮ LIỆU (INFO) ===")
print(df.info())
print("\n" + "="*50 + "\n")

print("=== THỐNG KÊ MÔ TẢ (DESCRIBE) ===")
print(df.describe())
print("\n" + "="*50 + "\n")

# 3. Kiểm tra giá trị khuyết thiếu (Missing Values)
print("=== KIỂM TRA GIÁ TRỊ KHUYẾT THIẾU ===")
missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    print("Tuyệt vời! Không có giá trị khuyết thiếu nào trong bộ dữ liệu.")
else:
    print("Danh sách các cột có giá trị khuyết thiếu:")
    print(missing_values[missing_values > 0])
print("\n" + "="*50 + "\n")

# 4. Phân tích phân phối của lớp (Class)
print("=== PHÂN TÍCH PHÂN PHỐI LỚP (CLASS DISTRIBUTION) ===")
class_counts = df['Class'].value_counts()
class_percentage = df['Class'].value_counts(normalize=True) * 100

print(f"Số lượng giao dịch bình thường (Class 0): {class_counts[0]} ({class_percentage[0]:.4f}%)")
print(f"Số lượng giao dịch gian lận (Class 1): {class_counts[1]} ({class_percentage[1]:.4f}%)")
print("\nNhận xét: Bộ dữ liệu bị mất cân bằng lớp cực kỳ nghiêm trọng!")
print("\n" + "="*50 + "\n")

# 5. Vẽ biểu đồ phân phối lớp và lưu thành file ảnh
print("Đang vẽ biểu đồ phân phối lớp...")
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=df, palette='Set2')
plt.title('Phân Phối Lớp Giao Dịch (0: Bình thường, 1: Gian lận)')
plt.xlabel('Lớp (Class)')
plt.ylabel('Số lượng giao dịch')

# Thêm nhãn số lượng lên đầu các cột
for i, count in enumerate(class_counts):
    plt.text(i, count + 1000, f"{count}\n({class_percentage[i]:.4f}%)", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300)
print("Đã lưu biểu đồ phân phối lớp vào file 'class_distribution.png'