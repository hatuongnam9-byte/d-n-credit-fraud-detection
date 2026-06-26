import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 1. Đọc file dữ liệu và Tiền xử lý dữ liệu (Từ Bước 3)
print("Đang đọc file dữ liệu 'creditcard.csv'...")
df = pd.read_csv('creditcard.csv')
print("Đọc dữ liệu thành công!\n")

print("Đang tiền xử lý dữ liệu...")
# Chuẩn hóa Time và Amount bằng RobustScaler
robust_scaler = RobustScaler()
df['scaled_amount'] = robust_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = robust_scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# Tách X, y
X = df.drop('Class', axis=1)
y = df['Class']

# Chia tập dữ liệu Train/Test (80/20) có phân tầng
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Áp dụng SMOTE trên tập Train
print("Đang áp dụng SMOTE trên tập Train...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Tiền xử lý và cân bằng dữ liệu hoàn tất.\n")
print(f"Kích thước tập Train sau SMOTE: {X_train_res.shape}")
print(f"Kích thước tập Test: {X_test.shape}")
print("\n" + "="*50 + "\n")

# 2. Huấn luyện các mô hình
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
}

results = {}

for name, model in models.items():
    print(f"Đang huấn luyện mô hình {name}...")
    model.fit(X_train_res, y_train_res)
    print(f"Huấn luyện {name} thành công!")
    
    # Dự đoán trên tập Test
    y_pred = model.predict(X_test)
    
    # Tính toán các chỉ số
    results[name] = {
        "predictions": y_pred,
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    print(f"Đã đánh giá xong mô hình {name}.\n")

print("="*50 + "\n")

# 3. In kết quả đánh giá chi tiết
for name, metrics in results.items():
    print(f"=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH: {name} ===")
    print(metrics["classification_report"])
    print(f"Accuracy : {metrics['accuracy']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall   : {metrics['recall']:.6f}")
    print(f"F1-score : {metrics['f1']:.6f}")
    print("\n" + "-"*30 + "\n")

# 4. Trực quan hóa Ma trận nhầm lẫn (Confusion Matrix)
print("Đang vẽ biểu đồ Ma trận nhầm lẫn của các mô hình...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, metrics) in enumerate(results.items()):
    cm = metrics["confusion_matrix"]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                xticklabels=['Thường (0)', 'Gian lận (1)'],
                yticklabels=['Thường (0)', 'Gian lận (1)'])
    axes[idx].set_title(f"Ma trận nhầm lẫn - {name}")
    axes[idx].set_ylabel('Thực tế')
    axes[idx].set_xlabel('Dự đoán')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300)
print("Đã lưu biểu đồ ma trận nhầm lẫn vào file 'confusion_matrices.png'")

# 5. So sánh nhanh hiệu năng mô hình
print("\n=== BẢNG SO SÁNH HIỆU NĂNG MÔ HÌNH (SẮP XẾP THEO RECALL) ===")
summary_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": [metrics["accuracy"] for metrics in results.values()],
    "Precision": [metrics["precision"] for metrics in results.values()],
    "Recall (Độ nhạy)": [metrics["recall"] for metrics in results.values()],
    "F1-Score": [metrics["f1"] for metrics in results.values()]
}).sort_values(by="Recall (Độ nhạy)", ascending=False)

print(summary_df.to_string(index=False))
