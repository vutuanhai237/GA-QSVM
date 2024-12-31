import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
df_updated = pd.read_csv(r"C:\Desktop\GA+QSVM\germancredit_data.csv")  # Cập nhật đường dẫn chính xác


# 1. Chuyển đổi dữ liệu phân loại (object)
label_encoders = {}
for col in df_updated.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df_updated[col] = le.fit_transform(df_updated[col])
    label_encoders[col] = le

# 2. Xử lý cột boolean
# Chuyển đổi các cột boolean sang số (0 và 1) nếu có
boolean_cols = df_updated.select_dtypes(include=["bool"]).columns
df_updated[boolean_cols] = df_updated[boolean_cols].astype(int)

# 3. Tách dữ liệu đầu vào và nhãn mục tiêu
target_column = "Default"  # Cột mục tiêu
X = df_updated.drop(columns=[target_column])  # Dữ liệu đầu vào
y = df_updated[target_column]  # Nhãn mục tiêu

# 4. Chuẩn hóa dữ liệu số
scaler = StandardScaler()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# 5. Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Kiểm tra kết quả
print("Kích thước dữ liệu huấn luyện:", X_train.shape, y_train.shape)
print("Kích thước dữ liệu kiểm tra:", X_test.shape, y_test.shape)

