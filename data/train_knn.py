"""
Huấn luyện mô hình KNN phân loại tư thế Squat (Giai đoạn 4).
Thực hiện:
1. Đọc và làm sạch dữ liệu từ data/dataset_squat.csv (bỏ dòng comment, kiểm tra NaN, duplicates, phân bố nhãn)
2. Phân tách tập Train / Test có phân tầng (Stratified Split 80/20)
3. Đóng gói quy trình tiền xử lý StandardScaler và KNeighborsClassifier vào Pipeline
4. Huấn luyện và Đánh giá: Accuracy, Precision, Recall, F1-score, Confusion Matrix
5. Đóng gói Pipeline + Metadata (features, classes, metrics) vào data/knn_model.pkl bằng joblib
6. Xuất báo cáo đánh giá và hướng dẫn huấn luyện lại
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Đảm bảo in ký tự tiếng Việt an toàn trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Đảm bảo đường dẫn import core module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.biomechanics import FEATURE_NAMES

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset_squat.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "knn_model.pkl")


def train_model(dataset_path: str = DATASET_PATH, model_save_path: str = MODEL_PATH):
    print("=" * 65)
    print("VIRTUAL FITNESS COACH — HUẤN LUYỆN MÔ HÌNH PHÂN LOẠI KNN")
    print("=" * 65)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Không tìm thấy file dataset tại: {dataset_path}")

    # 1. Đọc dữ liệu (bỏ qua các dòng chú thích bắt đầu bằng #)
    print(f"[1/6] Đang nạp dataset từ: {dataset_path}")
    df = pd.read_csv(dataset_path, comment="#")

    # Kiểm tra các cột bắt buộc
    expected_cols = FEATURE_NAMES + ["label"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Dataset thiếu cột bắt buộc: '{col}'. Cần có: {expected_cols}")

    initial_len = len(df)
    print(f" -> Tổng số mẫu ban đầu: {initial_len}")

    # Kiểm tra và xử lý dữ liệu thiếu (NaN)
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f" -> Phát hiện {nan_count} giá trị thiếu (NaN). Tiến hành loại bỏ...")
        df = df.dropna().reset_index(drop=True)

    # Kiểm tra dữ liệu trùng lặp
    dup_count = df.duplicated(subset=FEATURE_NAMES).sum()
    if dup_count > 0:
        print(f" -> Phát hiện {dup_count} mẫu trùng lặp đặc trưng. Tiến hành giữ lại mẫu đầu tiên...")
        df = df.drop_duplicates(subset=FEATURE_NAMES).reset_index(drop=True)

    print(f" -> Số mẫu hợp lệ sau khi làm sạch: {len(df)}")

    # Phân bố nhãn
    label_counts = df["label"].value_counts()
    print(" -> Phân bố các lớp tư thế:")
    for label, count in label_counts.items():
        print(f"    * {label:20s}: {count:3d} mẫu ({count / len(df) * 100:.1f}%)")

    # 2. Tách X và y
    X = df[FEATURE_NAMES].values
    y = df["label"].values

    # Phân tách tập Train (80%) và Test (20%) có phân tầng (stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"[2/6] Phân chia dữ liệu: Train = {len(X_train)} mẫu | Test = {len(X_test)} mẫu")

    # 3. Xây dựng Pipeline (StandardScaler + KNeighborsClassifier)
    # n_neighbors=5, weights='distance' giúp các điểm gần hơn có trọng số đóng góp lớn hơn
    print("[3/6] Thiết lập Machine Learning Pipeline: StandardScaler -> KNeighborsClassifier(k=5, weights='distance')")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance", metric="minkowski", p=2))
    ])

    # 4. Huấn luyện Pipeline
    print("[4/6] Đang huấn luyện trên tập Train...")
    pipeline.fit(X_train, y_train)

    # 5. Đánh giá chất lượng mô hình trên tập Test
    print("[5/6] Đánh giá mô hình trên tập Test độc lập:")
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)

    print("\n" + "-" * 55)
    print(f"ACCURACY (Độ chính xác tổng thể): {acc * 100:.2f}%\n")
    print("CHI TIẾT BÁO CÁO PHÂN LOẠI (Classification Report):")
    print(report_text)
    print("MA TRẬN NHẦM LẪN (Confusion Matrix):")
    cm_df = pd.DataFrame(cm, index=pipeline.classes_, columns=pipeline.classes_)
    print(cm_df)
    print("-" * 55 + "\n")

    # 6. Đóng gói mô hình kèm Metadata
    model_artifact = {
        "pipeline": pipeline,
        "feature_names": FEATURE_NAMES,
        "classes": list(pipeline.classes_),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": float(acc),
        "classification_report": report_dict,
        "k_neighbors": 5,
        "weights": "distance",
        "scaler": "StandardScaler",
        "created_at": datetime.now().isoformat()
    }

    joblib.dump(model_artifact, model_save_path)
    print(f"[6/6] Đã lưu mô hình Pipeline và Metadata thành công vào: {model_save_path}")
    print("=" * 65)
    print("HƯỚNG DẪN HUẤN LUYỆN LẠI KHI CÓ DỮ LIỆU MỚI:")
    print("1. Thu thập thêm mẫu bằng: python data/collect_data.py --label <label_name>")
    print("2. Chạy lại script này:    python data/train_knn.py")
    print("=" * 65)

    return model_artifact


if __name__ == "__main__":
    train_model()
