"""
Unit Tests cho PoseClassifier (Giai đoạn 4).
Kiểm tra:
- Khởi tạo và nạp model KNN đã huấn luyện từ data/knn_model.pkl
- Xử lý lỗi nạp khi file model không tồn tại
- Phân loại tư thế Chuẩn (correct)
- Nhận diện lỗi chưa đủ độ sâu (insufficient_depth)
- Nhận diện lỗi chụm gối vào trong (knee_valgus)
- Xử lý khi vector đặc trưng không hợp lệ
- Cơ chế gán nhãn 'unknown' khi độ tự tin dưới ngưỡng
"""

import os
import pytest
import numpy as np
from core.pose_classifier import PoseClassifier, ClassificationResult

MODEL_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "knn_model.pkl")


@pytest.fixture(scope="module")
def classifier():
    """Khởi tạo PoseClassifier với model thật đã train."""
    clf = PoseClassifier(model_path=MODEL_FILE, min_confidence_threshold=0.55)
    return clf


def test_model_loaded_successfully(classifier):
    """Kiểm tra model được nạp thành công và có đủ 3 lớp nhãn."""
    assert classifier.is_loaded is True
    assert classifier.pipeline is not None
    assert set(classifier.classes) == {"correct", "insufficient_depth", "knee_valgus"}


def test_missing_model_fallback():
    """Khi file model không tồn tại, hệ thống không crash mà trả về 'unknown' an toàn."""
    clf = PoseClassifier(model_path="data/duong_dan_khong_ton_tai.pkl")
    assert clf.is_loaded is False

    sample_feat = {
        "knee_angle_left": 85.0, "knee_angle_right": 85.0,
        "hip_angle_left": 85.0, "hip_angle_right": 85.0,
        "torso_angle": 30.0, "knee_alignment_left": 0.0, "knee_alignment_right": 0.0
    }
    res = clf.classify(sample_feat)
    assert res.label == "unknown"
    assert res.is_reliable is False
    assert res.error_code == "ERR_MODEL_UNAVAILABLE"


def test_classify_correct_squat(classifier):
    """Mẫu tư thế chuẩn: gối ~85 độ, hông ~88 độ, thân nghiêng 30 độ, gối thẳng hàng."""
    sample = {
        "knee_angle_left": 84.5,
        "knee_angle_right": 85.0,
        "hip_angle_left": 87.0,
        "hip_angle_right": 87.5,
        "torso_angle": 31.0,
        "knee_alignment_left": 0.01,
        "knee_alignment_right": 0.02
    }
    res = classifier.classify(sample)
    assert res.label == "correct"
    assert res.is_reliable is True
    assert res.error_code is None
    assert res.confidence >= 0.70
    assert "chuẩn xác" in res.description_vi


def test_classify_insufficient_depth(classifier):
    """Mẫu lỗi squat nông: gối ~120 độ, hông ~125 độ."""
    sample = {
        "knee_angle_left": 120.0,
        "knee_angle_right": 119.5,
        "hip_angle_left": 124.0,
        "hip_angle_right": 123.5,
        "torso_angle": 16.0,
        "knee_alignment_left": 0.01,
        "knee_alignment_right": 0.01
    }
    res = classifier.classify(sample)
    assert res.label == "insufficient_depth"
    assert res.is_reliable is True
    assert res.error_code == "ERR_DEPTH"
    assert "chưa đủ độ sâu" in res.description_vi.lower()


def test_classify_knee_valgus(classifier):
    """Mẫu lỗi chụm gối: gối ~86 độ, alignment_left = 0.24, alignment_right = 0.25."""
    sample = {
        "knee_angle_left": 86.0,
        "knee_angle_right": 85.5,
        "hip_angle_left": 88.0,
        "hip_angle_right": 87.5,
        "torso_angle": 32.0,
        "knee_alignment_left": 0.24,
        "knee_alignment_right": 0.25
    }
    res = classifier.classify(sample)
    assert res.label == "knee_valgus"
    assert res.is_reliable is True
    assert res.error_code == "ERR_VALGUS"
    assert "chụm" in res.description_vi.lower() or "valgus" in res.description_vi.lower()


def test_invalid_features_handling(classifier):
    """Đặc trưng thiếu hoặc chứa NaN trả về 'unknown' an toàn."""
    # Thiếu key
    res_missing = classifier.classify({"knee_angle_left": 85.0})
    assert res_missing.label == "unknown"
    assert res_missing.error_code == "ERR_INVALID_FEATURES"

    # Chứa NaN
    nan_vector = np.array([85.0, np.nan, 85.0, 85.0, 30.0, 0.0, 0.0])
    res_nan = classifier.classify(nan_vector)
    assert res_nan.label == "unknown"
    assert res_nan.error_code == "ERR_INVALID_FEATURES"


def test_classification_result_to_dict():
    """Kiểm tra serialize kết quả sang dictionary."""
    res = ClassificationResult(
        label="correct",
        confidence=0.92345,
        is_reliable=True,
        error_code=None,
        description_vi="Tư thế tốt",
        probabilities={"correct": 0.92, "insufficient_depth": 0.08}
    )
    d = res.to_dict()
    assert d["label"] == "correct"
    assert pytest.approx(d["confidence"], abs=1e-3) == 0.923
    assert d["is_reliable"] is True
    assert d["probabilities"]["correct"] == 0.92
