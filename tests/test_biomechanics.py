"""
Unit Tests cho Module Biomechanics & Feature Extraction (Giai đoạn 2).
Kiểm tra độ chính xác toán học của các phép tính góc vector,
xử lý trường hợp biên (edge cases) và tính nhất quán của bộ 7 đặc trưng.
"""

import math
import pytest
import numpy as np
from core.biomechanics import (
    calculate_angle_3p,
    calculate_torso_angle,
    calculate_knee_alignment,
    extract_squat_features,
    features_to_vector,
    validate_feature_vector,
    FEATURE_NAMES,
    LANDMARK_IDX
)


class DummyPoint:
    """Mock đối tượng landmark có thuộc tính x, y tương tự MediaPipe."""
    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def test_calculate_angle_right_angle():
    """Góc vuông 90 độ: (0, 1) - (0, 0) - (1, 0)."""
    p1 = (0.0, 1.0)
    p2 = (0.0, 0.0)  # vertex
    p3 = (1.0, 0.0)
    angle = calculate_angle_3p(p1, p2, p3)
    assert pytest.approx(angle, abs=1e-2) == 90.0


def test_calculate_angle_straight_line():
    """Góc bẹt 180 độ: đường thẳng (-1, 0) - (0, 0) - (1, 0)."""
    p1 = (-1.0, 0.0)
    p2 = (0.0, 0.0)
    p3 = (1.0, 0.0)
    angle = calculate_angle_3p(p1, p2, p3)
    assert pytest.approx(angle, abs=1e-2) == 180.0


def test_calculate_angle_zero_degrees():
    """Góc 0 độ: hai tia trùng nhau (1, 1) - (0, 0) - (2, 2)."""
    p1 = (1.0, 1.0)
    p2 = (0.0, 0.0)
    p3 = (2.0, 2.0)
    angle = calculate_angle_3p(p1, p2, p3)
    assert pytest.approx(angle, abs=1e-2) == 0.0


def test_calculate_angle_45_degrees():
    """Góc 45 độ: (1, 0) - (0, 0) - (1, 1)."""
    p1 = (1.0, 0.0)
    p2 = (0.0, 0.0)
    p3 = (1.0, 1.0)
    angle = calculate_angle_3p(p1, p2, p3)
    assert pytest.approx(angle, abs=1e-2) == 45.0


def test_calculate_angle_zero_length_vector():
    """Điểm trùng đỉnh góc (chiều dài vector = 0) trả về 0.0 không crash."""
    p1 = (0.0, 0.0)
    p2 = (0.0, 0.0)  # trùng p1
    p3 = (1.0, 1.0)
    angle = calculate_angle_3p(p1, p2, p3)
    assert angle == 0.0


def test_calculate_angle_with_dummy_objects_and_dicts():
    """Hàm chấp nhận cả object có .x/.y và dict."""
    p1 = DummyPoint(0.0, 1.0)
    p2 = {"x": 0.0, "y": 0.0}
    p3 = [1.0, 0.0]
    angle = calculate_angle_3p(p1, p2, p3)
    assert pytest.approx(angle, abs=1e-2) == 90.0


def test_calculate_torso_angle_upright():
    """Đứng thẳng: vai nằm ngay phía trên hông -> góc nghiêng 0 độ."""
    ls = DummyPoint(0.4, 0.2)
    rs = DummyPoint(0.6, 0.2)
    lh = DummyPoint(0.4, 0.6)
    rh = DummyPoint(0.6, 0.6)
    angle = calculate_torso_angle(ls, rs, lh, rh)
    assert pytest.approx(angle, abs=1e-2) == 0.0


def test_calculate_torso_angle_leaning():
    """Thân trên nghiêng về phía trước: mid-shoulder lệch Y và X so với mid-hip."""
    # Mid hip ở (0.5, 0.6), Mid shoulder ở (0.7, 0.4)
    # Vector: (0.2, -0.2). So với (0, -1) góc là 45 độ
    ls = DummyPoint(0.6, 0.4)
    rs = DummyPoint(0.8, 0.4)
    lh = DummyPoint(0.4, 0.6)
    rh = DummyPoint(0.6, 0.6)
    angle = calculate_torso_angle(ls, rs, lh, rh)
    assert pytest.approx(angle, abs=1e-2) == 45.0


def test_calculate_knee_alignment_scale_invariance():
    """Chỉ số knee alignment không đổi khi tỷ lệ kích thước thay đổi (scale-invariant)."""
    # Tư thế chuẩn 1:
    lh1, rh1 = DummyPoint(0.4, 0.5), DummyPoint(0.6, 0.5)
    lk1, rk1 = DummyPoint(0.4, 0.7), DummyPoint(0.6, 0.7)
    la1, ra1 = DummyPoint(0.4, 0.9), DummyPoint(0.6, 0.9)
    al1, ar1 = calculate_knee_alignment(lh1, rh1, lk1, rk1, la1, ra1)

    # Phóng to gấp 2 lần:
    lh2, rh2 = DummyPoint(0.8, 1.0), DummyPoint(1.2, 1.0)
    lk2, rk2 = DummyPoint(0.8, 1.4), DummyPoint(1.2, 1.4)
    la2, ra2 = DummyPoint(0.8, 1.8), DummyPoint(1.2, 1.8)
    al2, ar2 = calculate_knee_alignment(lh2, rh2, lk2, rk2, la2, ra2)

    assert pytest.approx(al1, abs=1e-3) == al2
    assert pytest.approx(ar1, abs=1e-3) == ar2


def test_calculate_knee_valgus_increases_when_caving_in():
    """Khi đầu gối lệch vào trong (valgus), chỉ số alignment tăng lên."""
    lh, rh = DummyPoint(0.4, 0.5), DummyPoint(0.6, 0.5)
    la, ra = DummyPoint(0.4, 0.9), DummyPoint(0.6, 0.9)

    # Đầu gối thẳng hàng
    lk_normal, rk_normal = DummyPoint(0.4, 0.7), DummyPoint(0.6, 0.7)
    norm_left, norm_right = calculate_knee_alignment(lh, rh, lk_normal, rk_normal, la, ra)

    # Đầu gối chụm vào giữa (lk dịch phải, rk dịch trái)
    lk_valgus, rk_valgus = DummyPoint(0.45, 0.7), DummyPoint(0.55, 0.7)
    valg_left, valg_right = calculate_knee_alignment(lh, rh, lk_valgus, rk_valgus, la, ra)

    assert valg_left > norm_left
    assert valg_right > norm_right


def test_extract_squat_features_completeness():
    """Kiểm tra trích xuất đầy đủ 7 đặc trưng chuẩn hóa từ danh sách landmarks."""
    # Tạo danh sách 33 landmarks giả lập
    mock_landmarks = [DummyPoint(0.5, 0.5) for _ in range(33)]
    
    # Gán các tọa độ mô phỏng tư thế đứng thẳng
    mock_landmarks[LANDMARK_IDX["LEFT_SHOULDER"]] = DummyPoint(0.45, 0.2)
    mock_landmarks[LANDMARK_IDX["RIGHT_SHOULDER"]] = DummyPoint(0.55, 0.2)
    mock_landmarks[LANDMARK_IDX["LEFT_HIP"]] = DummyPoint(0.45, 0.5)
    mock_landmarks[LANDMARK_IDX["RIGHT_HIP"]] = DummyPoint(0.55, 0.5)
    mock_landmarks[LANDMARK_IDX["LEFT_KNEE"]] = DummyPoint(0.45, 0.75)
    mock_landmarks[LANDMARK_IDX["RIGHT_KNEE"]] = DummyPoint(0.55, 0.75)
    mock_landmarks[LANDMARK_IDX["LEFT_ANKLE"]] = DummyPoint(0.45, 0.95)
    mock_landmarks[LANDMARK_IDX["RIGHT_ANKLE"]] = DummyPoint(0.55, 0.95)

    feats = extract_squat_features(mock_landmarks)

    assert isinstance(feats, dict)
    assert len(feats) == 7
    for key in FEATURE_NAMES:
        assert key in feats
        assert isinstance(feats[key], float)
        assert not math.isnan(feats[key])

    # Khi đứng thẳng, góc gối xấp xỉ 180 độ
    assert pytest.approx(feats["knee_angle_left"], abs=1.0) == 180.0
    assert pytest.approx(feats["knee_angle_right"], abs=1.0) == 180.0


def test_features_to_vector_and_validation():
    """Kiểm tra chuyển đổi sang numpy vector và xác thực."""
    sample_features = {
        "knee_angle_left": 175.2,
        "knee_angle_right": 174.8,
        "hip_angle_left": 168.0,
        "hip_angle_right": 169.1,
        "torso_angle": 8.5,
        "knee_alignment_left": 0.02,
        "knee_alignment_right": 0.01
    }

    vec = features_to_vector(sample_features)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (7,)
    assert validate_feature_vector(vec) is True

    # Vector sai độ dài hoặc chứa NaN
    assert validate_feature_vector(np.array([1.0, 2.0])) is False
    assert validate_feature_vector(np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0])) is False

    # Thiếu key báo lỗi
    with pytest.raises(ValueError, match="Thiếu các đặc trưng bắt buộc"):
        features_to_vector({"knee_angle_left": 170.0})
