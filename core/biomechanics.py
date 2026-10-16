"""
Core Module: Biomechanics & Feature Extraction
Tính toán góc khớp cơ sinh học bằng vector và tích vô hướng:
- Góc gối trái / phải (knee angles)
- Góc hông trái / phải (hip angles)
- Độ nghiêng thân trên so với phương thẳng đứng (torso angle)
- Độ thẳng hàng đầu gối / chụm gối (knee alignment / valgus metrics)

Cung cấp bộ 7 đặc trưng thống nhất dùng xuyên suốt quá trình Huấn luyện (Train) và Dự đoán (Inference).
"""

import math
from typing import Dict, List, Tuple, Union, Sequence, Any
import numpy as np

# Danh sách 7 đặc trưng chuẩn hóa bắt buộc phải thống nhất giữa CSV, train_knn.py và pose_classifier.py
FEATURE_NAMES = [
    "knee_angle_left",
    "knee_angle_right",
    "hip_angle_left",
    "hip_angle_right",
    "torso_angle",
    "knee_alignment_left",
    "knee_alignment_right"
]

# Chỉ số landmarks MediaPipe Pose
LANDMARK_IDX = {
    "NOSE": 0,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
}


def _get_point_coords(point: Any) -> Tuple[float, float]:
    """Trích xuất tọa độ (x, y) từ tuple, list, dict hoặc LandmarkPoint object."""
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(point.x), float(point.y)
    elif isinstance(point, dict) and "x" in point and "y" in point:
        return float(point["x"]), float(point["y"])
    elif isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
        return float(point[0]), float(point[1])
    raise ValueError(f"Không thể trích xuất tọa độ (x, y) từ đối tượng: {type(point)}")


def calculate_angle_3p(
    p1: Any,
    p2: Any,
    p3: Any,
    eps: float = 1e-7
) -> float:
    """
    Tính góc tạo bởi 3 điểm (p1, p2, p3) với p2 là đỉnh góc (vertex).
    Sử dụng công thức tích vô hướng vector:
        cos(theta) = dot(u, v) / (||u|| * ||v||)
    trong đó:
        u = p1 - p2
        v = p3 - p2

    :param p1: Điểm đầu (ví dụ: hông)
    :param p2: Đỉnh góc (ví dụ: gối)
    :param p3: Điểm cuối (ví dụ: cổ chân)
    :param eps: Ngưỡng kiểm tra vector 0 để tránh chia cho 0
    :return: Góc theo độ (degrees), nằm trong khoảng [0.0, 180.0]
    """
    x1, y1 = _get_point_coords(p1)
    x2, y2 = _get_point_coords(p2)
    x3, y3 = _get_point_coords(p3)

    # Vector u: từ đỉnh p2 đến p1
    ux = x1 - x2
    uy = y1 - y2

    # Vector v: từ đỉnh p2 đến p3
    vx = x3 - x2
    vy = y3 - y2

    # Độ dài 2 vector (norm)
    norm_u = math.sqrt(ux * ux + uy * uy)
    norm_v = math.sqrt(vx * vx + vy * vy)

    # Xử lý trường hợp điểm trùng nhau hoặc vector độ dài bằng 0
    if norm_u < eps or norm_v < eps:
        return 0.0

    # Tích vô hướng (dot product)
    dot_product = ux * vx + uy * vy

    # Tính cosine và giới hạn nghiêm ngặt trong đoạn [-1.0, 1.0] để chống lỗi làm tròn dấu phẩy động
    cos_theta = dot_product / (norm_u * norm_v)
    cos_theta = max(-1.0, min(1.0, cos_theta))

    # Tính góc arccos và đổi sang độ
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)

    return float(round(angle_deg, 2))


def calculate_torso_angle(
    left_shoulder: Any,
    right_shoulder: Any,
    left_hip: Any,
    right_hip: Any,
    eps: float = 1e-7
) -> float:
    """
    Tính độ nghiêng của thân trên so với phương thẳng đứng hướng lên (upward vertical).
    Trục đứng trong hệ tọa độ ảnh có vector hướng lên là (0, -1) vì trục Y hướng xuống dưới.

    :return: Góc nghiêng thân trên (0 độ là đứng thẳng lưng, 30-45 độ là nghiêng người squat)
    """
    ls_x, ls_y = _get_point_coords(left_shoulder)
    rs_x, rs_y = _get_point_coords(right_shoulder)
    lh_x, lh_y = _get_point_coords(left_hip)
    rh_x, rh_y = _get_point_coords(right_hip)

    # Điểm giữa hông (mid-hip) và điểm giữa vai (mid-shoulder)
    mid_hip_x = (lh_x + rh_x) / 2.0
    mid_hip_y = (lh_y + rh_y) / 2.0

    mid_shoulder_x = (ls_x + rs_x) / 2.0
    mid_shoulder_y = (ls_y + rs_y) / 2.0

    # Vector thân trên từ mid-hip hướng lên mid-shoulder
    torso_x = mid_shoulder_x - mid_hip_x
    torso_y = mid_shoulder_y - mid_hip_y

    torso_norm = math.sqrt(torso_x * torso_x + torso_y * torso_y)
    if torso_norm < eps:
        return 0.0

    # Vector thẳng đứng hướng lên trong tọa độ màn hình: u_vert = (0, -1), norm = 1
    # dot = torso_x * 0 + torso_y * (-1) = -torso_y
    dot_product = -torso_y
    cos_theta = dot_product / torso_norm
    cos_theta = max(-1.0, min(1.0, cos_theta))

    angle_deg = math.degrees(math.acos(cos_theta))
    return float(round(angle_deg, 2))


def calculate_knee_alignment(
    left_hip: Any,
    right_hip: Any,
    left_knee: Any,
    right_knee: Any,
    left_ankle: Any,
    right_ankle: Any,
    eps: float = 1e-4
) -> Tuple[float, float]:
    """
    Tính chỉ số thẳng hàng / độ chụm đầu gối (knee alignment / knee valgus metric).
    Chuẩn hóa theo khoảng cách giữa 2 hông (hip width) để kết quả không bị ảnh hưởng bởi
    khoảng cách từ người tập tới camera (scale-invariant).

    Định nghĩa đối xứng:
    - knee_alignment_left: (knee_left.x - ankle_left.x) / hip_width
    - knee_alignment_right: (ankle_right.x - knee_right.x) / hip_width
    Khi đầu gối chụm vào trong (valgus), cả 2 chỉ số này sẽ tăng dương rõ rệt.

    :return: (knee_alignment_left, knee_alignment_right)
    """
    lh_x, _ = _get_point_coords(left_hip)
    rh_x, _ = _get_point_coords(right_hip)
    lk_x, _ = _get_point_coords(left_knee)
    rk_x, _ = _get_point_coords(right_knee)
    la_x, _ = _get_point_coords(left_ankle)
    ra_x, _ = _get_point_coords(right_ankle)

    hip_width = max(abs(rh_x - lh_x), eps)

    align_left = (lk_x - la_x) / hip_width
    align_right = (ra_x - rk_x) / hip_width

    return float(round(align_left, 4)), float(round(align_right, 4))


def extract_squat_features(landmarks: Union[List[Any], Dict[str, Any]]) -> Dict[str, float]:
    """
    Trích xuất toàn bộ 7 đặc trưng cơ sinh học cho bài tập Squat từ danh sách landmarks.
    :param landmarks: Danh sách 33 landmarks từ MediaPipe Pose
    :return: Dictionary chứa 7 đặc trưng:
        - knee_angle_left
        - knee_angle_right
        - hip_angle_left
        - hip_angle_right
        - torso_angle
        - knee_alignment_left
        - knee_alignment_right
    """
    # Lấy các điểm cần thiết
    p_ls = landmarks[LANDMARK_IDX["LEFT_SHOULDER"]]
    p_rs = landmarks[LANDMARK_IDX["RIGHT_SHOULDER"]]
    p_lh = landmarks[LANDMARK_IDX["LEFT_HIP"]]
    p_rh = landmarks[LANDMARK_IDX["RIGHT_HIP"]]
    p_lk = landmarks[LANDMARK_IDX["LEFT_KNEE"]]
    p_rk = landmarks[LANDMARK_IDX["RIGHT_KNEE"]]
    p_la = landmarks[LANDMARK_IDX["LEFT_ANKLE"]]
    p_ra = landmarks[LANDMARK_IDX["RIGHT_ANKLE"]]

    # 1. Góc khớp gối (Hông - Gối - Cổ chân)
    knee_angle_left = calculate_angle_3p(p_lh, p_lk, p_la)
    knee_angle_right = calculate_angle_3p(p_rh, p_rk, p_ra)

    # 2. Góc khớp hông (Vai - Hông - Gối)
    hip_angle_left = calculate_angle_3p(p_ls, p_lh, p_lk)
    hip_angle_right = calculate_angle_3p(p_rs, p_rh, p_rk)

    # 3. Góc nghiêng thân trên so với phương thẳng đứng
    torso_angle = calculate_torso_angle(p_ls, p_rs, p_lh, p_rh)

    # 4. Độ thẳng hàng đầu gối (Valgus indicator)
    valgus_left, valgus_right = calculate_knee_alignment(p_lh, p_rh, p_lk, p_rk, p_la, p_ra)

    return {
        "knee_angle_left": knee_angle_left,
        "knee_angle_right": knee_angle_right,
        "hip_angle_left": hip_angle_left,
        "hip_angle_right": hip_angle_right,
        "torso_angle": torso_angle,
        "knee_alignment_left": valgus_left,
        "knee_alignment_right": valgus_right
    }


def features_to_vector(features_dict: Dict[str, float]) -> np.ndarray:
    """
    Chuyển đổi dictionary 7 đặc trưng thành vector 1D numpy array theo đúng thứ tự FEATURE_NAMES.
    Đảm bảo tính nhất quán tuyệt đối giữa dữ liệu huấn luyện CSV và pipeline dự đoán.
    :param features_dict: Từ điển chứa 7 đặc trưng
    :return: np.ndarray shape (7,) kiểu float64
    """
    missing = [f for f in FEATURE_NAMES if f not in features_dict]
    if missing:
        raise ValueError(f"Thiếu các đặc trưng bắt buộc trong vector: {missing}")

    vector = [features_dict[name] for name in FEATURE_NAMES]
    return np.array(vector, dtype=np.float64)


def validate_feature_vector(vector: np.ndarray) -> bool:
    """Kiểm tra vector đặc trưng có đúng kích thước (7,) và không chứa NaN hay vô cùng."""
    if not isinstance(vector, np.ndarray):
        return False
    if vector.shape != (len(FEATURE_NAMES),):
        return False
    if np.isnan(vector).any() or np.isinf(vector).any():
        return False
    return True
