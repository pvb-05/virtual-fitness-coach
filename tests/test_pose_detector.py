"""
Unit Tests cho PoseDetector (Giai đoạn 2).
Kiểm tra:
- Khởi tạo MediaPipe Pose Detector
- Xử lý frame rỗng hoặc không có người
- Kiểm tra logic visibility và cảnh báo khuất chi dưới
- Vẽ khung xương và xuất dữ liệu JSON
"""

import pytest
import numpy as np
from core.pose_detector import PoseDetector, LandmarkPoint, PoseDetectionResult


@pytest.fixture(scope="module")
def detector():
    """Khởi tạo detector dùng chung cho các test trong module."""
    det = PoseDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        visibility_threshold=0.5,
        model_complexity=0  # Lite model cho test nhanh
    )
    yield det
    det.close()


def test_detector_empty_frame(detector):
    """Frame rỗng hoặc None trả về is_valid=False."""
    res_none = detector.process_frame(None)
    assert res_none.detected is False
    assert res_none.is_valid is False

    empty_mat = np.zeros((0, 0, 3), dtype=np.uint8)
    res_empty = detector.process_frame(empty_mat)
    assert res_empty.detected is False
    assert res_empty.is_valid is False


def test_detector_black_frame_no_person(detector):
    """Ảnh đen hoàn toàn không có người trả về detected=False."""
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.process_frame(black_frame)
    assert res.detected is False
    assert res.is_valid is False
    assert "Không phát hiện người" in res.error_message


def test_landmark_point_namedtuple():
    """Kiểm tra cấu trúc dữ liệu LandmarkPoint."""
    p = LandmarkPoint(x=0.5, y=0.8, z=-0.1, visibility=0.95)
    assert p.x == 0.5
    assert p.y == 0.8
    assert p.z == -0.1
    assert p.visibility == 0.95


def test_pose_detection_result_to_dict():
    """Kiểm tra serialize kết quả thành dictionary phục vụ WebSocket."""
    lm = LandmarkPoint(x=0.45678, y=0.87654, z=0.0, visibility=0.987)
    res = PoseDetectionResult(
        detected=True,
        is_valid=True,
        landmarks=[lm],
        frame_width=640,
        frame_height=480
    )
    d = res.to_dict()
    assert d["detected"] is True
    assert d["is_valid"] is True
    assert d["landmark_count"] == 1
    assert d["keypoints"][0]["x"] == 0.4568
    assert d["keypoints"][0]["y"] == 0.8765


def test_draw_skeleton_preserves_shape(detector):
    """Hàm draw_skeleton trả về ảnh cùng kích thước với ảnh gốc."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Tạo 33 landmarks giả lập có visibility cao
    mock_lms = [
        LandmarkPoint(x=0.5, y=0.1 * i, z=0.0, visibility=0.9)
        for i in range(33)
    ]
    res = PoseDetectionResult(
        detected=True,
        is_valid=True,
        landmarks=mock_lms,
        frame_width=640,
        frame_height=480
    )
    
    annotated = detector.draw_skeleton(img, res)
    assert annotated.shape == img.shape
    assert annotated.dtype == img.dtype
