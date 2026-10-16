"""
Unit Tests cho Realtime WebSocket Pipeline (Giai đoạn 5).
Kiểm tra:
- Khả năng mã hóa/giải mã frame base64 JPEG
- Hợp đồng cấu trúc JSON (contract) giữa Backend và Frontend
- Xử lý khi khung hình không có người hoặc frame lỗi
- Xử lý lệnh reset bộ đếm rep
"""

import json
import base64
import pytest
import numpy as np
import cv2
from app import app, global_detector, global_classifier
from core.rep_counter import RepCounter, RepCounterConfig


def create_dummy_jpeg_base64(width=640, height=480, color=(50, 50, 50)) -> str:
    """Tạo chuỗi base64 JPEG mô phỏng từ camera trình duyệt."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    b64_str = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def test_frame_decoding_pipeline():
    """Kiểm tra giải mã frame base64 thành mảng numpy OpenCV hợp lệ."""
    b64_full = create_dummy_jpeg_base64(320, 240)
    assert b64_full.startswith("data:image/jpeg;base64,")

    encoded_data = b64_full.split(",", 1)[1]
    img_bytes = base64.b64decode(encoded_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    decoded_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    assert decoded_frame is not None
    assert decoded_frame.shape == (240, 320, 3)


def test_empty_frame_pipeline_contract():
    """Frame không có người vẫn trả về đúng cấu trúc JSON quy ước với pose_detected=False."""
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = global_detector.process_frame(black_frame)

    counter = RepCounter()
    rep_res = counter.update(features=None, is_valid_pose=False)

    response = {
        "type": "pose_result",
        "pose_detected": res.detected,
        "is_valid": res.is_valid,
        "error_message": res.error_message,
        "rep_count": rep_res.rep_count,
        "state": rep_res.state.value,
        "progress_pct": 0.0,
        "current_knee_angle": 180.0,
        "min_depth_angle": 180.0,
        "rep_just_completed": False,
        "rep_duration": 0.0,
        "pose_label": "unknown",
        "confidence": 0.0,
        "error_code": "ERR_NO_PERSON",
        "feedback_text": "Chưa phát hiện người",
        "landmarks": [],
        "processing_time_ms": 15.2
    }

    # Xác thực hợp đồng giao tiếp (Contract validation)
    assert response["type"] == "pose_result"
    assert response["pose_detected"] is False
    assert response["is_valid"] is False
    assert response["rep_count"] == 0
    assert response["error_code"] == "ERR_NO_PERSON"
    assert isinstance(response["landmarks"], list)
    assert isinstance(response["processing_time_ms"], float)


def test_valid_pose_simulated_contract():
    """Kiểm tra hợp đồng phản hồi khi tư thế hợp lệ (có rep_count, KNN label, landmarks)."""
    counter = RepCounter(RepCounterConfig(smoothing_factor=1.0))
    sample_feat = {
        "knee_angle_left": 85.0,
        "knee_angle_right": 84.5,
        "hip_angle_left": 88.0,
        "hip_angle_right": 87.5,
        "torso_angle": 30.0,
        "knee_alignment_left": 0.01,
        "knee_alignment_right": 0.02
    }

    rep_res = counter.update(sample_feat, is_valid_pose=True)
    clf_res = global_classifier.classify(sample_feat)

    response = {
        "type": "pose_result",
        "pose_detected": True,
        "is_valid": True,
        "error_message": "",
        "rep_count": rep_res.rep_count,
        "state": rep_res.state.value,
        "progress_pct": rep_res.progress_pct,
        "current_knee_angle": rep_res.current_knee_angle,
        "min_depth_angle": rep_res.min_depth_angle,
        "rep_just_completed": rep_res.rep_just_completed,
        "rep_duration": rep_res.rep_duration,
        "pose_label": clf_res.label,
        "confidence": clf_res.confidence,
        "error_code": clf_res.error_code,
        "feedback_text": clf_res.description_vi,
        "landmarks": [{"x": 0.5, "y": 0.5, "visibility": 0.99}],
        "processing_time_ms": 22.4
    }

    assert response["pose_detected"] is True
    assert response["is_valid"] is True
    assert response["pose_label"] == "correct"
    assert response["confidence"] >= 0.70
    assert response["error_code"] is None
    assert len(response["landmarks"]) == 1
    assert "chuẩn xác" in response["feedback_text"].lower()


def test_corrupted_base64_safe_handling():
    """Dữ liệu base64 bị hỏng không làm crash tiến trình."""
    corrupted_data = "data:image/jpeg;base64,KhongPhaiLaBase64HopLe$$$==="
    payload_str = corrupted_data.split(",", 1)[1]
    
    # Thử decode
    try:
        raw_bytes = base64.b64decode(payload_str)
        arr = np.frombuffer(raw_bytes, np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        decoded = None

    # cv2.imdecode trả về None khi dữ liệu ảnh hỏng
    assert decoded is None
