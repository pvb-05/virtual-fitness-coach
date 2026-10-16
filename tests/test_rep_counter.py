"""
Unit Tests cho Rep Counter State Machine (Giai đoạn 3).
Kiểm tra:
- Khởi tạo và reset trạng thái ban đầu
- Chu trình squat hợp lệ đầy đủ 4 pha (STANDING -> DESCENDING -> BOTTOM -> ASCENDING -> STANDING)
- Nhiều rep liên tiếp
- Squat nông (chưa đủ độ sâu) không được cộng rep
- Giữ nguyên ở đáy squat không bị đếm nhiều lần
- Cơ chế trễ (Hysteresis) chống rung lắc tại vùng chuyển trạng thái
- Xử lý mất landmarks an toàn giữa chừng
- Cấu hình ngưỡng góc tùy biến
"""

import pytest
from core.rep_counter import RepCounter, RepCounterConfig, SquatState


def make_features(knee_angle: float) -> dict:
    """Hàm phụ trợ tạo dictionary features với góc gối mong muốn."""
    return {
        "knee_angle_left": knee_angle,
        "knee_angle_right": knee_angle,
        "hip_angle_left": 160.0,
        "hip_angle_right": 160.0,
        "torso_angle": 15.0,
        "knee_alignment_left": 0.0,
        "knee_alignment_right": 0.0
    }


def test_initial_state():
    """Trạng thái ban đầu luôn là STANDING với rep_count = 0."""
    counter = RepCounter()
    assert counter.state == SquatState.STANDING
    assert counter.rep_count == 0
    assert counter.has_reached_bottom is False


def test_valid_single_squat_cycle():
    """Một chu trình squat hợp lệ đầy đủ tăng rep_count lên 1."""
    # Tắt smoothing (smoothing_factor = 1.0) để kiểm tra chính xác giá trị góc từng bước
    cfg = RepCounterConfig(smoothing_factor=1.0)
    counter = RepCounter(cfg)

    # 1. Đang đứng thẳng (170 độ)
    res = counter.update(make_features(170.0), is_valid_pose=True)
    assert res.state == SquatState.STANDING
    assert res.rep_count == 0

    # 2. Bắt đầu hạ người (135 độ < descending_angle 140)
    res = counter.update(make_features(135.0), is_valid_pose=True)
    assert res.state == SquatState.DESCENDING
    assert res.rep_count == 0

    # 3. Đạt đáy squat chuẩn (88 độ <= bottom_angle 95)
    res = counter.update(make_features(88.0), is_valid_pose=True)
    assert res.state == SquatState.BOTTOM
    assert res.rep_count == 0
    assert counter.has_reached_bottom is True

    # 4. Đang đẩy người lên (120 độ >= ascending_angle 115)
    res = counter.update(make_features(120.0), is_valid_pose=True)
    assert res.state == SquatState.ASCENDING
    assert res.rep_count == 0

    # 5. Hoàn thành trở về tư thế đứng (165 độ >= standing_complete_angle 155)
    res = counter.update(make_features(165.0), is_valid_pose=True)
    assert res.state == SquatState.STANDING
    assert res.rep_count == 1
    assert res.rep_just_completed is True
    assert res.feedback_event == "REP_COMPLETED"


def test_multiple_consecutive_reps():
    """Thực hiện 3 lần squat chuẩn liên tiếp, rep_count tăng chính xác lên 3."""
    cfg = RepCounterConfig(smoothing_factor=1.0)
    counter = RepCounter(cfg)

    angles_per_rep = [170.0, 130.0, 85.0, 125.0, 165.0]

    for expected_count in range(1, 4):
        for angle in angles_per_rep:
            res = counter.update(make_features(angle), is_valid_pose=True)
        assert counter.rep_count == expected_count

    assert counter.rep_count == 3


def test_incomplete_squat_depth_no_count():
    """Squat không đủ độ sâu (chỉ xuống tới 110 độ) thì KHÔNG được cộng rep."""
    cfg = RepCounterConfig(smoothing_factor=1.0)
    counter = RepCounter(cfg)

    # Đứng thẳng -> Hạ người -> Chưa chạm 95 độ đã đứng lên
    counter.update(make_features(170.0))
    counter.update(make_features(135.0))
    assert counter.state == SquatState.DESCENDING

    counter.update(make_features(110.0))  # Chưa chạm đáy
    assert counter.state == SquatState.DESCENDING
    assert counter.has_reached_bottom is False

    # Đứng thẳng dậy sớm
    res = counter.update(make_features(165.0))
    assert res.state == SquatState.STANDING
    assert res.rep_count == 0
    assert res.rep_just_completed is False
    assert res.feedback_event == "INCOMPLETE_DEPTH"


def test_holding_at_bottom_does_not_multiply_reps():
    """Giữ nguyên ở đáy squat trong nhiều frames KHÔNG được cộng rep nhiều lần."""
    cfg = RepCounterConfig(smoothing_factor=1.0)
    counter = RepCounter(cfg)

    counter.update(make_features(170.0))
    counter.update(make_features(135.0))
    counter.update(make_features(90.0))
    assert counter.state == SquatState.BOTTOM
    assert counter.rep_count == 0

    # Giữ ở đáy 20 frames liên tiếp
    for bottom_jitter in [90.0, 89.0, 92.0, 88.0, 93.0] * 4:
        res = counter.update(make_features(bottom_jitter))
        assert res.state == SquatState.BOTTOM
        assert res.rep_count == 0

    # Sau đó mới đứng lên
    counter.update(make_features(120.0))
    res = counter.update(make_features(165.0))
    assert res.rep_count == 1


def test_hysteresis_buffer_at_bottom():
    """Hysteresis ngăn dao động giữa BOTTOM và ASCENDING khi góc biến động nhỏ quanh đáy."""
    cfg = RepCounterConfig(bottom_angle=95.0, ascending_angle=115.0, smoothing_factor=1.0)
    counter = RepCounter(cfg)

    counter.update(make_features(170.0))
    counter.update(make_features(130.0))
    counter.update(make_features(92.0))
    assert counter.state == SquatState.BOTTOM

    # Góc dao động từ 96 -> 105 -> 100 độ (vẫn < ascending_angle 115)
    counter.update(make_features(98.0))
    assert counter.state == SquatState.BOTTOM
    counter.update(make_features(108.0))
    assert counter.state == SquatState.BOTTOM

    # Khi vượt hẳn 115 độ mới chuyển sang ASCENDING
    counter.update(make_features(116.0))
    assert counter.state == SquatState.ASCENDING


def test_lost_landmarks_timeout_resets_safely():
    """Mất tín hiệu camera quá max_lost_frames_reset sẽ tự động hủy rep dở dang."""
    cfg = RepCounterConfig(max_lost_frames_reset=5, smoothing_factor=1.0)
    counter = RepCounter(cfg)

    # Đang hạ người
    counter.update(make_features(170.0))
    counter.update(make_features(135.0))
    assert counter.state == SquatState.DESCENDING

    # Mất tín hiệu 4 frames (chưa quá 5) -> vẫn giữ trạng thái
    for _ in range(4):
        counter.update(features=None, is_valid_pose=False)
    assert counter.state == SquatState.DESCENDING

    # Mất tín hiệu frame thứ 5 -> reset về STANDING an toàn
    res = counter.update(features=None, is_valid_pose=False)
    assert res.state == SquatState.STANDING
    assert res.rep_count == 0


def test_custom_config_thresholds():
    """Kiểm tra thay đổi cấu hình ngưỡng góc đáy squat sâu hơn (80 độ)."""
    deep_cfg = RepCounterConfig(bottom_angle=80.0, smoothing_factor=1.0)
    counter = RepCounter(deep_cfg)

    counter.update(make_features(170.0))
    counter.update(make_features(130.0))
    
    # 90 độ chưa chạm đáy với cấu hình squat sâu (80 độ)
    counter.update(make_features(90.0))
    assert counter.state == SquatState.DESCENDING

    # 78 độ mới chạm đáy
    counter.update(make_features(78.0))
    assert counter.state == SquatState.BOTTOM


def test_progress_percentage():
    """Kiểm tra tính toán thanh tiến độ squat 0% - 100%."""
    cfg = RepCounterConfig(standing_angle=160.0, bottom_angle=100.0, smoothing_factor=1.0)
    counter = RepCounter(cfg)

    # Đứng thẳng 160 độ -> 0%
    res = counter.update(make_features(160.0))
    assert pytest.approx(res.progress_pct, abs=0.1) == 0.0

    # Hạ người 130 độ -> đúng 50%
    res = counter.update(make_features(130.0))
    assert pytest.approx(res.progress_pct, abs=0.1) == 50.0

    # Đạt đáy 100 độ -> 100%
    res = counter.update(make_features(100.0))
    assert pytest.approx(res.progress_pct, abs=0.1) == 100.0
