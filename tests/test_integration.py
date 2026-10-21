"""
End-to-End Integration Tests cho Toàn bộ Hệ thống Virtual Fitness Coach (Giai đoạn 8).
Kiểm thử tích hợp luồng hoàn chỉnh:
1. Định tuyến giao diện & Health Check API.
2. Luồng chu trình Squat đầy đủ: Trích xuất góc -> Đếm Rep (FSM) -> Phân loại (KNN) -> Bắt lỗi kéo dài -> LLM Agent -> TTS.
3. Tích hợp Lưu trữ & Truy vấn Cơ sở dữ liệu SQLite qua REST API (Lưu phiên, tính KPI, xem lịch sử, xóa cascade).
4. Tính chịu lỗi và phục hồi khi mất dấu landmarks (out of frame).
"""

import os
import time
import json
import pytest
import numpy as np
from app import app, global_detector, global_classifier, global_db, global_tts
from core.rep_counter import RepCounter, RepCounterConfig, SquatState
from core.biomechanics import extract_squat_features
from core.pose_detector import LandmarkPoint
from services.llm_agent import LLMAgent
from services.tts_service import TTSService


@pytest.fixture
def client():
    """Tạo test client cho Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_system_health_and_all_routes(client):
    """Kiểm tra phản hồi của toàn bộ các trang giao diện và API health check."""
    # 1. Trang chủ
    res_home = client.get("/")
    assert res_home.status_code == 200
    assert "Virtual Fitness Coach" in res_home.get_data(as_text=True)

    # 2. Trang phòng tập
    res_workout = client.get("/workout")
    assert res_workout.status_code == 200
    assert "Phòng Tập" in res_workout.get_data(as_text=True)

    # 3. Trang lịch sử
    res_history = client.get("/history")
    assert res_history.status_code == 200
    assert "Lịch Sử" in res_history.get_data(as_text=True)

    # 4. Health Check API
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    data = res_health.get_json()
    assert data["status"] == "healthy"
    assert data["detector_ready"] is True
    assert data["classifier_ready"] is True
    assert "correct" in data["classes"]
    assert "knee_valgus" in data["classes"]
    assert "insufficient_depth" in data["classes"]


def test_full_workout_lifecycle_e2e(client):
    """
    Kịch bản Tích hợp Toàn diện (E2E):
    - Chuỗi frames squat hợp lệ 1 rep.
    - Phát hiện lỗi gối hướng vào trong (knee_valgus) kéo dài.
    - LLM Agent đưa ra lời khuyên sửa tư thế.
    - TTS tổng hợp giọng nói tiếng Việt.
    - Lưu phiên tập qua REST API /api/workout/save.
    - Kiểm tra Dashboard API /api/statistics và /api/sessions.
    - Dọn dẹp phiên tập qua REST DELETE.
    """
    # 1. Khởi tạo State Machine đếm Rep độc lập
    counter = RepCounter(
        RepCounterConfig(
            standing_angle=160.0,
            descending_angle=140.0,
            bottom_angle=95.0,
            ascending_angle=115.0,
            standing_complete_angle=155.0,
            smoothing_factor=1.0  # Không làm mượt để kiểm tra chính xác giá trị test
        )
    )

    # Mô phỏng chuỗi đặc trưng của 1 rep squat chuẩn
    # Phase 1: Đứng thẳng (Standing)
    f_stand = {
        "knee_angle_left": 172.0, "knee_angle_right": 171.0,
        "hip_angle_left": 170.0, "hip_angle_right": 169.0,
        "torso_angle": 10.0,
        "knee_alignment_left": 0.02, "knee_alignment_right": 0.02
    }
    r1 = counter.update(f_stand, is_valid_pose=True)
    assert r1.state == SquatState.STANDING

    # Phase 2: Hạ người (Descending)
    f_desc = {
        "knee_angle_left": 125.0, "knee_angle_right": 124.0,
        "hip_angle_left": 120.0, "hip_angle_right": 119.0,
        "torso_angle": 25.0,
        "knee_alignment_left": 0.02, "knee_alignment_right": 0.02
    }
    r2 = counter.update(f_desc, is_valid_pose=True)
    assert r2.state == SquatState.DESCENDING

    # Phase 3: Đáy squat (Bottom) với góc gối 85 độ (đủ sâu <= 95)
    f_bottom = {
        "knee_angle_left": 85.0, "knee_angle_right": 84.0,
        "hip_angle_left": 82.0, "hip_angle_right": 81.0,
        "torso_angle": 32.0,
        "knee_alignment_left": 0.02, "knee_alignment_right": 0.02
    }
    r3 = counter.update(f_bottom, is_valid_pose=True)
    assert r3.state == SquatState.BOTTOM
    assert r3.min_depth_angle <= 85.0

    # Phân loại tư thế tại đáy bằng KNN Model
    clf_res_bottom = global_classifier.classify(f_bottom)
    assert clf_res_bottom.label == "correct"
    assert clf_res_bottom.confidence >= 0.70

    # Phase 4: Đứng lên (Ascending)
    f_asc = {
        "knee_angle_left": 130.0, "knee_angle_right": 129.0,
        "hip_angle_left": 125.0, "hip_angle_right": 124.0,
        "torso_angle": 20.0,
        "knee_alignment_left": 0.02, "knee_alignment_right": 0.02
    }
    r4 = counter.update(f_asc, is_valid_pose=True)
    assert r4.state == SquatState.ASCENDING

    # Phase 5: Trở về đứng thẳng -> Hoàn thành 1 Rep
    r5 = counter.update(f_stand, is_valid_pose=True)
    assert r5.state == SquatState.STANDING
    assert r5.rep_count == 1
    assert r5.rep_just_completed is True

    # 2. Mô phỏng tư thế lỗi Knee Valgus kéo dài
    f_valgus = {
        "knee_angle_left": 86.0, "knee_angle_right": 85.0,
        "hip_angle_left": 84.0, "hip_angle_right": 83.0,
        "torso_angle": 30.0,
        "knee_alignment_left": 0.28, "knee_alignment_right": 0.29  # Gối lệch mạnh vào trong
    }
    clf_valgus = global_classifier.classify(f_valgus)
    assert clf_valgus.label == "knee_valgus"
    assert clf_valgus.error_code == "ERR_VALGUS"

    # LLM Agent bắt lỗi kéo dài > 0.5 giây
    agent = LLMAgent(api_key="offline", min_error_duration_sec=0.5, cooldown_sec=2.0)
    # Lần 1: Lỗi mới xuất hiện lúc t=10.0s -> Chưa đủ thời gian
    adv1 = agent.process_error(
        error_code="ERR_VALGUS",
        error_description=clf_valgus.description_vi,
        confidence=clf_valgus.confidence,
        rep_count=1,
        timestamp=10.0
    )
    assert adv1 is None

    # Lần 2: Lỗi duy trì tới t=10.6s (> 0.5s) -> Kích hoạt LLM (hoặc Fallback)
    agent.process_error(
        error_code="ERR_VALGUS",
        error_description=clf_valgus.description_vi,
        confidence=clf_valgus.confidence,
        rep_count=1,
        timestamp=10.6
    )
    for _ in range(20):
        if agent.latest_advice:
            break
        time.sleep(0.05)

    adv2 = agent.latest_advice
    assert adv2 is not None
    assert "đầu gối" in adv2.lower()

    # 3. Tổng hợp giọng nói TTS
    tts_url = global_tts.synthesize_blocking_for_test(adv2)
    assert tts_url is not None
    assert tts_url.startswith("/static/audios/")
    local_audio_path = os.path.join(global_tts.cache_dir, os.path.basename(tts_url))
    assert os.path.exists(local_audio_path)

    # 4. Lưu phiên tập vào SQLite qua REST API /api/workout/save
    workout_payload = {
        "exercise_type": "squat",
        "total_reps": 1,
        "correct_reps": 0,
        "duration_seconds": 35,
        "errors": [
            {
                "error_code": "ERR_VALGUS",
                "error_description": clf_valgus.description_vi,
                "duration_seconds": 1.5,
                "feedback_text": adv2,
                "audio_path": tts_url
            }
        ]
    }

    save_res = client.post(
        "/api/workout/save",
        data=json.dumps(workout_payload),
        content_type="application/json"
    )
    assert save_res.status_code == 201
    save_data = save_res.get_json()
    assert save_data["success"] is True
    session_id = save_data["session_id"]
    assert session_id > 0

    # 5. Kiểm tra Dashboard APIs
    # Chi tiết buổi tập
    det_res = client.get(f"/api/sessions/{session_id}")
    assert det_res.status_code == 200
    det_data = det_res.get_json()["session"]
    assert det_data["total_reps"] == 1
    assert det_data["correct_reps"] == 0
    assert len(det_data["errors"]) == 1
    assert det_data["errors"][0]["error_code"] == "ERR_VALGUS"
    assert det_data["errors"][0]["feedback_text"] == adv2

    # Thống kê tổng hợp
    stats_res = client.get("/api/statistics")
    assert stats_res.status_code == 200
    stats_data = stats_res.get_json()["statistics"]
    assert stats_data["total_sessions"] >= 1
    assert stats_data["total_reps"] >= 1

    # 6. Dọn dẹp phiên tập vừa tạo
    del_res = client.delete(f"/api/sessions/{session_id}")
    assert del_res.status_code == 200
    del_data = del_res.get_json()
    assert del_data["success"] is True


def test_out_of_frame_robustness():
    """Kiểm tra hệ thống không bị đếm sai hoặc crash khi người dùng ra khỏi khung hình."""
    counter = RepCounter(RepCounterConfig())
    
    # 1. Người dùng đang ở trạng thái hạ người (DESCENDING)
    f_desc = {
        "knee_angle_left": 130.0, "knee_angle_right": 130.0,
        "hip_angle_left": 125.0, "hip_angle_right": 125.0,
        "torso_angle": 20.0,
        "knee_alignment_left": 0.02, "knee_alignment_right": 0.02
    }
    counter.update(f_desc, is_valid_pose=True)
    assert counter.state == SquatState.DESCENDING

    # 2. Người dùng bị khuất / bước ra ngoài camera (is_valid_pose=False) liên tục
    for _ in range(10):
        res = counter.update(features=None, is_valid_pose=False)

    # 3. State machine phải giữ an toàn hoặc reset về STANDING sau timeout mà không cộng rep sai
    assert res.rep_count == 0
    assert res.rep_just_completed is False
