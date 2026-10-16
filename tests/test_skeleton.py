"""
Unit Tests cho Giai đoạn 1 — Skeleton dự án.
Kiểm tra:
- Khởi tạo Flask App
- Các Router: Trang chủ (/), Workout (/workout), History (/history)
- Endpoint Health Check (/api/health)
- Khả năng nạp các thư viện CV, ML, Audio
"""

import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """Kiểm tra trang chủ tải thành công và có chứa tiêu đề dự án."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Virtual Fitness Coach" in response.data
    assert b"Squat" in response.data


def test_workout_route(client):
    """Kiểm tra trang phòng tập tải thành công."""
    response = client.get("/workout")
    assert response.status_code == 200


def test_history_route(client):
    """Kiểm tra trang lịch sử tải thành công."""
    response = client.get("/history")
    assert response.status_code == 200


def test_health_check_endpoint(client):
    """Kiểm tra API health check trả về JSON với trạng thái healthy."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    assert data["status"] == "healthy"
    assert data["opencv_version"] is not None
    assert data["mediapipe_version"] is not None
    assert data["sklearn_version"] is not None
    assert data["websocket_support"] is True


def test_404_error_handler(client):
    """Kiểm tra trang không tồn tại trả về lỗi 404 JSON."""
    response = client.get("/trang-khong-ton-tai")
    assert response.status_code == 404
    data = response.get_json()
    assert "error" in data
