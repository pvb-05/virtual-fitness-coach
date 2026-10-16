"""
Unit Tests cho Database Module (Giai đoạn 7).
Kiểm tra:
- Khởi tạo SQLite database & thực thi schema.sql
- Tạo và kết thúc buổi tập (workout_sessions)
- Lưu trọn vẹn session kèm danh sách lỗi tư thế (form_errors)
- Truy vấn danh sách và chi tiết buổi tập
- Tính toán số liệu thống kê Dashboard (Tổng rep, thời gian, phân bố lỗi)
- Xóa buổi tập và tính toàn vẹn khóa ngoại (Cascade Delete)
"""

import os
import pytest
from database.db_manager import DatabaseManager


@pytest.fixture
def db(tmp_path):
    """Tạo DatabaseManager trên file SQLite tạm cho từng test case."""
    db_file = os.path.join(tmp_path, "test_fitness.db")
    manager = DatabaseManager(db_path=db_file)
    return manager


def test_db_initialization(db):
    """Kiểm tra database được tạo và user mặc định tồn tại."""
    assert os.path.exists(db.db_path)
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users WHERE id = 1;")
        row = cursor.fetchone()
        assert row is not None
        assert row["username"] == "Vận động viên"


def test_create_and_end_session(db):
    """Kiểm tra tạo mới và kết thúc cập nhật buổi tập."""
    session_id = db.create_session(exercise_type="squat", start_time="2026-10-17 08:00:00")
    assert session_id > 0

    success = db.end_session(
        session_id=session_id,
        total_reps=15,
        correct_reps=12,
        duration_seconds=300,
        end_time="2026-10-17 08:05:00"
    )
    assert success is True

    details = db.get_session_details(session_id)
    assert details is not None
    assert details["total_reps"] == 15
    assert details["correct_reps"] == 12
    assert details["duration_seconds"] == 300
    assert details["exercise_type"] == "squat"


def test_save_full_session_with_errors(db):
    """Kiểm tra lưu toàn diện buổi tập kèm danh sách lỗi tư thế."""
    session_data = {
        "exercise_type": "squat",
        "start_time": "2026-10-17 09:00:00",
        "end_time": "2026-10-17 09:10:00",
        "duration_seconds": 600,
        "total_reps": 20,
        "correct_reps": 16,
        "errors": [
            {
                "error_code": "ERR_DEPTH",
                "error_description": "Chưa đủ độ sâu",
                "duration_seconds": 1.5,
                "feedback_text": "Hãy hạ thấp hông sâu hơn",
                "audio_path": "/static/audios/depth.mp3"
            },
            {
                "error_code": "ERR_VALGUS",
                "error_description": "Đầu gối chụm vào trong",
                "duration_seconds": 2.0,
                "feedback_text": "Mở rộng gối ra ngoài",
                "audio_path": "/static/audios/valgus.mp3"
            }
        ]
    }

    session_id = db.save_full_session(session_data)
    assert session_id > 0

    details = db.get_session_details(session_id)
    assert details is not None
    assert details["total_reps"] == 20
    assert len(details["errors"]) == 2
    assert details["errors"][0]["error_code"] == "ERR_DEPTH"
    assert details["errors"][1]["error_code"] == "ERR_VALGUS"


def test_get_workout_statistics(db):
    """Kiểm tra tính toán số liệu thống kê Dashboard."""
    # Tạo 2 buổi tập mẫu
    db.save_full_session({
        "exercise_type": "squat", "duration_seconds": 600,
        "total_reps": 20, "correct_reps": 15,
        "errors": [{"error_code": "ERR_DEPTH"}, {"error_code": "ERR_DEPTH"}]
    })
    db.save_full_session({
        "exercise_type": "squat", "duration_seconds": 300,
        "total_reps": 10, "correct_reps": 8,
        "errors": [{"error_code": "ERR_VALGUS"}]
    })

    stats = db.get_workout_statistics()
    assert stats["total_sessions"] == 2
    assert stats["total_reps"] == 30
    assert stats["total_correct_reps"] == 23
    assert stats["total_duration_seconds"] == 900
    assert stats["total_duration_minutes"] == 15.0
    assert stats["accuracy_rate"] == pytest.approx(76.7, abs=0.1)

    # Phân bố lỗi
    dist = stats["error_distribution"]
    assert dist.get("ERR_DEPTH") == 2
    assert dist.get("ERR_VALGUS") == 1


def test_delete_session_and_cascading(db):
    """Kiểm tra xóa buổi tập thì toàn bộ lỗi của buổi đó bị xóa theo (Foreign key cascade)."""
    session_id = db.save_full_session({
        "exercise_type": "squat", "total_reps": 5,
        "errors": [{"error_code": "ERR_DEPTH"}]
    })

    assert db.get_session_details(session_id) is not None

    deleted = db.delete_session(session_id)
    assert deleted is True
    assert db.get_session_details(session_id) is None

    # Kiểm tra bảng form_errors không còn lỗi mồ côi
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM form_errors WHERE session_id = ?;", (session_id,))
        assert cursor.fetchone()["cnt"] == 0
