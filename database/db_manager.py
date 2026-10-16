"""
Database Module: SQLite Manager
Quản lý kết nối, khởi tạo schema và thực hiện các thao tác CRUD an toàn với SQLite:
- Lưu trữ buổi tập (workout_sessions)
- Ghi nhận nhật ký lỗi tư thế (form_errors)
- Truy vấn thống kê dashboard tổng quan
- Sử dụng Parameterized Queries tuyệt đối chống tấn công SQL Injection
"""

import os
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger("DatabaseManager")

DEFAULT_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "fitness_coach.db"))
SCHEMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "schema.sql"))


class DatabaseManager:
    """Quản lý cơ sở dữ liệu SQLite cho Virtual Fitness Coach."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("DATABASE_PATH", DEFAULT_DB_PATH)
        # Đảm bảo đường dẫn tuyệt đối
        self.db_path = os.path.abspath(self.db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Tạo kết nối SQLite mới với cấu hình hỗ trợ Row dictionary và Foreign Keys."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self):
        """Khởi tạo các bảng từ schema.sql và chèn người dùng mặc định nếu chưa có."""
        try:
            if not os.path.exists(SCHEMA_PATH):
                logger.error(f"Không tìm thấy file schema tại: {SCHEMA_PATH}")
                return

            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                schema_sql = f.read()

            with self.get_connection() as conn:
                conn.executescript(schema_sql)
                # Thêm user mặc định
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM users WHERE id = 1;")
                if not cursor.fetchone():
                    cursor.execute(
                        "INSERT OR IGNORE INTO users (id, username) VALUES (1, 'Vận động viên');"
                    )
                conn.commit()
            logger.info(f"Cơ sở dữ liệu SQLite đã sẵn sàng tại: {self.db_path}")

        except Exception as e:
            logger.error(f"Lỗi khởi tạo cơ sở dữ liệu: {e}")

    # =========================================================================
    # QUẢN LÝ BUỔI TẬP (WORKOUT SESSIONS)
    # =========================================================================

    def create_session(
        self,
        exercise_type: str = "squat",
        start_time: Optional[str] = None,
        user_id: int = 1
    ) -> int:
        """Tạo mới một buổi tập và trả về session_id."""
        start_iso = start_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query = """
            INSERT INTO workout_sessions (user_id, exercise_type, start_time, total_reps, correct_reps)
            VALUES (?, ?, ?, 0, 0);
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (user_id, exercise_type, start_iso))
            conn.commit()
            return cursor.lastrowid

    def end_session(
        self,
        session_id: int,
        total_reps: int,
        correct_reps: int = 0,
        duration_seconds: int = 0,
        end_time: Optional[str] = None
    ) -> bool:
        """Cập nhật kết thúc buổi tập với số rep và thời lượng hoàn thành."""
        end_iso = end_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query = """
            UPDATE workout_sessions
            SET end_time = ?, duration_seconds = ?, total_reps = ?, correct_reps = ?
            WHERE id = ?;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (end_iso, duration_seconds, total_reps, correct_reps, session_id))
            conn.commit()
            return cursor.rowcount > 0

    def save_full_session(self, session_data: Dict[str, Any]) -> int:
        """
        Lưu trọn vẹn một buổi tập cùng danh sách lỗi tư thế gửi từ giao diện web.
        Thực hiện toàn bộ trong 1 transaction an toàn.
        """
        exercise_type = session_data.get("exercise_type", "squat")
        start_time = session_data.get("start_time") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        end_time = session_data.get("end_time") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration_seconds = int(session_data.get("duration_seconds", 0))
        total_reps = int(session_data.get("total_reps", 0))
        correct_reps = int(session_data.get("correct_reps", max(0, total_reps)))
        errors = session_data.get("errors", [])

        with self.get_connection() as conn:
            cursor = conn.cursor()
            # 1. Chèn session
            cursor.execute(
                """
                INSERT INTO workout_sessions (user_id, exercise_type, start_time, end_time, duration_seconds, total_reps, correct_reps)
                VALUES (1, ?, ?, ?, ?, ?, ?);
                """,
                (exercise_type, start_time, end_time, duration_seconds, total_reps, correct_reps)
            )
            session_id = cursor.lastrowid

            # 2. Chèn danh sách lỗi
            for err in errors:
                cursor.execute(
                    """
                    INSERT INTO form_errors (session_id, error_code, error_description, timestamp_occurred, duration_seconds, feedback_text, audio_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        session_id,
                        err.get("error_code", "ERR_UNKNOWN"),
                        err.get("error_description", ""),
                        err.get("timestamp_occurred") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        float(err.get("duration_seconds", 0.0)),
                        err.get("feedback_text", ""),
                        err.get("audio_path", "")
                    )
                )
            conn.commit()
            logger.info(f"Đã lưu thành công buổi tập #{session_id} ({total_reps} reps, {len(errors)} lỗi)")
            return session_id

    # =========================================================================
    # QUẢN LÝ LỖI TƯ THẾ (FORM ERRORS)
    # =========================================================================

    def log_form_error(
        self,
        session_id: int,
        error_code: str,
        error_description: str,
        duration_seconds: float = 0.0,
        feedback_text: str = "",
        audio_path: str = ""
    ) -> int:
        """Ghi nhận một lỗi tư thế vào cơ sở dữ liệu."""
        now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query = """
            INSERT INTO form_errors (session_id, error_code, error_description, timestamp_occurred, duration_seconds, feedback_text, audio_path)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                query,
                (session_id, error_code, error_description, now_iso, duration_seconds, feedback_text, audio_path)
            )
            conn.commit()
            return cursor.lastrowid

    # =========================================================================
    # TRUY VẤN VÀ THỐNG KÊ (QUERIES & ANALYTICS)
    # =========================================================================

    def get_all_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lấy danh sách các buổi tập gần nhất kèm số lượng lỗi phát hiện."""
        query = """
            SELECT 
                s.id, s.exercise_type, s.start_time, s.end_time,
                s.duration_seconds, s.total_reps, s.correct_reps, s.created_at,
                COUNT(e.id) AS error_count
            FROM workout_sessions s
            LEFT JOIN form_errors e ON s.id = e.session_id
            GROUP BY s.id
            ORDER BY s.id DESC
            LIMIT ?;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_session_details(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Lấy chi tiết buổi tập và danh sách toàn bộ các lỗi đã ghi nhận trong buổi đó."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM workout_sessions WHERE id = ?;", (session_id,))
            session_row = cursor.fetchone()
            if not session_row:
                return None

            result = dict(session_row)

            # Lấy danh sách lỗi
            cursor.execute(
                """
                SELECT id, error_code, error_description, timestamp_occurred, duration_seconds, feedback_text, audio_path
                FROM form_errors
                WHERE session_id = ?
                ORDER BY id ASC;
                """,
                (session_id,)
            )
            result["errors"] = [dict(r) for r in cursor.fetchall()]
            return result

    def get_workout_statistics(self) -> Dict[str, Any]:
        """Tính toán tổng hợp thống kê cho Dashboard: tổng rep, tổng giờ tập, phân bố lỗi."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Thống kê tổng quan
            cursor.execute("""
                SELECT 
                    COUNT(id) AS total_sessions,
                    COALESCE(SUM(total_reps), 0) AS total_reps,
                    COALESCE(SUM(correct_reps), 0) AS total_correct_reps,
                    COALESCE(SUM(duration_seconds), 0) AS total_duration_seconds
                FROM workout_sessions;
            """)
            overall = dict(cursor.fetchone())

            # 2. Phân bố các mã lỗi
            cursor.execute("""
                SELECT error_code, COUNT(id) AS count
                FROM form_errors
                GROUP BY error_code
                ORDER BY count DESC;
            """)
            error_dist = {row["error_code"]: row["count"] for row in cursor.fetchall()}

            # 3. Tính tỷ lệ chuẩn xác ước tính
            total_reps = overall["total_reps"]
            total_correct = overall["total_correct_reps"]
            accuracy_rate = round((total_correct / total_reps * 100), 1) if total_reps > 0 else 100.0

            total_minutes = round(overall["total_duration_seconds"] / 60.0, 1)

            return {
                "total_sessions": overall["total_sessions"],
                "total_reps": total_reps,
                "total_correct_reps": total_correct,
                "total_duration_seconds": overall["total_duration_seconds"],
                "total_duration_minutes": total_minutes,
                "accuracy_rate": accuracy_rate,
                "error_distribution": error_dist
            }

    def delete_session(self, session_id: int) -> bool:
        """Xóa một buổi tập và các lỗi liên kết (cascade delete)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM workout_sessions WHERE id = ?;", (session_id,))
            conn.commit()
            return cursor.rowcount > 0
